"""对抗性噪声测试：人群嘈杂 / 音乐 / 电视 / 婴儿哭 —— 这些是字幕产品真正的噩梦。

对每类噪声测：
  1. 正常 zh 通道识别出什么（会不会吐出长句幻觉）
  2. voicing 特征值
  3. language=lang_nospeech 通道是否给出 NOSPEECH 标记
  4. 两条判据的组合效果
"""
import glob
import os
import re
import wave

import numpy as np
import onnxruntime as ort

ROOT = "D:/Audivox事务/Hearizon Sync手机版"
MODEL = f"{ROOT}/app/src/main/assets/models/sv/model.int8.onnx"
TOKENS = f"{ROOT}/app/src/main/assets/models/sv/tokens.txt"

sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
meta = sess.get_modelmeta().custom_metadata_map
LFR_M = int(meta["lfr_window_size"])
LFR_N = int(meta["lfr_window_shift"])
NEG = np.fromstring(meta["neg_mean"], sep=",")
INV = np.fromstring(meta["inv_stddev"], sep=",")
LANG_NOSPEECH = int(meta["lang_nospeech"])
LANG_ZH = int(meta["lang_zh"])
tokens = {}
for line in open(TOKENS, encoding="utf-8"):
    line = line.rstrip("\n")
    if line:
        i = line.rfind(" ")
        tokens[int(line[i + 1:])] = line[:i]

WIN, HOP, NFFT = 400, 160, 512


def load(path):
    with wave.open(path, "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def fbank(x, sr=16000):
    n = len(x)
    nf = 1 + (n - WIN) // HOP
    idx = np.arange(WIN)[None, :] + HOP * np.arange(nf)[:, None]
    fr = x[idx].astype(np.float64)
    fr = fr - fr.mean(axis=1, keepdims=True)
    fr = np.append(fr[:, 0:1], fr[:, 1:] - 0.97 * fr[:, :-1], axis=1)
    fr = fr * np.hamming(WIN)
    pad = np.zeros((nf, NFFT))
    pad[:, :WIN] = fr
    spec = np.fft.rfft(pad, n=NFFT)
    p = spec.real ** 2 + spec.imag ** 2

    def hz2mel(f):
        return 1127 * np.log(1 + f / 700)

    def mel2hz(m):
        return 700 * (np.exp(m / 1127) - 1)

    mels = np.linspace(hz2mel(20), hz2mel(sr / 2 - 400), 82)
    bins = np.floor((NFFT + 1) * mel2hz(mels) / sr).astype(int)
    fb = np.zeros((80, NFFT // 2 + 1))
    for m in range(1, 81):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c):
            if c > l:
                fb[m - 1, k] = (k - l) / (c - l)
        for k in range(c, r):
            if r > c:
                fb[m - 1, k] = (r - k) / (r - c)
    return np.log(np.maximum(p @ fb.T, 1.1920929e-07))


def lfr(f, m, n):
    t = f.shape[0]
    left = (m - 1) // 2
    padded = np.concatenate([np.repeat(f[0:1], left, axis=0), f], axis=0)
    tl, dim = int(np.ceil(t / n)), f.shape[1]
    out = np.zeros((tl, dim * m))
    for i in range(tl):
        s = i * n
        if s + m <= padded.shape[0]:
            out[i] = padded[s:s + m].reshape(-1)
        else:
            flat = padded[s:].reshape(-1)
            out[i, :flat.size] = flat
            last, k, j = padded[-1], flat.size, 0
            while k < dim * m:
                out[i, k] = last[j % dim]
                k += 1
                j += 1
    return out


def run(x, language=LANG_ZH, text_norm=14):
    cm = (lfr(fbank(x), LFR_M, LFR_N) + NEG) * INV
    logits = sess.run(None, {
        "x": cm[None, :, :].astype(np.float32),
        "x_length": np.array([cm.shape[0]], dtype=np.int32),
        "language": np.array([language], dtype=np.int32),
        "text_norm": np.array([text_norm], dtype=np.int32),
    })[0][0]
    ids = logits.argmax(axis=1)
    out, prev, events = [], -1, set()
    for i in ids:
        if i != prev and i != 0:
            tok = tokens.get(int(i), "")
            if tok.startswith("<|") and tok.endswith("|>"):
                events.add(tok)
            else:
                out.append(tok)
        prev = i
    return re.sub(r"<\|[^|]*\|>", "", "".join(out).replace("▁", " ")).strip(), events


def voicing_of(x):
    n = 1 + (len(x) - WIN) // HOP if len(x) >= WIN else 0
    if n <= 0:
        return 0.0
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    fr = x[idx]
    rms_all = np.sqrt(np.mean(fr.astype(np.float64) ** 2, axis=1))
    thr = max(np.percentile(rms_all, 30), 1e-6)
    active = fr[rms_all > thr]
    if active.shape[0] < 2:
        return 0.0
    ac = []
    for f in active[:64]:
        f = f - f.mean()
        c = np.correlate(f, f, mode="full")[len(f) - 1:]
        if c[0] <= 1e-12:
            continue
        c = c / c[0]
        lo, hi = 40, min(267, len(c) - 1)
        if hi > lo:
            ac.append(float(np.max(c[lo:hi])))
    return float(np.mean(ac)) if ac else 0.0


rng = np.random.default_rng(11)
SR = 16000
speech = load("test-audio/zh_16k.wav")
DUR = 3
N = SR * DUR


def music(n, sr=SR):
    """合成音乐：和弦 + 节拍，有明显的谐波结构（voicing 会高）。"""
    t = np.arange(n) / sr
    out = np.zeros(n)
    for f0, amp in [(220, 0.3), (277.2, 0.25), (330, 0.2), (440, 0.15)]:
        for h in range(1, 6):
            out += amp / h * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6.28))
    # 节拍包络
    beat = (np.sin(2 * np.pi * 2.0 * t) > 0.6).astype(float)
    beat = np.convolve(beat, np.exp(-np.arange(200) / 50), mode="same")
    out *= (0.3 + 0.7 * beat)
    return out / np.max(np.abs(out)) * 0.3


def babble(n, sr=SR, voices=6):
    """合成人群嘈杂：多个不重叠的类语音调制噪声叠加。"""
    out = np.zeros(n)
    for _ in range(voices):
        base = rng.normal(0, 1, n)
        # 用随机共振峰近似语音频谱
        f = rng.uniform(150, 400)
        t = np.arange(n) / sr
        mod = (np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) * 0.5 + 0.5)
        # 音节包络
        syl = np.maximum(np.sin(2 * np.pi * rng.uniform(2.5, 5) * t), 0) ** 2
        out += base * mod * syl * 0.15
    return out / max(np.max(np.abs(out)), 1e-9) * 0.25


def tv(n, sr=SR):
    """合成电视对白：类语音但无意义（共振峰扫频）。"""
    t = np.arange(n) / sr
    out = np.zeros(n)
    for _ in range(3):
        f0 = rng.uniform(90, 200)
        fm = np.cumsum(rng.normal(0, 0.3, n))
        ph = 2 * np.pi * (f0 * t + fm / sr * 50)
        out += np.sin(ph) * 0.2
        out += rng.normal(0, 0.03, n)
    syl = np.maximum(np.sin(2 * np.pi * 3.5 * t), 0) ** 1.5
    return out * syl / max(np.max(np.abs(out * syl)), 1e-9) * 0.25


def baby(n, sr=SR):
    """合成婴儿哭：高频强调制。"""
    t = np.arange(n) / sr
    f = 400 + 300 * np.sin(2 * np.pi * 1.2 * t)
    out = np.sin(2 * np.pi * np.cumsum(f) / sr) * 0.3
    out += rng.normal(0, 0.05, n)
    env = np.maximum(np.sin(2 * np.pi * 0.8 * t), 0) ** 0.5
    return out * env / max(np.max(np.abs(out * env)), 1e-9) * 0.3


adversarial = {
    "合成音乐(和弦)": music(N),
    "合成人群babble": babble(N),
    "合成电视对白": tv(N),
    "合成婴儿哭": baby(N),
    "真实语音(对照)": speech[:N] if len(speech) >= N else np.pad(speech, (0, N - len(speech))),
}

print("=" * 118)
print("对抗性噪声：会不会吐出长句幻觉？")
print("=" * 118)
print(f"{'场景':<18}{'voicing':>9}{'字数':>6}  {'zh通道识别结果':<52}{'nospeech通道'}")
print("-" * 118)
rows = []
for name, x in adversarial.items():
    xf = x.astype(np.float32)
    t_zh, ev_zh = run(xf, LANG_ZH)
    t_ns, ev_ns = run(xf, LANG_NOSPEECH)
    v = voicing_of(xf)
    has_ns = any("nospeech" in e for e in ev_ns)
    rows.append((name, v, len(t_zh), t_zh, has_ns))
    disp = t_zh if len(t_zh) <= 48 else t_zh[:48] + "…"
    print(f"{name:<18}{v:>9.3f}{len(t_zh):>6}  {disp:<52}{'NOSPEECH' if has_ns else '有语音'}")

print("=" * 118)
print("判据效果：")
print("  判据A = voicing < 0.20  → 拒绝")
print("  判据B = nospeech通道出现 NOSPEECH 标记 → 拒绝")
print("  判据C = A 或 B")
for name, v, n, t, has_ns in rows:
    a = v < 0.20
    b = has_ns
    c = a or b
    truth = "语音" if "对照" in name else "噪声"
    ok = (truth == "噪声" and c) or (truth == "语音" and not c)
    print(f"{name:<18} 真值={truth}  A={a!s:<5} B={b!s:<5} C={c!s:<5} {'✅' if ok else '❌ 误判'}")
