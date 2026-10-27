"""验证 SenseVoice 自带的 <|nospeech|> 判定能力 + 混合场景下的 voicing 表现。

三个实验：
  E1. language=lang_nospeech 时模型输出什么？能否用它做「有无语音」判定？
  E2. 语音 + 噪声混合时 voicing 特征的衰减情况（真实使用场景）
  E3. 组合判据（voicing + 文本长度）的误杀率/漏杀率
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
LANG_AUTO = int(meta["lang_auto"])
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
    out, prev, events = [], -1, []
    for i in ids:
        if i != prev and i != 0:
            tok = tokens.get(int(i), "")
            if tok.startswith("<|") and tok.endswith("|>"):
                events.append(tok)
            else:
                out.append(tok)
        prev = i
    text = re.sub(r"<\|[^|]*\|>", "", "".join(out).replace("▁", " ")).strip()
    return text, events


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


rng = np.random.default_rng(7)

# ============================================================ E1
print("=" * 96)
print("E1：language=lang_nospeech(%d) / auto(%d) / zh(%d) 对噪声输入的输出对比"
      % (LANG_NOSPEECH, LANG_AUTO, LANG_ZH))
print("=" * 96)
N = 16000 * 3
noise_cases = {
    "纯白噪 -30dB": rng.normal(0, 10 ** (-30 / 20), N),
    "纯白噪 -50dB": rng.normal(0, 10 ** (-50 / 20), N),
    "全零静音": np.zeros(N),
    "键盘+空调": rng.normal(0, 10 ** (-40 / 20), N),
}
speech = load("test-audio/zh_16k.wav")

print(f"{'样本':<20}{'nospeech':>26}{'auto':>26}{'zh':>26}")
for name, x in noise_cases.items():
    xf = x.astype(np.float32)
    r = []
    for lang in (LANG_NOSPEECH, LANG_AUTO, LANG_ZH):
        t, ev = run(xf, lang)
        r.append(f"[{t}]{'|NOSPEECH' if any('nospeech' in e for e in ev) else ''}")
    print(f"{name:<20}" + "".join(f"{s:>26}" for s in r))
t, ev = run(speech, LANG_NOSPEECH)
print(f"{'真语音(用nospeech跑)':<20}{'[' + t + ']':>26}")
t, ev = run(speech, LANG_AUTO)
print(f"{'真语音(auto)':<20}{'[' + t + ']':>26}  事件={ev}")

# ============================================================ E2
print()
print("=" * 96)
print("E2：语音 + 噪声混合时 voicing 的衰减（真实使用场景）")
print("=" * 96)
print(f"{'SNR':>6}{'voicing':>10}{'CER':>8}   识别结果")
REFERENCE = "大家好，今天天气不错，我们现在测试一下实时语音识别系统，看看断句和标点准不准。"


def cer(ref, hyp):
    strip = lambda s: re.sub(r"[，。！？、,.!?；;：:\s]", "", s)
    a, b = strip(ref), strip(hyp)
    if not a:
        return 0.0 if not b else 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / len(a)


for snr in [30, 20, 15, 10, 5, 0, -5]:
    noise = rng.normal(0, 1, len(speech))
    sp = np.sqrt(np.mean(speech ** 2))
    nz = np.sqrt(np.mean(noise ** 2))
    mixed = (speech + noise * (sp / nz) * 10 ** (-snr / 20)).astype(np.float32)
    v = voicing_of(mixed)
    t, ev = run(mixed)
    print(f"{snr:>5}dB{v:>10.3f}{cer(REFERENCE, t):>8.3f}   [{t}]")

# ============================================================ E3
print()
print("=" * 96)
print("E3：组合判据扫描 —— 目标：真语音全部保留，噪声全部拒绝")
print("=" * 96)
pos_samples = []
for path in sorted(glob.glob("test-audio/zh_16k*.wav")):
    name = os.path.basename(path).replace(".wav", "")
    x = load(path)
    t, _ = run(x)
    pos_samples.append((name, voicing_of(x), len(t), t))

neg_samples = []
for name, x in noise_cases.items():
    xf = x.astype(np.float32)
    t, _ = run(xf)
    neg_samples.append((name, voicing_of(xf), len(t), t))
for dur_ms in [320, 600, 1000, 1500]:
    seg = rng.normal(0, 10 ** (-35 / 20), 16 * dur_ms).astype(np.float32)
    t, _ = run(seg)
    neg_samples.append((f"短噪声{dur_ms}ms", voicing_of(seg), len(t), t))
# 加几个 SNR 极低的语音（应当保留，宁可漏杀不可误杀）
for snr in [0, -5]:
    noise = rng.normal(0, 1, len(speech))
    sp = np.sqrt(np.mean(speech ** 2))
    nz = np.sqrt(np.mean(noise ** 2))
    mixed = (speech + noise * (sp / nz) * 10 ** (-snr / 20)).astype(np.float32)
    t, _ = run(mixed)
    pos_samples.append((f"语音{snr}dB", voicing_of(mixed), len(t), t))

print(f"{'样本':<20}{'voicing':>9}{'字数':>6}  {'文本'}")
print("-" * 96)
print("正样本（应保留）：")
for name, v, n, t in pos_samples:
    print(f"{name:<20}{v:>9.3f}{n:>6}  {t}")
print("-" * 96)
print("负样本（应拒绝）：")
for name, v, n, t in neg_samples:
    print(f"{name:<20}{v:>9.3f}{n:>6}  [{t}]")
print("-" * 96)

print("阈值扫描：voicing >= T 且 字数 >= L 才接受")
print(f"{'voicing阈值':>12}{'字数阈值':>10}{'正样本保留':>12}{'负样本误放':>12}")
for vt in [0.15, 0.18, 0.20, 0.22, 0.25, 0.30]:
    for ln in [2, 3, 4, 5]:
        keep = sum(1 for _, v, n, _ in pos_samples if v >= vt and n >= ln)
        leak = sum(1 for _, v, n, _ in neg_samples if v >= vt and n >= ln)
        mark = "  <== 最优" if keep == len(pos_samples) and leak == 0 else ""
        print(f"{vt:>12.2f}{ln:>10}{keep:>8}/{len(pos_samples)}{leak:>8}/{len(neg_samples)}{mark}")
