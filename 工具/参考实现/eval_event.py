"""验证 SenseVoice 自带的音频事件分类（<|Speech|> / <|BGM|> / <|Cry|> ...）能否当语音检测器。

关键认知纠正：lang_nospeech 只是把输入的 language token 回显成输出第一个 token，不是检测器。
真正的检测信号是模型输出的音频事件 token 集合。
"""
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
LANG_ZH = int(meta["lang_zh"])
tokens = {}
for line in open(TOKENS, encoding="utf-8"):
    line = line.rstrip("\n")
    if line:
        i = line.rfind(" ")
        tokens[int(line[i + 1:])] = line[:i]

# 把全部事件类 token 列出来
EVENT_TOKENS = sorted({v for v in tokens.values() if v.startswith("<|") and v.endswith("|>")})
print("模型全部特殊/事件 token：")
print("  " + "  ".join(EVENT_TOKENS))
print()

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
    """返回 (文本, 事件集合, 每个特殊 token 的出现帧数)。"""
    cm = (lfr(fbank(x), LFR_M, LFR_N) + NEG) * INV
    logits = sess.run(None, {
        "x": cm[None, :, :].astype(np.float32),
        "x_length": np.array([cm.shape[0]], dtype=np.int32),
        "language": np.array([language], dtype=np.int32),
        "text_norm": np.array([text_norm], dtype=np.int32),
    })[0][0]
    ids = logits.argmax(axis=1)
    out, prev = [], -1
    special_count = {}
    for i in ids:
        tok = tokens.get(int(i), "")
        if tok.startswith("<|") and tok.endswith("|>"):
            special_count[tok] = special_count.get(tok, 0) + 1
        elif i != prev and i != 0:
            out.append(tok)
        prev = i
    text = re.sub(r"<\|[^|]*\|>", "", "".join(out).replace("▁", " ")).strip()
    return text, special_count, len(ids)


rng = np.random.default_rng(11)
SR = 16000
speech_full = load("test-audio/zh_16k.wav")
N = SR * 3


def music(n, sr=SR):
    t = np.arange(n) / sr
    out = np.zeros(n)
    for f0, amp in [(220, 0.3), (277.2, 0.25), (330, 0.2), (440, 0.15)]:
        for h in range(1, 6):
            out += amp / h * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6.28))
    beat = (np.sin(2 * np.pi * 2.0 * t) > 0.6).astype(float)
    beat = np.convolve(beat, np.exp(-np.arange(200) / 50), mode="same")
    out *= (0.3 + 0.7 * beat)
    return out / np.max(np.abs(out)) * 0.3


def babble(n, sr=SR, voices=6):
    out = np.zeros(n)
    t = np.arange(n) / sr
    for _ in range(voices):
        base = rng.normal(0, 1, n)
        f = rng.uniform(150, 400)
        mod = (np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) * 0.5 + 0.5)
        syl = np.maximum(np.sin(2 * np.pi * rng.uniform(2.5, 5) * t), 0) ** 2
        out += base * mod * syl * 0.15
    return out / max(np.max(np.abs(out)), 1e-9) * 0.25


def tv(n, sr=SR):
    t = np.arange(n) / sr
    out = np.zeros(n)
    for _ in range(3):
        f0 = rng.uniform(90, 200)
        fm = np.cumsum(rng.normal(0, 0.3, n))
        ph = 2 * np.pi * (f0 * t + fm / sr * 50)
        out += np.sin(ph) * 0.2 + rng.normal(0, 0.03, n)
    syl = np.maximum(np.sin(2 * np.pi * 3.5 * t), 0) ** 1.5
    return out * syl / max(np.max(np.abs(out * syl)), 1e-9) * 0.25


def baby(n, sr=SR):
    t = np.arange(n) / sr
    f = 400 + 300 * np.sin(2 * np.pi * 1.2 * t)
    out = np.sin(2 * np.pi * np.cumsum(f) / sr) * 0.3 + rng.normal(0, 0.05, n)
    env = np.maximum(np.sin(2 * np.pi * 0.8 * t), 0) ** 0.5
    return out * env / max(np.max(np.abs(out * env)), 1e-9) * 0.3


cases = {
    "【真语音】完整10.3s": speech_full,
    "【真语音】前3s": speech_full[:N],
    "【真语音】后3s": speech_full[-N:],
    "【真语音】中3s": speech_full[len(speech_full)//2 - N//2: len(speech_full)//2 + N//2],
    "【噪声】白噪-30dB": rng.normal(0, 10 ** (-30 / 20), N),
    "【噪声】白噪-50dB": rng.normal(0, 10 ** (-50 / 20), N),
    "【噪声】全零": np.zeros(N),
    "【噪声】粉噪-40dB": np.convolve(rng.normal(0, 1, N), np.ones(8) / 8, mode="same")[:N],
    "【噪声】音乐": music(N),
    "【噪声】babble": babble(N),
    "【噪声】电视": tv(N),
    "【噪声】婴儿哭": baby(N),
}
# 低信噪比语音
for snr in [5, 0, -5]:
    noise = rng.normal(0, 1, len(speech_full))
    sp = np.sqrt(np.mean(speech_full ** 2))
    nz = np.sqrt(np.mean(noise ** 2))
    cases[f"【真语音】SNR{snr}dB"] = (speech_full + noise * (sp / nz) * 10 ** (-snr / 20)).astype(np.float32)

print("=" * 124)
print(f"{'场景':<22}{'帧数':>6}{'Speech帧':>10}{'BGM帧':>8}{'其他事件':>22}{'字数':>6}  文本")
print("=" * 124)
rows = []
for name, x in cases.items():
    xf = np.asarray(x, dtype=np.float32)
    text, sc, nf = run(xf)
    sp_frames = sc.get("<|Speech|>", 0)
    bgm_frames = sc.get("<|BGM|>", 0)
    others = {k: v for k, v in sc.items()
              if k not in ("<|Speech|>", "<|BGM|>", "<|zh|>", "<|en|>", "<|NEUTRAL|>",
                           "<|HAPPY|>", "<|SAD|>", "<|ANGRY|>", "<|withitn|>", "<|withoutitn|>")}
    rows.append((name, nf, sp_frames, bgm_frames, len(text), text, others))
    disp = text if len(text) <= 40 else text[:40] + "…"
    print(f"{name:<22}{nf:>6}{sp_frames:>10}{bgm_frames:>8}{str(others):>22}{len(text):>6}  {disp}")

print("=" * 124)
print("判据：Speech帧占比 = Speech帧/总帧")
print("=" * 124)
print(f"{'场景':<22}{'Speech占比':>12}{'结论':>10}")
for name, nf, sp_frames, bgm, ln, text, others in rows:
    ratio = sp_frames / max(nf, 1)
    verdict = "语音" if ratio > 0.5 else "非语音"
    print(f"{name:<22}{ratio:>12.3f}{verdict:>10}")
