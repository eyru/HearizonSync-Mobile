"""对比几种降噪方案在 SenseVoice 上的真实识别收益（CER），而不是能量指标。

候选：
  A. 时域帧级噪声门（桌面版 v2.5.6 的做法）
  B. 谱减法（spectral subtraction，频域稳态噪声抑制）
  C. 维纳滤波（用噪声谱估计的先验 SNR）
  D. A + 增益归一

用法（在 工具/参考实现 目录）：
    PYTHONIOENCODING=utf-8 python eval_denoise2.py
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
REFERENCE = "大家好，今天天气不错，我们现在测试一下实时语音识别系统，看看断句和标点准不准。"

sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
meta = sess.get_modelmeta().custom_metadata_map
LFR_M = int(meta["lfr_window_size"])
LFR_N = int(meta["lfr_window_shift"])
NEG = np.fromstring(meta["neg_mean"], sep=",")
INV = np.fromstring(meta["inv_stddev"], sep=",")
tokens = {}
for line in open(TOKENS, encoding="utf-8"):
    line = line.rstrip("\n")
    if line:
        i = line.rfind(" ")
        tokens[int(line[i + 1:])] = line[:i]


def load(path):
    with wave.open(path, "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def fbank(x, sr=16000):
    nfft, win, hop = 512, 400, 160
    n = len(x)
    nf = 1 + (n - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(nf)[:, None]
    fr = x[idx].astype(np.float64)
    fr = fr - fr.mean(axis=1, keepdims=True)
    fr = np.append(fr[:, 0:1], fr[:, 1:] - 0.97 * fr[:, :-1], axis=1)
    fr = fr * np.hamming(win)
    pad = np.zeros((nf, nfft))
    pad[:, :win] = fr
    spec = np.fft.rfft(pad, n=nfft)
    p = spec.real ** 2 + spec.imag ** 2

    def hz2mel(f):
        return 1127 * np.log(1 + f / 700)

    def mel2hz(m):
        return 700 * (np.exp(m / 1127) - 1)

    mels = np.linspace(hz2mel(20), hz2mel(sr / 2 - 400), 82)
    bins = np.floor((nfft + 1) * mel2hz(mels) / sr).astype(int)
    fb = np.zeros((80, nfft // 2 + 1))
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


def recognize(x, sr=16000):
    cm = (lfr(fbank(x, sr), LFR_M, LFR_N) + NEG) * INV
    logits = sess.run(None, {
        "x": cm[None, :, :].astype(np.float32),
        "x_length": np.array([cm.shape[0]], dtype=np.int32),
        "language": np.array([3], dtype=np.int32),
        "text_norm": np.array([14], dtype=np.int32),
    })[0][0]
    ids = logits.argmax(axis=1)
    out, prev = [], -1
    for i in ids:
        if i != prev and i != 0:
            out.append(tokens.get(int(i), ""))
        prev = i
    return re.sub(r"<\|[^|]*\|>", "", "".join(out).replace("▁", " ")).strip()


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


# ------------------------------------------------------------------ 方案 A：时域噪声门
def denoise_gate(x, strength=1.0):
    a = np.asarray(x, dtype=np.float32)
    frame = 320
    n = a.size // frame
    if n < 2:
        return a
    body = a[:n * frame].reshape(n, frame)
    energy = np.sqrt(np.mean(body.astype(np.float64) ** 2, axis=1))
    noise = float(np.percentile(energy, 10))
    if noise <= 1e-6:
        return a
    gate = noise * 1.1 / max(0.3, strength)
    quiet = energy < gate
    if not np.any(quiet):
        return a
    gain = np.ones(n)
    gain[quiet] = 0.30
    out = (body.astype(np.float64) * gain[:, None]).ravel()
    return np.concatenate([out, a[n * frame:]]).astype(np.float32)


def normalize_gain(x, target=0.12, max_gain=6.0):
    a = np.asarray(x, dtype=np.float32)
    rms = float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
    if rms <= 1e-5:
        return a
    g = target / rms
    if g <= 1.0:
        return a
    g = min(g, max_gain)
    out = a * g
    peak = float(np.max(np.abs(out)))
    if peak > 0.97:
        out = out * (0.97 / peak)
    return out.astype(np.float32)


# ------------------------------------------------------------------ 频域工具
NFFT, WIN, HOP = 512, 400, 160


def stft(x):
    n = len(x)
    nf = 1 + (n - WIN) // HOP
    idx = np.arange(WIN)[None, :] + HOP * np.arange(nf)[:, None]
    frames = x[idx].astype(np.float64) * np.hamming(WIN)
    pad = np.zeros((nf, NFFT))
    pad[:, :WIN] = frames
    return np.fft.rfft(pad, n=NFFT), n


def istft(spec, n):
    frames = np.fft.irfft(spec, n=NFFT)[:, :WIN]
    nf = frames.shape[0]
    out = np.zeros((nf - 1) * HOP + WIN)
    norm = np.zeros_like(out)
    w = np.hamming(WIN)
    for i in range(nf):
        s = i * HOP
        out[s:s + WIN] += frames[i] * w
        norm[s:s + WIN] += w * w
    out = out / np.maximum(norm, 1e-8)
    return out[:n].astype(np.float32)


def estimate_noise_power(power, percentile=10):
    """按频点取时间维分位数当噪声功率谱（稳态噪声假设）。"""
    return np.percentile(power, percentile, axis=0)


# ------------------------------------------------------------------ 方案 B：谱减法
def denoise_spectral(x, over_sub=2.0, floor=0.02):
    spec, n = stft(x)
    power = np.abs(spec) ** 2
    noise = estimate_noise_power(power)
    clean = power - over_sub * noise[None, :]
    clean = np.maximum(clean, floor * power)
    mag = np.sqrt(clean)
    phase = spec / np.maximum(np.abs(spec), 1e-12)
    return istft(mag * phase, n)


# ------------------------------------------------------------------ 方案 C：维纳滤波
def denoise_wiener(x, floor=0.02):
    spec, n = stft(x)
    power = np.abs(spec) ** 2
    noise = estimate_noise_power(power)
    snr_post = np.maximum(power / np.maximum(noise[None, :], 1e-12) - 1.0, 0.0)
    gain = snr_post / (1.0 + snr_post)
    gain = np.maximum(gain, floor)
    phase = spec / np.maximum(np.abs(spec), 1e-12)
    return istft(np.sqrt(power) * gain * phase, n)


METHODS = {
    "原始": lambda x: x,
    "A 时域门": lambda x: normalize_gain(denoise_gate(x)),
    "B 谱减": lambda x: denoise_spectral(x),
    "C 维纳": lambda x: denoise_wiener(x),
    "B+增益": lambda x: normalize_gain(denoise_spectral(x)),
}

files = sorted(glob.glob("test-audio/zh_16k*.wav"))
results = {}
print("=" * 96)
print(f"{'音频':<22}" + "".join(f"{k:>12}" for k in METHODS))
print("=" * 96)
for path in files:
    name = os.path.basename(path).replace(".wav", "")
    x = load(path)
    row = {}
    for key, fn in METHODS.items():
        row[key] = cer(REFERENCE, recognize(fn(x)))
    results[name] = row
    print(f"{name:<22}" + "".join(f"{row[k]:>12.3f}" for k in METHODS))

print("=" * 96)
avg = {k: np.mean([results[n][k] for n in results]) for k in METHODS}
print(f"{'平均 CER':<22}" + "".join(f"{avg[k]:>12.3f}" for k in METHODS))
print("=" * 96)
