"""降噪 / 增益归一效果评估：同一套 SenseVoice 前端，对比「原始」与「增强后」的识别结果。

用法（在 工具/参考实现 目录）：
    PYTHONIOENCODING=utf-8 python eval_denoise.py
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
    if not line:
        continue
    i = line.rfind(" ")
    tokens[int(line[i + 1:])] = line[:i]


def load(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return a.astype(np.float32) / 32768.0, sr


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

    lo, hi = 20, sr / 2 - 400
    mels = np.linspace(hz2mel(lo), hz2mel(hi), 82)
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
    tl = int(np.ceil(t / n))
    dim = f.shape[1]
    out = np.zeros((tl, dim * m))
    for i in range(tl):
        s = i * n
        if s + m <= padded.shape[0]:
            out[i] = padded[s:s + m].reshape(-1)
        else:
            flat = padded[s:].reshape(-1)
            out[i, :flat.size] = flat
            last = padded[-1]
            k, j = flat.size, 0
            while k < dim * m:
                out[i, k] = last[j % dim]
                k += 1
                j += 1
    return out


def recognize(x, sr=16000):
    f = fbank(x, sr)
    cm = (lfr(f, LFR_M, LFR_N) + NEG) * INV
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


# ---------------------------------------------------------------- 增强算法（与 Kotlin 一致）
def denoise(x, strength=1.0):
    a = np.asarray(x, dtype=np.float32)
    if a.size < 512:
        return a
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
    out = body.astype(np.float64) * gain[:, None]
    tail = a[n * frame:].astype(np.float64)
    out = np.concatenate([out.ravel(), tail]).astype(np.float32)
    peak = float(np.max(np.abs(out)))
    if peak > 0.98:
        out = (out * (0.98 / peak)).astype(np.float32)
    return out


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


def enhance(x):
    return normalize_gain(denoise(x))


def cer(ref, hyp):
    """字错率：去掉标点后按字符做编辑距离。"""
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


print("=" * 78)
print(f"{'音频':<24}{'原始 CER':>10}{'增强 CER':>10}   {'SNR':>8}")
print("=" * 78)

for path in sorted(glob.glob("test-audio/zh_16k*.wav")):
    name = os.path.basename(path).replace(".wav", "")
    x, sr = load(path)
    raw = recognize(x, sr)
    enh = recognize(enhance(x), sr)
    # 估个信噪比给表里看
    frame = 320
    n_f = len(x) // frame
    en = np.array([np.sqrt(np.mean(x[i * frame:(i + 1) * frame] ** 2)) for i in range(n_f)])
    lo = np.percentile(en, 10)
    hi = np.percentile(en, 90)
    snr = 20 * np.log10(hi / max(lo, 1e-9)) if lo > 0 else 99
    print(f"{name:<24}{cer(REFERENCE, raw):>10.3f}{cer(REFERENCE, enh):>10.3f}   {snr:>7.1f}dB")
    if cer(REFERENCE, raw) > 0.02 or cer(REFERENCE, enh) > 0.02:
        print(f"    原始: {raw}")
        print(f"    增强: {enh}")
print("=" * 78)
