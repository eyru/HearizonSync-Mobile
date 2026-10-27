"""纯噪声 / 静音输入是否会让 SenseVoice 产生幻觉字幕？

这是离线字幕类产品最要命的失效模式：没人说话却一直往外蹦字。
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
tokens = {}
for line in open(TOKENS, encoding="utf-8"):
    line = line.rstrip("\n")
    if line:
        i = line.rfind(" ")
        tokens[int(line[i + 1:])] = line[:i]


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


def recognize(x, sr=16000, language=3):
    cm = (lfr(fbank(x, sr), LFR_M, LFR_N) + NEG) * INV
    logits = sess.run(None, {
        "x": cm[None, :, :].astype(np.float32),
        "x_length": np.array([cm.shape[0]], dtype=np.int32),
        "language": np.array([language], dtype=np.int32),
        "text_norm": np.array([14], dtype=np.int32),
    })[0][0]
    ids = logits.argmax(axis=1)
    # 顺便看看有没有直接给出 nospeech 事件
    events = set()
    out, prev = [], -1
    for i in ids:
        if i != prev and i != 0:
            tok = tokens.get(int(i), "")
            if tok.startswith("<|") and tok.endswith("|>"):
                events.add(tok)
            else:
                out.append(tok)
        prev = i
    text = re.sub(r"<\|[^|]*\|>", "", "".join(out).replace("▁", " ")).strip()
    return text, events, ids


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


rng = np.random.default_rng(7)
DUR = 3
N = 16000 * DUR

cases = {}
cases["纯白噪 -30dB"] = rng.normal(0, 10 ** (-30 / 20), N)
cases["纯白噪 -50dB"] = rng.normal(0, 10 ** (-50 / 20), N)
p = rng.normal(0, 1, N)
for _ in range(3):
    p = np.convolve(p, np.ones(8) / 8, mode="same")
cases["粉噪 -40dB"] = p / np.sqrt(np.mean(p ** 2)) * 10 ** (-40 / 20)
x = rng.normal(0, 1e-5, N)
for pos in [8000, 24000, 40000]:
    x[pos:pos + 400] += rng.normal(0, 0.05, 400) * np.exp(-np.arange(400) / 60)
cases["敲击+底噪"] = x
cases["全零静音"] = np.zeros(N, dtype=np.float64)

print("=" * 84)
print("纯噪声 / 静音输入 → SenseVoice 是否产生幻觉字幕")
print("=" * 84)
for name, x in cases.items():
    xf = x.astype(np.float32)
    raw, ev_raw, _ = recognize(xf)
    enh, ev_enh, _ = recognize(normalize_gain(denoise_gate(xf)))
    print(f"{name:<14} 原始=[{raw}]  事件={sorted(ev_raw)}")
    print(f"{'':<14} 降噪=[{enh}]  事件={sorted(ev_enh)}")
    print("-" * 84)

# 短片段（模拟 VAD 误切出来的 0.4 秒噪声段）
print("=" * 84)
print("短噪声段（VAD 误触发场景）")
print("=" * 84)
for dur_ms in [320, 400, 600, 1000]:
    seg = rng.normal(0, 10 ** (-35 / 20), 16 * dur_ms).astype(np.float32)
    t, ev, _ = recognize(seg)
    print(f"{dur_ms:>5}ms  文本=[{t}]  事件={sorted(ev)}")
print("=" * 84)
