"""幻觉抑制方案验证。

目标：区分「真语音」与「噪声/静音被误判为语音」。
候选判据：
  1. 平均帧置信度（max softmax）
  2. 语音帧占比（置信度 > 阈值的帧比例）
  3. 文本黑名单（「我。」「。」这类幻觉产物）
  4. <|nospeech|> 语言 id（lang_nospeech）
  5. 能量门槛（VAD 已有）

本脚本用真语音（干净 + 加噪）与纯噪声/静音做正负样本，扫描判据阈值。
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


def analyze(x, sr=16000, language=3):
    """返回 (文本, 平均置信度, 语音帧占比, 帧数, 事件集合)。"""
    cm = (lfr(fbank(x, sr), LFR_M, LFR_N) + NEG) * INV
    logits = sess.run(None, {
        "x": cm[None, :, :].astype(np.float32),
        "x_length": np.array([cm.shape[0]], dtype=np.int32),
        "language": np.array([language], dtype=np.int32),
        "text_norm": np.array([14], dtype=np.int32),
    })[0][0]

    # softmax
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    prob = e / e.sum(axis=1, keepdims=True)
    conf = prob.max(axis=1)
    ids = logits.argmax(axis=1)

    blank = 0
    # 找到非 blank 帧（近似：非 0 且非特殊）
    non_blank = np.array([i != blank for i in ids])
    frame_conf = conf[non_blank] if non_blank.any() else np.array([0.0])

    out, prev, events = [], -1, set()
    for i in ids:
        if i != prev and i != 0:
            tok = tokens.get(int(i), "")
            if tok.startswith("<|") and tok.endswith("|>"):
                events.add(tok)
            else:
                out.append(tok)
        prev = i
    text = re.sub(r"<\|[^|]*\|>", "", "".join(out).replace("▁", " ")).strip()
    return text, float(frame_conf.mean()), float((conf > 0.5).mean()), len(ids), events


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


print("=" * 100)
print("【正样本】真语音")
print("=" * 100)
print(f"{'音频':<22}{'置信度':>9}{'高置信帧比':>11}{'帧数':>7}  CER")
pos = []
for path in sorted(glob.glob("test-audio/zh_16k*.wav")):
    name = os.path.basename(path).replace(".wav", "")
    x = load(path)
    text, conf, ratio, nf, ev = analyze(x)
    c = cer(REFERENCE, text)
    pos.append((name, conf, ratio, nf, c))
    print(f"{name:<22}{conf:>9.3f}{ratio:>11.3f}{nf:>7}  {c:.3f}")

print("=" * 100)
print("【负样本】纯噪声 / 静音")
print("=" * 100)
rng = np.random.default_rng(7)
N = 16000 * 3
neg = []
cases = {
    "纯白噪 -30dB": rng.normal(0, 10 ** (-30 / 20), N),
    "纯白噪 -50dB": rng.normal(0, 10 ** (-50 / 20), N),
    "全零静音": np.zeros(N),
    "敲击+底噪": np.where(
        (np.arange(N) % 16000 < 400),
        rng.normal(0, 0.05, N) * np.exp(-(np.arange(N) % 16000) / 60),
        rng.normal(0, 1e-5, N),
    ),
}
p = rng.normal(0, 1, N)
for _ in range(3):
    p = np.convolve(p, np.ones(8) / 8, mode="same")
cases["粉噪 -40dB"] = p / np.sqrt(np.mean(p ** 2)) * 10 ** (-40 / 20)
for dur_ms in [320, 400, 600, 1000, 1500]:
    cases[f"短噪声 {dur_ms}ms"] = rng.normal(0, 10 ** (-35 / 20), 16 * dur_ms)

print(f"{'场景':<22}{'置信度':>9}{'高置信帧比':>11}{'帧数':>7}  文本")
for name, x in cases.items():
    text, conf, ratio, nf, ev = analyze(x.astype(np.float32))
    neg.append((name, conf, ratio, nf, text))
    print(f"{name:<22}{conf:>9.3f}{ratio:>11.3f}{nf:>7}  [{text}]")

print("=" * 100)
print("【判据可分性】")
print("=" * 100)
pc = [p[1] for p in pos]
nc = [n[1] for n in neg]
print(f"正样本 平均置信度: min={min(pc):.3f} max={max(pc):.3f}")
print(f"负样本 平均置信度: min={min(nc):.3f} max={max(nc):.3f}")
pr = [p[2] for p in pos]
nr = [n[2] for n in neg]
print(f"正样本 高置信帧比: min={min(pr):.3f} max={max(pr):.3f}")
print(f"负样本 高置信帧比: min={min(nr):.3f} max={max(nr):.3f}")

# 文本黑名单统计
print("=" * 100)
print("【负样本文本分布】—— 幻觉产物清单")
print("=" * 100)
for name, conf, ratio, nf, text in neg:
    print(f"{name:<22} [{text}]  长度={len(text)}")
