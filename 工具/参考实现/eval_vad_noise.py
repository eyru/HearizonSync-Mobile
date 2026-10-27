"""VAD 噪声估计方法对比 + 文本级幻觉过滤规则设计。

上一轮暴露两个问题：
  1. 我的 eval 里噪声底用「滑动平均」估计，语音持续时会跟着爬升，把弱音切掉 → 切句碎片化
  2. 音乐 / 电视这类「类语音噪声」，voicing 特征挡不住（音乐 voicing=0.34）

本脚本：
  E1. 对比三种噪声底估计法（滑动平均 / 最小值统计 / 分位数）的切句质量
  E2. 统计噪声产物 vs 真语音产物的文本特征，设计过滤规则
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

SR = 16000
WIN, HOP, NFFT = 400, 160, 512
FRAME = 320


def load(path):
    with wave.open(path, "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def fbank(x, sr=SR):
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


def recognize(x):
    cm = (lfr(fbank(x), LFR_M, LFR_N) + NEG) * INV
    logits = sess.run(None, {
        "x": cm[None, :, :].astype(np.float32),
        "x_length": np.array([cm.shape[0]], dtype=np.int32),
        "language": np.array([LANG_ZH], dtype=np.int32),
        "text_norm": np.array([14], dtype=np.int32),
    })[0][0]
    ids = logits.argmax(axis=1)
    out, prev = [], -1
    for i in ids:
        if i != prev and i != 0:
            out.append(tokens.get(int(i), ""))
        prev = i
    return re.sub(r"<\|[^|]*\|>", "", "".join(out).replace("▁", " ")).strip()


def core(text):
    return re.sub(r"[，。！？、,.!?；;：:\s\"'（）()【】]", "", text)


# ---------------------------------------------------------------- 三种噪声底估计
class VadBase:
    def __init__(self, noise_mult=2.6, min_ms=320, tail_frames=12, abs_floor=0.003):
        self.noise_mult = noise_mult
        self.min_ms = min_ms
        self.tail_frames = tail_frames
        self.abs_floor = abs_floor
        self.in_speech = False
        self.ring = []
        self.seg = []
        self.silence = 0

    def threshold(self):
        raise NotImplementedError

    def push(self, frame):
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        self.observe(rms)
        thr = max(self.threshold() * self.noise_mult, self.abs_floor)
        active = rms > thr
        if not self.in_speech:
            self.ring.append(frame)
            if len(self.ring) > 10:
                self.ring.pop(0)
            if active:
                self.in_speech = True
                self.seg = list(self.ring)
                self.ring = []
                self.silence = 0
            return None
        self.seg.append(frame)
        self.silence = 0 if active else self.silence + 1
        dur_ms = len(self.seg) * 20
        if self.silence >= self.tail_frames or dur_ms >= 15000:
            audio = np.concatenate(self.seg)
            self.in_speech = False
            self.seg = []
            self.silence = 0
            return audio if dur_ms >= self.min_ms else None
        return None


class VadEma(VadBase):
    """滑动平均（原实现）。"""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.noise = 0.0
        self.warm = 0

    def observe(self, rms):
        if self.warm < 25:
            self.warm += 1
            self.noise = rms if self.warm == 1 else self.noise * 0.8 + rms * 0.2
        else:
            self.noise = self.noise * 0.995 + min(rms, self.noise * 2) * 0.005

    def threshold(self):
        return self.noise


class VadMinStat(VadBase):
    """最小值统计：维护滑动窗口内的最小 RMS 当噪声底。业界标准做法。"""

    def __init__(self, window=150, **kw):  # 150 帧 = 3 秒
        super().__init__(**kw)
        self.buf = []
        self.window = window
        self.warm = 0

    def observe(self, rms):
        self.warm += 1
        self.buf.append(rms)
        if len(self.buf) > self.window:
            self.buf.pop(0)

    def threshold(self):
        if self.warm < 25 or not self.buf:
            return self.abs_floor
        return float(np.percentile(self.buf, 10))


class VadMinStatSlow(VadBase):
    """最小值统计 + 更慢的窗口（5 秒），对持续语音更稳。"""

    def __init__(self, window=250, **kw):
        super().__init__(**kw)
        self.buf = []
        self.window = window
        self.warm = 0

    def observe(self, rms):
        self.warm += 1
        self.buf.append(rms)
        if len(self.buf) > self.window:
            self.buf.pop(0)

    def threshold(self):
        if self.warm < 25 or not self.buf:
            return self.abs_floor
        return float(np.percentile(self.buf, 5))


def run_vad(cls, audio, **kw):
    vad = cls(**kw)
    segs = []
    for i in range(0, len(audio) - FRAME, FRAME):
        r = vad.push(audio[i:i + FRAME])
        if r is not None:
            segs.append(r)
    return segs


speech = load("test-audio/zh_16k.wav")
REFERENCE = "大家好，今天天气不错，我们现在测试一下实时语音识别系统，看看断句和标点准不准。"


def cer(ref, hyp):
    a, b = core(ref), core(hyp)
    if not a:
        return 0.0 if not b else 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / len(a)


print("=" * 112)
print("E1：噪声底估计方法对切句质量的影响（10.3s 连续语音）")
print("=" * 112)
for cls, name in [(VadEma, "滑动平均"), (VadMinStat, "最小值统计3s"), (VadMinStatSlow, "最小值统计5s")]:
    for mult in [2.2, 2.6, 3.0]:
        segs = run_vad(cls, speech, noise_mult=mult)
        texts = [recognize(s) for s in segs]
        texts = [t for t in texts if core(t)]
        joined = "".join(texts)
        print(f"  {name:<14} mult={mult:<4} 切{len(segs):>2}段 共{sum(len(s) for s in segs)/SR:>5.2f}s "
              f"CER={cer(REFERENCE, joined):.3f}  {' / '.join(texts)}")

print()
print("=" * 112)
print("E2：噪声产物 vs 真语音产物的文本特征")
print("=" * 112)


def gen_music(n):
    rng = np.random.default_rng(5)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for f0, amp in [(220, 0.3), (277.2, 0.25), (330, 0.2), (440, 0.15)]:
        for h in range(1, 6):
            out += amp / h * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6.28))
    beat = (np.sin(2 * np.pi * 2.0 * t) > 0.55).astype(float)
    beat = np.convolve(beat, np.exp(-np.arange(250) / 60), mode="same")
    out *= (0.15 + 0.85 * beat)
    return out / max(np.max(np.abs(out)), 1e-9) * 0.25


def gen_tv(n):
    rng = np.random.default_rng(6)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for _ in range(3):
        f0 = rng.uniform(90, 220)
        fm = np.cumsum(rng.normal(0, 0.35, n))
        ph = 2 * np.pi * (f0 * t + fm / SR * 60)
        out += np.sin(ph) * 0.22 + rng.normal(0, 0.035, n)
    syl = np.maximum(np.sin(2 * np.pi * 3.6 * t), 0) ** 1.4
    return out * syl / max(np.max(np.abs(out * syl)), 1e-9) * 0.22


FILLER = set("嗯啊哦呃唉诶呀哈嘿噢喔呐哪吧嘛呗咧咯啦嘞")

samples = []
for gname, gen in [("音乐", gen_music), ("电视", gen_tv)]:
    audio = gen(SR * 20).astype(np.float32)
    for cls in [VadEma, VadMinStat]:
        for seg in run_vad(cls, audio, noise_mult=2.6):
            t = recognize(seg)
            c = core(t)
            if c:
                samples.append((gname, len(seg) / SR, t, c, len(c)))

print(f"{'来源':<8}{'时长':>7}{'字数':>6}{'语气词占比':>11}{'是否全语气词':>13}  文本")
for gname, dur, t, c, n in samples[:40]:
    filler_ratio = sum(1 for ch in c if ch in FILLER) / max(len(c), 1)
    all_filler = all(ch in FILLER for ch in c)
    print(f"{gname:<8}{dur:>7.2f}{n:>6}{filler_ratio:>11.2f}{str(all_filler):>13}  [{t}]")

print()
print("真语音样本（对照）：")
segs = run_vad(VadMinStat, speech, noise_mult=2.6)
for seg in segs:
    t = recognize(seg)
    c = core(t)
    if c:
        filler_ratio = sum(1 for ch in c if ch in FILLER) / max(len(c), 1)
        all_filler = all(ch in FILLER for ch in c)
        print(f"{'语音':<8}{len(seg)/SR:>7.2f}{len(c):>6}{filler_ratio:>11.2f}{str(all_filler):>13}  [{t}]")

print()
print("=" * 112)
print("过滤规则候选：")
print("  R1: 纯语气词（全部字符都在语气词表）→ 丢弃")
print("  R2: 去标点后 < 2 字 → 丢弃")
print("  R3: R1 或 R2")
print("=" * 112)
