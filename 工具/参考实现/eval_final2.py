"""精确复刻 Kotlin 最终实现（VadEnergy 分位底 + TranscriptFilter 带时长条件 + 可选降噪），
用于在写 Kotlin 之前锁定数字，之后模拟器实测拿来对照。
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
LFR_M, LFR_N = int(meta["lfr_window_size"]), int(meta["lfr_window_shift"])
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
WIN, HOP, NFFT, FRAME = 400, 160, 512, 320
REF = "大家好，今天天气不错，我们现在测试一下实时语音识别系统，看看断句和标点准不准。"
FILLERS = set("嗯啊哦呃唉诶呀哈嘿噢喔呐哪吧嘛呗咧咯啦嘞哟嗷唔哼咦哇")


def load(path):
    with wave.open(path, "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def fbank(x, sr=SR):
    n = len(x)
    nf = 1 + (n - WIN) // HOP
    idx = np.arange(WIN)[None, :] + HOP * np.arange(nf)[:, None]
    fr = x[idx].astype(np.float64)
    fr -= fr.mean(axis=1, keepdims=True)
    fr = np.append(fr[:, 0:1], fr[:, 1:] - 0.97 * fr[:, :-1], axis=1)
    fr *= np.hamming(WIN)
    pad = np.zeros((nf, NFFT))
    pad[:, :WIN] = fr
    spec = np.fft.rfft(pad, n=NFFT)
    p = spec.real ** 2 + spec.imag ** 2
    hz2mel = lambda f: 1127 * np.log(1 + f / 700)
    mel2hz = lambda m: 700 * (np.exp(m / 1127) - 1)
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
    if len(x) < 400:
        return ""
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
    return "".join(ch for ch in text if ch.isalnum())


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


# ------------------------------------------------ 精确复刻 Kotlin TranscriptFilter
SHORT_MS = 1200
MIN_CORE = 2


def is_hallucination(text, segment_ms):
    c = core(text)
    if not c:
        return True
    if segment_ms >= SHORT_MS:
        return False
    if len(c) < MIN_CORE:
        return True
    return all(ch in FILLERS for ch in c)


# ------------------------------------------------ 精确复刻 Kotlin VadEnergy
class Vad:
    def __init__(self, noise_mult=2.6, abs_floor=0.003, window=150, pct=10.0,
                 enter=2, exit_=12):
        self.noise_mult, self.abs_floor, self.window, self.pct = noise_mult, abs_floor, window, pct
        self.enter, self.exit = enter, exit_
        self.buf, self.count = [], 0
        self.voiced_run = self.silent_run = 0
        self.state = False

    def process(self, frame):
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        self.buf.append(rms)
        if len(self.buf) > self.window:
            self.buf.pop(0)
        floor = self.abs_floor if len(self.buf) < 20 else max(float(np.percentile(self.buf, self.pct)), self.abs_floor)
        thr = max(floor * self.noise_mult, self.abs_floor)
        voiced = rms > thr
        if voiced:
            self.voiced_run += 1
            self.silent_run = 0
        else:
            self.silent_run += 1
            self.voiced_run = 0
        if not self.state and self.voiced_run >= self.enter:
            self.state = True
        elif self.state and self.silent_run >= self.exit:
            self.state = False
        return self.state


# ------------------------------------------------ 精确复刻 Kotlin Denoiser
def denoise(samples, strength=1.0):
    if len(samples) < 512:
        return samples
    n = len(samples) // FRAME
    if n < 2:
        return samples
    energy = np.array([np.sqrt(np.mean(samples[i * FRAME:(i + 1) * FRAME].astype(np.float64) ** 2)) for i in range(n)])
    noise = float(np.percentile(energy, 10))
    if noise <= 1e-6:
        return samples
    gate = noise * 1.1 / max(0.3, strength)
    if not np.any(energy < gate):
        return samples
    out = samples.copy()
    for i in range(n):
        if energy[i] < gate:
            out[i * FRAME:(i + 1) * FRAME] *= 0.30
    return out


def normalize_gain(samples, target=0.12, max_gain=6.0):
    if len(samples) == 0:
        return samples
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    if rms <= 1e-5:
        return samples
    gain = target / rms
    if gain <= 1.0:
        return samples
    gain = min(gain, max_gain)
    peak = float(np.max(np.abs(samples * gain)))
    if peak > 0.97:
        gain *= 0.97 / peak
    return np.clip(samples * gain, -1.0, 1.0).astype(np.float32)


# ------------------------------------------------ LiveSession 精确复刻
class LiveSession:
    """与 Kotlin LiveSession.kt 一一对应：预卷 10 帧、尾巴 12 帧、最短 320ms、最长 15s。"""

    PREROLL, TAIL, MIN_MS, MAX_MS = 10, 12, 320, 15000

    def __init__(self, denoise_strength=0.0, use_filter=True, **vad_kw):
        self.vad = Vad(**vad_kw)
        self.denoise_strength = denoise_strength
        self.use_filter = use_filter
        self.preroll = []
        self.current = []
        self.silent = 0
        self.last = False
        self.out = []
        self.dropped = []

    def push(self, frame):
        speaking = self.vad.process(frame)
        if not self.current:
            if speaking:
                self.current = list(self.preroll) + [frame]
                self.preroll = []
                self.silent = 0
            else:
                self.preroll.append(frame)
                if len(self.preroll) > self.PREROLL:
                    self.preroll.pop(0)
        else:
            self.current.append(frame)
            self.silent = 0 if speaking else self.silent + 1
            if self.silent >= self.TAIL or len(self.current) * 20 >= self.MAX_MS:
                self.flush()
                self.silent = 0
        self.last = speaking
        return speaking

    def flush(self):
        if not self.current:
            return
        pcm = np.concatenate(self.current).astype(np.float32)
        dur_ms = int(len(pcm) / SR * 1000)
        self.current = []
        self.preroll = []
        if dur_ms < self.MIN_MS:
            return
        if self.denoise_strength > 0:
            pcm = normalize_gain(denoise(pcm, self.denoise_strength))
        text = recognize(pcm)
        if self.use_filter and is_hallucination(text, dur_ms):
            self.dropped.append((dur_ms, text))
            return
        if core(text):
            self.out.append((dur_ms, text))

    def finish(self):
        if self.current:
            self.flush()


def run(audio, **kw):
    s = LiveSession(**kw)
    for i in range(0, len(audio) - FRAME, FRAME):
        s.push(audio[i:i + FRAME])
    s.finish()
    return s


def gen_music(n, seed=5):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for f0, amp in [(220, 0.3), (277.2, 0.25), (330, 0.2), (440, 0.15)]:
        for h in range(1, 6):
            out += amp / h * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6.28))
    beat = (np.sin(2 * np.pi * 2.0 * t) > 0.55).astype(float)
    beat = np.convolve(beat, np.exp(-np.arange(250) / 60), mode="same")
    out *= (0.15 + 0.85 * beat)
    return (out / max(np.max(np.abs(out)), 1e-9) * 0.25).astype(np.float32)


def gen_tv(n, seed=6):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for _ in range(3):
        f0 = rng.uniform(90, 220)
        fm = np.cumsum(rng.normal(0, 0.35, n))
        out += np.sin(2 * np.pi * (f0 * t + fm / SR * 60)) * 0.22 + rng.normal(0, 0.035, n)
    syl = np.maximum(np.sin(2 * np.pi * 3.6 * t), 0) ** 1.4
    v = out * syl
    return (v / max(np.max(np.abs(v)), 1e-9) * 0.22).astype(np.float32)


def gen_keyboard(n, seed=8):
    rng = np.random.default_rng(seed)
    out = rng.normal(0, 6e-4, n)
    for _ in range(n // 7000):
        pos = rng.integers(0, n - 800)
        out[pos:pos + 800] += rng.normal(0, 0.06, 800) * np.exp(-np.arange(800) / 90)
    return out.astype(np.float32)


def add_noise(clean, snr_db, seed=0):
    rng = np.random.default_rng(seed)
    p = np.mean(clean ** 2)
    n = rng.normal(0, np.sqrt(p / (10 ** (snr_db / 10))), len(clean)).astype(np.float32)
    return (clean + n).astype(np.float32)


speech = load("test-audio/zh_16k.wav")
music, tv, kb = gen_music(SR * 20), gen_tv(SR * 20), gen_keyboard(SR * 20)
noises = {"音乐": music, "电视": tv, "键盘": kb}

print("=" * 118)
print("A. 干净语音 10.3s（VAD 修复 + 幻觉过滤，降噪关闭）")
print("=" * 118)
s = run(speech, denoise_strength=0.0, use_filter=True)
cer_a = cer(REF, "".join(t for _, t in s.out))
print(f"  切 {len(s.out) + len(s.dropped)} 段，保留 {len(s.out)} 段，CER={cer_a:.3f}")
for d, t in s.out:
    print(f"    +{d/1000:.2f}s  {t}")
if s.dropped:
    print(f"  被过滤：{s.dropped}")

print()
print("=" * 118)
print("B. 类语音噪声 20s（幻觉条数）")
print("=" * 118)
print(f"  {'噪声':<6}{'降噪':<8}{'过滤':<8}{'保留条数':>10}  内容")
for name, audio in noises.items():
    for dn in (0.0, 1.0):
        for filt in (False, True):
            r = run(audio, denoise_strength=dn, use_filter=filt)
            disp = " / ".join(t for _, t in r.out[:4]) if r.out else "（无）"
            print(f"  {name:<6}{'开' if dn else '关':<8}{'开' if filt else '关':<8}{len(r.out):>10}  {disp[:50]}")

print()
print("=" * 118)
print("C. 带噪语音（真语音 + 白噪，考察降噪对识别的影响）")
print("=" * 118)
print(f"  {'SNR':<8}{'降噪关':<30}{'降噪开':<30}")
for snr in (-5, 0, 5, 10, 15):
    noisy = add_noise(speech, snr, seed=42)
    r0 = run(noisy, denoise_strength=0.0, use_filter=True)
    r1 = run(noisy, denoise_strength=1.0, use_filter=True)
    t0 = "".join(t for _, t in r0.out)
    t1 = "".join(t for _, t in r1.out)
    print(f"  {snr:>3}dB   CER={cer(REF, t0):.3f} {len(r0.out)}段            CER={cer(REF, t1):.3f} {len(r1.out)}段")
    if snr <= 0:
        print(f"          关: {t0[:60]}")
        print(f"          开: {t1[:60]}")

print()
print("=" * 118)
print("D. 混合流 60s（语音/静音/噪声交替，最接近真实使用）")
print("=" * 118)
rng = np.random.default_rng(303)
chunks = [speech[i:i + SR * 3] for i in range(0, len(speech) - SR * 3 + 1, SR * 3)] or [speech]
stream, marks = [], []
t = 0.0
while t < 60:
    r = rng.random()
    if r < 0.45:
        seg, kind = chunks[rng.integers(0, len(chunks))], "speech"
    elif r < 0.70:
        seg, kind = np.zeros(int(SR * rng.uniform(0.8, 1.6)), dtype=np.float32), "silence"
    else:
        seg, kind = rng.normal(0, 10 ** (rng.uniform(-42, -30) / 20), int(SR * rng.uniform(1.0, 2.0))).astype(np.float32), "noise"
    marks.append((kind, t, t + len(seg) / SR))
    stream.append(seg)
    t += len(seg) / SR
stream = np.concatenate(stream)
speech_s = sum(e - s2 for k, s2, e in marks if k == "speech")
print(f"  流长 {len(stream)/SR:.1f}s，语音 {speech_s:.1f}s，噪声 {sum(e-s2 for k,s2,e in marks if k=='noise'):.1f}s")
for label, dn, filt in [("旧(EMA,无过滤)", None, False), ("VAD修复+过滤", 0.0, True), ("VAD修复+过滤+降噪", 1.0, True)]:
    if dn is None:
        continue
    r = run(stream, denoise_strength=dn, use_filter=filt)
    nchar = sum(len(core(x)) for _, x in r.out)
    print(f"  {label:<20} 保留 {len(r.out):>3} 条 共 {nchar:>4} 字（理想约 {speech_s*2.5:.0f} 字），过滤掉 {len(r.dropped)} 段")
    for d, x in r.out[:6]:
        print(f"      +{d/1000:.2f}s  {x}")
print("=" * 118)
