"""最终方案综合验证：最小值统计噪声底 + 文本级幻觉过滤。

对比 4 组配置在 3 类场景下的表现：
  场景A 干净连续语音（10.3s，考察切句质量与 CER）
  场景B 类语音噪声（音乐/电视/键盘，考察幻觉字幕数量）
  场景C 混合流（语音+静音+噪声交替 60s，考察综合体验）
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
REFERENCE = "大家好，今天天气不错，我们现在测试一下实时语音识别系统，看看断句和标点准不准。"
FILLER = set("嗯啊哦呃唉诶呀哈嘿噢喔呐哪吧嘛呗咧咯啦嘞呐哎哟")


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


# ---------------------------------------------------------------- 过滤器
def is_hallucination(text, min_chars=2, drop_filler=True):
    c = core(text)
    if len(c) < min_chars:
        return True
    if drop_filler and c and all(ch in FILLER for ch in c):
        return True
    return False


# ---------------------------------------------------------------- VAD（最小值统计噪声底）
class Vad:
    def __init__(self, noise_mult=2.6, min_ms=320, tail_frames=12, abs_floor=0.003,
                 window=150, pct=10, filter_fn=None):
        self.noise_mult = noise_mult
        self.min_ms = min_ms
        self.tail_frames = tail_frames
        self.abs_floor = abs_floor
        self.window = window
        self.pct = pct
        self.filter_fn = filter_fn
        self.buf = []
        self.in_speech = False
        self.ring = []
        self.seg = []
        self.silence = 0

    def threshold(self):
        if len(self.buf) < 20:
            return self.abs_floor
        return max(float(np.percentile(self.buf, self.pct)), self.abs_floor)

    def push(self, frame):
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        self.buf.append(rms)
        if len(self.buf) > self.window:
            self.buf.pop(0)
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


def run_vad(audio, **kw):
    vad = Vad(**kw)
    segs = []
    for i in range(0, len(audio) - FRAME, FRAME):
        r = vad.push(audio[i:i + FRAME])
        if r is not None:
            segs.append(r)
    return segs


def transcript(segs, filter_fn=None):
    texts = []
    for s in segs:
        t = recognize(s)
        if filter_fn is not None and filter_fn(t):
            continue
        if core(t):
            texts.append(t)
    return texts


# ---------------------------------------------------------------- 类语音噪声
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
        ph = 2 * np.pi * (f0 * t + fm / SR * 60)
        out += np.sin(ph) * 0.22 + rng.normal(0, 0.035, n)
    syl = np.maximum(np.sin(2 * np.pi * 3.6 * t), 0) ** 1.4
    return (out * syl / max(np.max(np.abs(out * syl)), 1e-9) * 0.22).astype(np.float32)


def gen_keyboard(n, seed=8):
    rng = np.random.default_rng(seed)
    out = rng.normal(0, 6e-4, n)
    for _ in range(n // 7000):
        pos = rng.integers(0, n - 800)
        out[pos:pos + 800] += rng.normal(0, 0.06, 800) * np.exp(-np.arange(800) / 90)
    return out.astype(np.float32)


print("=" * 112)
print("场景 A：干净连续语音 10.3s —— 切句质量与 CER")
print("=" * 112)
speech = load("test-audio/zh_16k.wav")
CONFIGS = {
    "①旧(EMA,无过滤)": dict(noise_mult=2.6, filter_fn=None),
    "②最小值底(无过滤)": dict(noise_mult=2.6, filter_fn=None, _minstat=True),
    "③最小值底+过滤": dict(noise_mult=2.6, filter_fn=is_hallucination, _minstat=True),
    "④最小值底+过滤+严": dict(noise_mult=3.2, filter_fn=is_hallucination, _minstat=True),
}


class VadEma(Vad):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.ema = 0.0
        self.warm = 0

    def threshold(self):
        return max(self.ema, self.abs_floor)

    def push(self, frame):
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        if self.warm < 25:
            self.warm += 1
            self.ema = rms if self.warm == 1 else self.ema * 0.8 + rms * 0.2
        else:
            self.ema = self.ema * 0.995 + min(rms, self.ema * 2) * 0.005
        return super().push(frame)


def run_cfg(cfg, audio):
    cfg = dict(cfg)
    minstat = cfg.pop("_minstat", False)
    filt = cfg.pop("filter_fn", None)
    cls = Vad if minstat else VadEma
    vad = cls(**cfg)
    segs = []
    for i in range(0, len(audio) - FRAME, FRAME):
        r = vad.push(audio[i:i + FRAME])
        if r is not None:
            segs.append(r)
    return segs, filt


for name, cfg in CONFIGS.items():
    segs, filt = run_cfg(cfg, speech)
    texts = transcript(segs, filt)
    print(f"  {name:<20} 切{len(segs):>2}段 共{sum(len(s) for s in segs)/SR:>5.2f}s "
          f"CER={cer(REFERENCE, ''.join(texts)):.3f}  {' / '.join(texts)}")

print()
print("=" * 112)
print("场景 B：类语音噪声 20s —— 幻觉字幕条数")
print("=" * 112)
noises = {"音乐": gen_music(SR * 20), "电视": gen_tv(SR * 20), "键盘": gen_keyboard(SR * 20)}
print(f"  {'噪声':<8}{'配置':<22}{'切段':>6}{'字幕条数':>10}  内容")
for nname, audio in noises.items():
    for cname, cfg in CONFIGS.items():
        segs, filt = run_cfg(cfg, audio)
        texts = transcript(segs, filt)
        disp = " / ".join(texts[:5]) if texts else "（无）"
        if len(disp) > 40:
            disp = disp[:40] + "…"
        print(f"  {nname:<8}{cname:<22}{len(segs):>6}{len(texts):>10}  {disp}")
    print()

print("=" * 112)
print("场景 C：混合流 60s（语音/静音/噪声交替）")
print("=" * 112)
rng = np.random.default_rng(303)
CHUNK = SR * 3
chunks = [speech[i:i + CHUNK] for i in range(0, len(speech) - CHUNK + 1, CHUNK)] or [speech]
stream, marks = [], []
t = 0.0
while t < 60:
    r = rng.random()
    if r < 0.45:
        seg = chunks[rng.integers(0, len(chunks))]
        kind = "speech"
    elif r < 0.70:
        seg = np.zeros(int(SR * rng.uniform(0.8, 1.6)), dtype=np.float32)
        kind = "silence"
    else:
        dur = rng.uniform(1.0, 2.0)
        lvl = 10 ** (rng.uniform(-42, -30) / 20)
        seg = rng.normal(0, lvl, int(SR * dur)).astype(np.float32)
        kind = "noise"
    marks.append((kind, t, t + len(seg) / SR))
    stream.append(seg)
    t += len(seg) / SR
stream = np.concatenate(stream)
speech_s = sum(e - s for k, s, e in marks if k == "speech")
print(f"  流长 {len(stream)/SR:.1f}s，其中语音 {speech_s:.1f}s")
for cname, cfg in CONFIGS.items():
    cfg2 = dict(cfg)
    minstat = cfg2.pop("_minstat", False)
    filt = cfg2.pop("filter_fn", None)
    cls = Vad if minstat else VadEma
    vad = cls(**cfg2)
    segs = []
    pos = 0
    for i in range(0, len(stream) - FRAME, FRAME):
        r = vad.push(stream[i:i + FRAME])
        if r is not None:
            segs.append((i * 0.02, r))
    texts = []
    for t0, s in segs:
        tt = recognize(s)
        if filt is not None and filt(tt):
            continue
        if core(tt):
            texts.append((t0, len(s) / SR, tt))
    total_core = sum(len(core(x[2])) for x in texts)
    print(f"  {cname:<20} 切{len(segs):>3}段 出字幕{len(texts):>3}条 共{total_core:>3}字 "
          f"（语音{int(speech_s)}s，理想约{speech_s*2.5:.0f}字）")
    for t0, dur, tt in texts[:8]:
        print(f"      {t0:>5.1f}s +{dur:.2f}s  {tt}")

print("=" * 112)
