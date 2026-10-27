"""真实威胁场景：类语音噪声（电视对白 / 音乐 / 人群）触发能量 VAD 时会不会出幻觉字幕。

上一轮用稳态噪声测试，VAD 完全不误触发（误切率 0）——那不是我担心的场景。
真正的风险是「类语音的噪声」：能量起伏像说话，能量 VAD 会当成语音切出来。
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


def core_len(text):
    return len(re.sub(r"[，。！？、,.!?；;：:\s\"'（）()【】]", "", text))


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


# ---------------------------------------------------------------- 类语音噪声生成器
rng = np.random.default_rng(77)


def gen_music(n):
    """音乐：谐波 + 节拍，能量起伏大。"""
    t = np.arange(n) / SR
    out = np.zeros(n)
    for f0, amp in [(220, 0.3), (277.2, 0.25), (330, 0.2), (440, 0.15)]:
        for h in range(1, 6):
            out += amp / h * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6.28))
    beat = (np.sin(2 * np.pi * 2.0 * t) > 0.55).astype(float)
    beat = np.convolve(beat, np.exp(-np.arange(250) / 60), mode="same")
    out *= (0.15 + 0.85 * beat)
    return out / max(np.max(np.abs(out)), 1e-9) * 0.25


def gen_babble(n, voices=8):
    """人群嘈杂：多人声叠加。"""
    out = np.zeros(n)
    t = np.arange(n) / SR
    for _ in range(voices):
        base = rng.normal(0, 1, n)
        f = rng.uniform(150, 400)
        mod = np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) * 0.5 + 0.5
        syl = np.maximum(np.sin(2 * np.pi * rng.uniform(2.5, 5.5) * t + rng.uniform(0, 6.28)), 0) ** 2
        out += base * mod * syl * 0.12
    return out / max(np.max(np.abs(out)), 1e-9) * 0.22


def gen_tv(n):
    """电视对白：类语音共振峰扫频 + 音节包络。"""
    t = np.arange(n) / SR
    out = np.zeros(n)
    for _ in range(3):
        f0 = rng.uniform(90, 220)
        fm = np.cumsum(rng.normal(0, 0.35, n))
        ph = 2 * np.pi * (f0 * t + fm / SR * 60)
        out += np.sin(ph) * 0.22 + rng.normal(0, 0.035, n)
    syl = np.maximum(np.sin(2 * np.pi * 3.6 * t + rng.uniform(0, 6.28)), 0) ** 1.4
    return out * syl / max(np.max(np.abs(out * syl)), 1e-9) * 0.22


def gen_cough(n):
    """咳嗽 / 敲击：突发瞬态。"""
    out = rng.normal(0, 8e-4, n)
    pos = rng.integers(0, n - 3000)
    burst = rng.normal(0, 0.15, 2500) * np.exp(-np.arange(2500) / 350)
    out[pos:pos + 2500] += burst
    return out.astype(np.float32)


def gen_keyboard(n):
    """键盘敲击串。"""
    out = rng.normal(0, 6e-4, n)
    for _ in range(n // 7000):
        pos = rng.integers(0, n - 800)
        out[pos:pos + 800] += rng.normal(0, 0.06, 800) * np.exp(-np.arange(800) / 90)
    return out.astype(np.float32)


GENS = {"音乐": gen_music, "人群": gen_babble, "电视": gen_tv,
        "咳嗽": gen_cough, "键盘": gen_keyboard}

# ---------------------------------------------------------------- VAD
class Vad:
    def __init__(self, noise_mult=2.2, min_ms=320, tail_frames=12, abs_floor=0.004,
                 voicing_thr=None, min_chars=None):
        self.noise_mult = noise_mult
        self.min_ms = min_ms
        self.tail_frames = tail_frames
        self.abs_floor = abs_floor
        self.voicing_thr = voicing_thr
        self.min_chars = min_chars
        self.noise = 0.0
        self.warm = 0
        self.in_speech = False
        self.ring = []
        self.seg = []
        self.silence = 0

    def push(self, frame):
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        if self.warm < 25:
            self.warm += 1
            self.noise = rms if self.warm == 1 else self.noise * 0.8 + rms * 0.2
            return None
        self.noise = self.noise * 0.995 + min(rms, self.noise * 2) * 0.005
        thr = max(self.noise * self.noise_mult, self.abs_floor)
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
            if dur_ms >= self.min_ms:
                if self.voicing_thr is not None and voicing_of(audio) < self.voicing_thr:
                    return None
                return audio
            return None
        return None


CONFIGS = {
    "基线": dict(noise_mult=2.2, min_ms=320),
    "能量严格": dict(noise_mult=3.5, min_ms=400, abs_floor=0.006),
    "严格+voicing0.15": dict(noise_mult=3.5, min_ms=400, abs_floor=0.006, voicing_thr=0.15),
    "严格+voicing0.20": dict(noise_mult=3.5, min_ms=400, abs_floor=0.006, voicing_thr=0.20),
    "严格+voicing0.25": dict(noise_mult=3.5, min_ms=400, abs_floor=0.006, voicing_thr=0.25),
}

print("=" * 116)
print("类语音噪声 → VAD 误触发 → 幻觉字幕")
print("=" * 116)

for gname, gen in GENS.items():
    print()
    print(f"### {gname}")
    audio = gen(SR * 20).astype(np.float32)
    print(f"  {'配置':<20}{'切段':>6}{'出字幕段':>10}{'垃圾(<3字)':>11}  样本")
    for cname, cfg in CONFIGS.items():
        vad = Vad(**cfg)
        segs = []
        for i in range(0, len(audio) - FRAME, FRAME):
            r = vad.push(audio[i:i + FRAME])
            if r is not None:
                segs.append(r)
        texts = []
        for seg in segs:
            t = recognize(seg)
            if core_len(t) >= 1:
                texts.append(t)
        junk = [t for t in texts if core_len(t) < 3]
        sample = " | ".join(texts[:3]) if texts else "（无）"
        if len(sample) > 44:
            sample = sample[:44] + "…"
        print(f"  {cname:<20}{len(segs):>6}{len(texts):>10}{len(junk):>11}  {sample}")

print("=" * 116)
print("对照组：真语音在各配置下的召回（不应被误杀）")
print("=" * 116)
speech = load("test-audio/zh_16k.wav")
for cname, cfg in CONFIGS.items():
    vad = Vad(**cfg)
    segs = []
    for i in range(0, len(speech) - FRAME, FRAME):
        r = vad.push(speech[i:i + FRAME])
        if r is not None:
            segs.append(r)
    texts = [recognize(s) for s in segs]
    texts = [t for t in texts if core_len(t) >= 1]
    total = sum(len(s) for s in segs) / SR
    print(f"  {cname:<20} 切{len(segs):>3}段 共{total:>6.2f}s 文本：{' / '.join(texts)}")
print("=" * 116)
