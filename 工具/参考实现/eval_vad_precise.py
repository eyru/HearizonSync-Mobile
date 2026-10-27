"""精确评估：VAD 在【非语音区域】误切出多少段，以及这些段是否产生字幕。

标注真值：拼接音频时记录每段的类型与起止时间。
指标：
  - 误切率 = 落在非语音区域的切段数 / 非语音总时长（每分钟误切次数）
  - 漏检率 = 落在语音区域但未被切出的语音时长占比
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
FRAME = 320  # 20ms


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


# ---------------------------------------------------------------- VAD（可配参）
class Vad:
    def __init__(self, noise_mult=2.2, min_ms=320, tail_frames=12, abs_floor=0.004,
                 voicing_thr=None):
        self.noise_mult = noise_mult
        self.min_ms = min_ms
        self.tail_frames = tail_frames
        self.abs_floor = abs_floor
        self.voicing_thr = voicing_thr
        self.noise = 0.0
        self.warm = 0
        self.in_speech = False
        self.ring = []
        self.seg = []
        self.silence = 0
        self.seg_start = 0
        self.pos = 0

    def push(self, frame):
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        if self.warm < 25:
            self.warm += 1
            self.noise = rms if self.warm == 1 else self.noise * 0.8 + rms * 0.2
            self.pos += 1
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
                self.seg_start = max(0, self.pos - len(self.ring) + 1)
                self.ring = []
                self.silence = 0
            self.pos += 1
            return None

        self.seg.append(frame)
        self.silence = 0 if active else self.silence + 1
        self.pos += 1
        dur_ms = len(self.seg) * 20
        if self.silence >= self.tail_frames or dur_ms >= 15000:
            audio = np.concatenate(self.seg)
            start = self.seg_start
            self.in_speech = False
            self.seg = []
            self.silence = 0
            if dur_ms >= self.min_ms:
                if self.voicing_thr is not None and voicing_of(audio) < self.voicing_thr:
                    return None
                return (start, audio)
            return None
        return None


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


# ---------------------------------------------------------------- 构造带标注的流
rng = np.random.default_rng(101)
speech = load("test-audio/zh_16k.wav")
CHUNK = SR * 3
chunks = [speech[i:i + CHUNK] for i in range(0, len(speech) - CHUNK + 1, CHUNK)]
if not chunks:
    chunks = [speech]


def build_stream(kind, total_s=60):
    """kind: 'quiet' | 'noisy'"""
    out, marks = [], []
    t = 0.0
    while t < total_s:
        r = rng.random()
        if kind == "quiet":
            if r < 0.5:
                c = "speech"
            elif r < 0.85:
                c = "silence"
            else:
                c = "noise"
        else:
            if r < 0.35:
                c = "speech"
            elif r < 0.5:
                c = "silence"
            else:
                c = "noise"

        if c == "speech":
            seg = chunks[rng.integers(0, len(chunks))]
        elif c == "silence":
            seg = np.zeros(int(SR * rng.uniform(0.8, 2.0)), dtype=np.float32)
        else:
            dur = rng.uniform(1.0, 2.5)
            lvl = 10 ** (rng.uniform(-42, -28) / 20)
            seg = rng.normal(0, lvl, int(SR * dur)).astype(np.float32)
        marks.append((c, t, t + len(seg) / SR))
        out.append(seg)
        t += len(seg) / SR
    return np.concatenate(out), marks


def is_speech_at(marks, t0, t1):
    """切段 [t0,t1) 与语音区域的重叠比例。"""
    overlap = 0.0
    for c, s, e in marks:
        if c == "speech":
            overlap += max(0.0, min(t1, e) - max(t0, s))
    return overlap / max(t1 - t0, 1e-6)


CONFIGS = {
    "基线(现网参数)": dict(noise_mult=2.2, min_ms=320, tail_frames=12, abs_floor=0.004),
    "严格能量门限": dict(noise_mult=3.5, min_ms=400, tail_frames=12, abs_floor=0.006),
    "严格+voicing": dict(noise_mult=3.5, min_ms=400, tail_frames=12, abs_floor=0.006, voicing_thr=0.15),
    "极严": dict(noise_mult=4.5, min_ms=500, tail_frames=14, abs_floor=0.008),
}

print("=" * 112)
print("VAD 精确评估：非语音区误切率 vs 语音区召回")
print("=" * 112)

for kind, label in [("quiet", "安静环境"), ("noisy", "嘈杂环境")]:
    print()
    print(f"### {label}")
    audio, marks = build_stream(kind, total_s=60)
    nonspeech_s = sum(e - s for c, s, e in marks if c != "speech")
    speech_s = sum(e - s for c, s, e in marks if c == "speech")
    print(f"  流长 {len(audio)/SR:.1f}s  语音 {speech_s:.1f}s  非语音 {nonspeech_s:.1f}s")
    print(f"  {'配置':<18}{'切段':>6}{'误切':>6}{'误切/分':>9}{'语音召回':>10}{'垃圾字幕':>9}")
    for name, cfg in CONFIGS.items():
        vad = Vad(**cfg)
        segs = []
        for i in range(0, len(audio) - FRAME, FRAME):
            r = vad.push(audio[i:i + FRAME])
            if r is not None:
                start_frame, seg = r
                t0 = start_frame * 0.02
                t1 = t0 + len(seg) / SR
                segs.append((t0, t1, seg))
        false_segs = [s for s in segs if is_speech_at(marks, s[0], s[1]) < 0.3]
        true_segs = [s for s in segs if is_speech_at(marks, s[0], s[1]) >= 0.3]
        recall = sum(s[1] - s[0] for s in true_segs) / max(speech_s, 1e-6)
        junk = 0
        for t0, t1, seg in false_segs:
            txt = recognize(seg)
            if core_len(txt) >= 1:
                junk += 1
        print(f"  {name:<18}{len(segs):>6}{len(false_segs):>6}"
              f"{len(false_segs)/max(nonspeech_s/60,1e-6):>9.2f}{recall:>10.2f}{junk:>9}")

print("=" * 112)
print("垃圾字幕样本（基线配置，安静环境误切段）：")
audio, marks = build_stream("quiet", total_s=60)
vad = Vad(**CONFIGS["基线(现网参数)"])
for i in range(0, len(audio) - FRAME, FRAME):
    r = vad.push(audio[i:i + FRAME])
    if r is not None:
        sf, seg = r
        t0 = sf * 0.02
        t1 = t0 + len(seg) / SR
        if is_speech_at(marks, t0, t1) < 0.3:
            txt = recognize(seg)
            if core_len(txt) >= 1:
                print(f"  {t0:>6.2f}s +{len(seg)/SR:.2f}s  [{txt}]")
print("=" * 112)
