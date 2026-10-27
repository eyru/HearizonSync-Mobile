"""产品级验证：模拟真实连续输入流（语音-停顿-噪声-停顿-语音），
检验「双门限 VAD（能量 + voicing）+ 输出过滤」能否彻底消灭幻觉字幕。

这是功能 2 的真正价值所在——不是降噪提升识别率（实测无收益），
而是防止「没人说话时字幕乱蹦」。
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


# ---------------------------------------------------------------- voicing 特征
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


# ---------------------------------------------------------------- 模拟 VAD（与 Kotlin LiveSession 同逻辑）
FRAME_MS = 20
FRAME = SR * FRAME_MS // 1000  # 320
PREROLL_FRAMES = 10
TAIL_FRAMES = 12
MIN_SEGMENT_MS = 320
MAX_SEGMENT_MS = 15000


class Vad:
    """能量 + voicing 双门限 VAD。"""

    def __init__(self, use_voicing=True, voicing_thr=0.20):
        self.use_voicing = use_voicing
        self.voicing_thr = voicing_thr
        self.noise = 0.0
        self.warmup = 0
        self.reset()

    def reset(self):
        self.in_speech = False
        self.ring = []
        self.seg = []
        self.silence = 0

    def push(self, frame):
        """返回 (是否切出一段, 段音频)。"""
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        if self.warmup < 25:
            self.warmup += 1
            self.noise = max(self.noise, rms) if self.warmup == 1 else self.noise * 0.8 + rms * 0.2
            return False, None
        self.noise = self.noise * 0.995 + min(rms, self.noise * 2) * 0.005
        thr = max(self.noise * 2.2, 0.004)
        active = rms > thr

        if not self.in_speech:
            self.ring.append(frame)
            if len(self.ring) > PREROLL_FRAMES:
                self.ring.pop(0)
            if active:
                # 双门限：候选段再验 voicing
                self.in_speech = True
                self.seg = list(self.ring)
                self.ring = []
                self.silence = 0
            return False, None

        self.seg.append(frame)
        if active:
            self.silence = 0
        else:
            self.silence += 1

        dur_ms = len(self.seg) * FRAME_MS
        if self.silence >= TAIL_FRAMES or dur_ms >= MAX_SEGMENT_MS:
            audio = np.concatenate(self.seg)
            self.in_speech = False
            self.seg = []
            self.silence = 0
            if dur_ms >= MIN_SEGMENT_MS:
                if self.use_voicing:
                    v = voicing_of(audio)
                    if v < self.voicing_thr:
                        return False, None  # 被 voicing 门限拒绝
                return True, audio
            return False, None
        return False, None


def text_ok(text, min_chars=2):
    """输出过滤：至少 min_chars 个非标点字符。"""
    core = re.sub(r"[，。！？、,.!?；;：:\s\"'（）()【】]", "", text)
    return len(core) >= min_chars


rng = np.random.default_rng(23)
speech = load("test-audio/zh_16k.wav")
parts = [speech[i:i + 16000] for i in range(0, len(speech), 16000)][:6]


def make_stream(scenario, total_s=40):
    """构造真实场景的连续音频流，返回 (音频, 标注表)。"""
    out = []
    marks = []
    t = 0.0
    while t < total_s:
        kind = scenario(rng)
        if kind == "speech":
            seg = parts[rng.integers(0, len(parts))]
            out.append(seg)
            marks.append(("语音", t, len(seg) / SR))
            t += len(seg) / SR
        elif kind == "silence":
            dur = rng.uniform(0.6, 1.5)
            out.append(np.zeros(int(SR * dur), dtype=np.float32))
            marks.append(("静音", t, dur))
            t += dur
        else:
            dur = rng.uniform(0.8, 2.0)
            lvl = 10 ** (rng.uniform(-45, -25) / 20)
            out.append((rng.normal(0, lvl, int(SR * dur))).astype(np.float32))
            marks.append(("噪声", t, dur))
            t += dur
    return np.concatenate(out), marks


def scenario_meeting(rng):
    r = rng.random()
    if r < 0.45:
        return "speech"
    if r < 0.70:
        return "silence"
    return "noise"


def scenario_noisy_room(rng):
    r = rng.random()
    if r < 0.35:
        return "speech"
    if r < 0.55:
        return "silence"
    return "noise"


print("=" * 108)
print("产品级验证：连续流场景下，单门限 vs 双门限 VAD 的幻觉发生率")
print("=" * 108)

for sname, sfn in [("安静会议（多语音少噪声）", scenario_meeting),
                   ("嘈杂环境（噪声占一半）", scenario_noisy_room)]:
    print()
    print(f"### {sname}")
    for trial in range(3):
        audio, marks = make_stream(sfn, total_s=40)
        for use_v in [False, True]:
            vad = Vad(use_voicing=use_v, voicing_thr=0.20)
            segs = []
            for i in range(0, len(audio) - FRAME, FRAME):
                ok, seg = vad.push(audio[i:i + FRAME])
                if ok:
                    txt = recognize(seg)
                    dur = len(seg) / SR
                    segs.append((dur, txt))
            kept = [(d, t) for d, t in segs if text_ok(t)]
            hallu = [(d, t) for d, t in kept if len(re.sub(r"[，。！？、,.!?；;：:\s]", "", t)) <= 2]
            tag = "双门限" if use_v else "单门限"
            print(f"  trial{trial+1} {tag}: 切出{len(segs):>3}段 保留{len(kept):>3}段 "
                  f"其中短垃圾{len(hallu):>2}段")
            if use_v:
                for d, t in kept:
                    print(f"        {d:>5.2f}s  {t}")

print("=" * 108)
