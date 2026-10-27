"""寻找能真正区分「语音」与「噪声」的声学特征。

候选特征（每段音频算一组）：
  - rms        能量
  - flatness   谱平坦度（白噪声高、语音低）—— 区分噪声的经典特征
  - zcr        过零率
  - voicing    周期性（自相关峰值）—— 语音有基频
  - flux       谱通量变化（语音变化大，稳态噪声小）
  - peakRatio  峰值/均值比

目标：找到一组阈值，让真语音全部通过、纯噪声全部拒绝。
"""
import glob
import os
import re
import wave

import numpy as np

WIN, HOP, NFFT = 400, 160, 512


def load(path):
    with wave.open(path, "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def frames(x, win=WIN, hop=HOP):
    n = 1 + (len(x) - win) // hop if len(x) >= win else 0
    if n <= 0:
        return np.zeros((0, win))
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def spectrum(fr):
    pad = np.zeros((fr.shape[0], NFFT))
    pad[:, :WIN] = fr * np.hamming(WIN)
    return np.abs(np.fft.rfft(pad, n=NFFT))


def feats(x):
    fr = frames(x)
    if fr.shape[0] == 0:
        return None
    rms_all = np.sqrt(np.mean(fr.astype(np.float64) ** 2, axis=1))
    # 只看有能量的帧
    thr = max(np.percentile(rms_all, 30), 1e-6)
    active = fr[rms_all > thr]
    if active.shape[0] < 2:
        return None
    rms = float(np.sqrt(np.mean(active.astype(np.float64) ** 2)))

    mag = spectrum(active)
    power = mag ** 2 + 1e-12
    flat = float(np.mean(np.exp(np.mean(np.log(power), axis=1)) / np.mean(power, axis=1)))

    zcr = float(np.mean(np.abs(np.diff(np.sign(active), axis=1)) > 0))
    peak_ratio = float(np.max(np.abs(active)) / max(rms, 1e-9))

    # 谱通量
    nrm = mag / np.maximum(mag.sum(axis=1, keepdims=True), 1e-9)
    flux = float(np.mean(np.sqrt(np.sum(np.diff(nrm, axis=0) ** 2, axis=1)))) if nrm.shape[0] > 1 else 0.0

    # 周期性：归一化自相关峰值（60–400Hz 对应 40–267 采样点）
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
    voicing = float(np.mean(ac)) if ac else 0.0

    return dict(rms=rms, flatness=flat, zcr=zcr, peakRatio=peak_ratio, flux=flux, voicing=voicing)


print("=" * 108)
print(f"{'样本':<24}{'rms':>9}{'flatness':>10}{'zcr':>8}{'peakRatio':>11}{'flux':>9}{'voicing':>9}")
print("=" * 108)

print("--- 正样本：真语音 ---")
pos = []
for path in sorted(glob.glob("test-audio/zh_16k*.wav")):
    name = os.path.basename(path).replace(".wav", "")
    f = feats(load(path))
    pos.append((name, f))
    print(f"{name:<24}{f['rms']:>9.4f}{f['flatness']:>10.5f}{f['zcr']:>8.3f}"
          f"{f['peakRatio']:>11.2f}{f['flux']:>9.4f}{f['voicing']:>9.3f}")

print("--- 负样本：噪声 / 静音 ---")
rng = np.random.default_rng(7)
N = 16000 * 3
neg_cases = {
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
neg_cases["粉噪 -40dB"] = p / np.sqrt(np.mean(p ** 2)) * 10 ** (-40 / 20)
for dur_ms in [320, 600, 1000, 1500]:
    neg_cases[f"短噪声 {dur_ms}ms"] = rng.normal(0, 10 ** (-35 / 20), 16 * dur_ms)
# 空调 + 键盘（更真实）
env = rng.normal(0, 10 ** (-40 / 20), N)
for pos_i in range(0, N, 9000):
    env[pos_i:pos_i + 200] += rng.normal(0, 0.02, 200) * np.exp(-np.arange(200) / 40)
neg_cases["键盘+空调"] = env

neg = []
for name, x in neg_cases.items():
    f = feats(x.astype(np.float32))
    if f is None:
        print(f"{name:<24}  (无有效帧，直接判为无语音)")
        neg.append((name, None))
        continue
    neg.append((name, f))
    print(f"{name:<24}{f['rms']:>9.4f}{f['flatness']:>10.5f}{f['zcr']:>8.3f}"
          f"{f['peakRatio']:>11.2f}{f['flux']:>9.4f}{f['voicing']:>9.3f}")

print("=" * 108)
print("【各特征可分性】正样本范围 vs 负样本范围")
print("=" * 108)
for key in ["rms", "flatness", "zcr", "peakRatio", "flux", "voicing"]:
    pv = [p[1][key] for p in pos]
    nv = [n[1][key] for n in neg if n[1] is not None]
    print(f"{key:<12} 正: [{min(pv):.5f}, {max(pv):.5f}]   负: [{min(nv):.5f}, {max(nv):.5f}]")
