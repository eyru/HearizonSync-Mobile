# -*- coding: utf-8 -*-
"""SenseVoice-small ONNX 的纯 numpy 参考实现。

链路：16k PCM → kaldi 风格 fbank(80) → LFR(7/6) → CMVN → CTC 贪心解码。

这个脚本是 Kotlin 端的「金标准」：手机上的前端只要和它逐帧对齐，
识别结果就不应该偏。所有参数都可从命令行覆盖，方便做参数搜索。
"""
from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODEL = r"C:\Program Files\Hearizon Sync\_internal\models\sv\model.int8.onnx"
TOKENS = r"C:\Program Files\Hearizon Sync\_internal\models\sv\tokens.txt"

EPS = 1.1920929e-07  # float32 eps，kaldi 在 log 前用它兜底


# ---------------------------------------------------------------- wav
def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as fh:
        channels = fh.getnchannels()
        width = fh.getsampwidth()
        rate = fh.getframerate()
        raw = fh.readframes(fh.getnframes())
    if channels != 1 or width != 2:
        raise ValueError(f"需要单声道 16bit，实际 channels={channels} width={width}")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return samples, rate


# ---------------------------------------------------------------- mel
def mel_scale(freq: float, htk: bool = False) -> float:
    if htk:
        return 2595.0 * math.log10(1.0 + freq / 700.0)
    return 1127.0 * math.log(1.0 + freq / 700.0)


def inverse_mel_scale(mel: float, htk: bool = False) -> float:
    if htk:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    return 700.0 * (math.exp(mel / 1127.0) - 1.0)


def mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int,
    low_freq: float,
    high_freq: float,
    htk: bool = False,
) -> np.ndarray:
    """kaldi 风格的三角 mel 滤波器组，形状 (n_mels, n_fft//2+1)。"""
    num_bins = n_fft // 2 + 1
    bin_width = sr / n_fft
    mel_low = mel_scale(low_freq, htk)
    mel_high = mel_scale(high_freq, htk)
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = np.array([inverse_mel_scale(m, htk) for m in mel_points])

    # 每个三角滤波器的起止 bin（kaldi 用 floor/ceil 的整数化方式更简单：
    # 直接在 bin 频率轴上算三角权重，二者在 8k 采样率下差异可忽略）
    freqs = np.arange(num_bins) * bin_width
    bank = np.zeros((n_mels, num_bins), dtype=np.float32)
    for m in range(n_mels):
        left, center, right = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        up = (freqs - left) / max(center - left, 1e-9)
        down = (right - freqs) / max(right - center, 1e-9)
        bank[m] = np.clip(np.minimum(up, down), 0.0, None)
    return bank


# ---------------------------------------------------------------- fbank
def fbank(
    samples: np.ndarray,
    sr: int,
    n_mels: int = 80,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
    window: str = "povey",
    preemph: float = 0.97,
    remove_dc: bool = True,
    low_freq: float = 20.0,
    high_freq: float = -400.0,
    snip_edges: bool = True,
) -> np.ndarray:
    frame_length = int(round(sr * frame_length_ms / 1000.0))  # 400
    frame_shift = int(round(sr * frame_shift_ms / 1000.0))    # 160
    n_fft = 1
    while n_fft < frame_length:
        n_fft <<= 1                                          # 512
    nyquist = sr / 2.0
    high = nyquist + high_freq if high_freq <= 0 else high_freq

    x = samples.astype(np.float64).copy()
    if preemph > 0.0:
        x[1:] -= preemph * x[:-1]

    if snip_edges:
        n_frames = 1 + (len(x) - frame_length) // frame_shift if len(x) >= frame_length else 0
    else:
        n_frames = 1 + len(x) // frame_shift
    if n_frames <= 0:
        raise ValueError("音频太短")

    idx = np.arange(frame_length)[None, :] + frame_shift * np.arange(n_frames)[:, None]
    if snip_edges:
        frames = x[idx]
    else:
        padded = np.pad(x, (frame_length // 2, frame_length // 2), mode="reflect")
        frames = padded[idx]

    if remove_dc:
        frames = frames - frames.mean(axis=1, keepdims=True)

    if window == "povey":
        win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(frame_length) / (frame_length - 1))) ** 0.85
    elif window == "hamming":
        win = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(frame_length) / (frame_length - 1))
    elif window == "hann":
        win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(frame_length) / (frame_length - 1))
    elif window == "rect":
        win = np.ones(frame_length)
    else:
        raise ValueError(window)
    frames = frames * win[None, :]

    spectrum = np.fft.rfft(frames, n=n_fft)
    power = spectrum.real ** 2 + spectrum.imag ** 2

    bank = mel_filterbank(sr, n_fft, n_mels, low_freq, high, htk=False)
    mel = power @ bank.T
    return np.log(np.maximum(mel, EPS)).astype(np.float32)


# ---------------------------------------------------------------- LFR + CMVN
def apply_lfr(feats: np.ndarray, lfr_m: int = 7, lfr_n: int = 6) -> np.ndarray:
    """低帧率：每 lfr_n 帧取 lfr_m 帧拼成一行（FunASR 的做法，左填 (m-1)//2 帧）。"""
    t = feats.shape[0]
    t_lfr = int(math.ceil(t / lfr_n))
    left = (lfr_m - 1) // 2
    padded = np.vstack([np.tile(feats[0:1], (left, 1)), feats])
    total = padded.shape[0]
    out = np.empty((t_lfr, feats.shape[1] * lfr_m), dtype=np.float32)
    for i in range(t_lfr):
        start = i * lfr_n
        if start + lfr_m <= total:
            out[i] = padded[start:start + lfr_m].reshape(-1)
        else:
            chunk = padded[start:].reshape(-1)
            pad = lfr_m - (total - start)
            out[i] = np.concatenate([chunk, np.tile(padded[-1], pad)])
    return out


def apply_cmvn(feats: np.ndarray, neg_mean: np.ndarray, inv_stddev: np.ndarray) -> np.ndarray:
    return ((feats + neg_mean[None, :]) * inv_stddev[None, :]).astype(np.float32)


# ---------------------------------------------------------------- tokens / CTC
def load_tokens(path: str | Path) -> dict[int, str]:
    table: dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            token, _, idx = line.rpartition(" ")
            table[int(idx)] = token
    return table


def ctc_greedy(logits: np.ndarray, table: dict[int, str]) -> tuple[str, list[int]]:
    ids = logits[0].argmax(axis=-1)
    collapsed: list[int] = []
    prev = -1
    for i in ids:
        if i != prev:
            collapsed.append(int(i))
        prev = i
    kept = [i for i in collapsed if i != 0]
    pieces = [table.get(i, "") for i in kept]
    text = "".join(pieces).replace("\u2581", " ").strip()
    # 去掉 SenseVoice 的 <|zh|><|NEUTRAL|><|Speech|><|withitn|> 这类控制 token
    import re
    text = re.sub(r"<\|[^|]*\|>", "", text).strip()
    return text, kept


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--window", default="povey", choices=["povey", "hamming", "hann", "rect"])
    ap.add_argument("--preemph", type=float, default=0.97)
    ap.add_argument("--low-freq", type=float, default=20.0)
    ap.add_argument("--high-freq", type=float, default=-400.0)
    ap.add_argument("--no-lfr", action="store_true")
    ap.add_argument("--no-cmvn", action="store_true")
    ap.add_argument("--language", type=int, default=3, help="0=auto 3=zh 4=en 11=ja")
    ap.add_argument("--text-norm", type=int, default=14, help="14=with itn 15=without")
    ap.add_argument("--dump-fbank", default="")
    args = ap.parse_args()

    samples, sr = load_wav(args.wav)
    print(f"audio: {len(samples)} samples, {len(samples)/sr:.2f}s, sr={sr}")

    feats = fbank(
        samples, sr,
        window=args.window, preemph=args.preemph,
        low_freq=args.low_freq, high_freq=args.high_freq,
    )
    print(f"fbank: {feats.shape}  mean={feats.mean():.4f} min={feats.min():.3f} "
          f"max={feats.max():.3f} rms={np.sqrt((feats**2).mean()):.4f}")
    if args.dump_fbank:
        np.save(args.dump_fbank, feats)
        print("fbank dumped to", args.dump_fbank)

    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    neg_mean = np.array([float(v) for v in meta["neg_mean"].split(",")], dtype=np.float32)
    inv_std = np.array([float(v) for v in meta["inv_stddev"].split(",")], dtype=np.float32)

    if not args.no_lfr:
        feats = apply_lfr(feats, int(meta["lfr_window_size"]), int(meta["lfr_window_shift"]))
        print(f"lfr: {feats.shape}")
    if not args.no_cmvn:
        feats = apply_cmvn(feats, neg_mean, inv_std)
        print(f"cmvn: mean={feats.mean():.4f} std={feats.std():.4f}")

    x = feats[None, :, :]
    feeds = {
        "x": x,
        "x_length": np.array([x.shape[1]], dtype=np.int32),
        "language": np.array([args.language], dtype=np.int32),
        "text_norm": np.array([args.text_norm], dtype=np.int32),
    }
    logits = sess.run(None, feeds)[0]
    print("logits:", logits.shape)

    table = load_tokens(TOKENS)
    text, kept = ctc_greedy(logits, table)
    print("\n=== 识别结果 ===")
    print(text)
    print("\n前 30 个 token:", [table.get(i, f"#{i}") for i in kept[:30]])
    print("平均帧置信度:", float(logits[0].max(axis=-1).mean()))


if __name__ == "__main__":
    main()
