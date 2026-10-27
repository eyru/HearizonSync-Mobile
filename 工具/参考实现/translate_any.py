"""通用单向翻译测试：任意 Xenova opus-mt ONNX 目录。

用法：python translate_any.py <模型目录> "文本"
目录内需有：encoder_model_quantized.onnx、decoder_model_quantized.onnx、
           src_spm.txt（三列 piece/score/jsonId）、vocab.txt、config.json
"""
import json
import sys

import numpy as np
import onnxruntime as ort


def load_table(d):
    pieces, scores, jids = [], [], []
    with open(f"{d}/src_spm.txt", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            a = line.split("\t")
            if len(a) < 3:
                continue
            pieces.append(a[0])
            scores.append(float(a[1]))
            jids.append(int(a[2]))
    return pieces, np.array(scores, dtype=np.float32), np.array(jids, dtype=np.int64)


def normalize(t: str) -> str:
    mp = {"，": ",", "！": "!", "？": "?", "；": ";", "：": ":", "（": "(", "）": ")",
          "％": "%", "＃": "#", "＆": "&", "＊": "*", "＋": "+", "／": "/", "＝": "=",
          "＠": "@", "｜": "|", "…": "...", "\u3000": " "}
    out = []
    for ch in t:
        if ch in mp:
            out.append(mp[ch])
        elif "Ａ" <= ch <= "Ｚ":
            out.append(chr(ord("A") + ord(ch) - ord("Ａ")))
        elif "ａ" <= ch <= "ｚ":
            out.append(chr(ord("a") + ord(ch) - ord("ａ")))
        elif "０" <= ch <= "９":
            out.append(chr(ord("0") + ord(ch) - ord("０")))
        elif ch in "\n\r\t":
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def encode(text, pieces, scores, jids, eos=0, unk=1):
    idx = {p: i for i, p in enumerate(pieces)}
    maxlen = max(len(p) for p in pieces)
    s = "\u2581" + normalize(text).replace(" ", "\u2581")
    n = len(s)
    best = np.full(n + 1, -1e18, dtype=np.float32)
    back = np.full(n + 1, -1, dtype=np.int32)
    best[0] = 0.0
    for i in range(n):
        if best[i] == -1e18:
            continue
        for ln in range(1, min(maxlen, n - i) + 1):
            p = s[i:i + ln]
            j = idx.get(p)
            if j is None:
                continue
            sc = best[i] + scores[j]
            if sc > best[i + ln]:
                best[i + ln] = sc
                back[i + ln] = i
    ids, pos = [], n
    while pos > 0:
        prev = int(back[pos])
        if prev < 0:
            prev = pos - 1
        j = idx.get(s[prev:pos])
        ids.append(int(jids[j]) if j is not None else unk)
        pos = prev
    ids.reverse()
    ids.append(eos)
    return np.array([ids], dtype=np.int64)


def _pick(d, *names):
    import os
    for n in names:
        p = os.path.join(d, n)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{d}: {' / '.join(names)}")


def translate(d, text, max_new=96, pad=None):
    cfg = json.load(open(f"{d}/config.json", encoding="utf-8"))
    if pad is None:
        pad = cfg.get("pad_token_id", cfg.get("decoder_start_token_id"))
    eos = cfg.get("eos_token_id", 0)
    pieces, scores, jids = load_table(d)
    vocab = [l.rstrip("\n") for l in open(f"{d}/vocab.txt", encoding="utf-8")]
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    enc = ort.InferenceSession(_pick(d, "encoder_model_quantized.onnx", "encoder.onnx"), so,
                               providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(_pick(d, "decoder_model_quantized.onnx", "decoder.onnx"), so,
                               providers=["CPUExecutionProvider"])
    ids = encode(text, pieces, scores, jids, eos=eos)
    mask = np.ones_like(ids)
    hidden = enc.run(None, {"input_ids": ids, "attention_mask": mask})[0]
    cur = [pad]
    out = []
    for _ in range(max_new):
        logits = dec.run(None, {
            "input_ids": np.array([cur], dtype=np.int64),
            "encoder_hidden_states": hidden,
            "encoder_attention_mask": mask,
        })[0]
        nxt = int(np.argmax(logits[0, -1]))
        if nxt == eos:
            break
        out.append(nxt)
        cur.append(nxt)
    return "".join(vocab[i] for i in out if 0 <= i < len(vocab)).replace("\u2581", " ").strip()


if __name__ == "__main__":
    d = sys.argv[1]
    txt = sys.argv[2] if len(sys.argv) > 2 else "こんにちは、今日はいい天気ですね。"
    print("SRC:", txt)
    print("OUT:", translate(d, txt))
