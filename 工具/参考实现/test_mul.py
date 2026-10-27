"""测 opus-mt-mul-en：任意语言 → 英语（用作日/韩/法/德… → 中文的第一跳）。

用法：python test_mul.py "こんにちは、今日はいい天気ですね。"
"""
import sys

import numpy as np
import onnxruntime as ort

D = "opus_mul_en"
EOS, UNK, PAD = 0, 1, 64171


def load_table():
    pieces, scores, jids = [], [], []
    with open(f"{D}/src_spm.txt", encoding="utf-8") as f:
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


def encode(text, pieces, scores, jids):
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
        ids.append(int(jids[j]) if j is not None else UNK)
        pos = prev
    ids.reverse()
    ids.append(EOS)
    return np.array([ids], dtype=np.int64)


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "こんにちは、今日はいい天気ですね。"
    pieces, scores, jids = load_table()
    vocab = [l.rstrip("\n") for l in open(f"{D}/vocab.txt", encoding="utf-8")]

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    enc = ort.InferenceSession(f"{D}/encoder_model_quantized.onnx", so, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(f"{D}/decoder_model_quantized.onnx", so, providers=["CPUExecutionProvider"])

    ids = encode(text, pieces, scores, jids)
    mask = np.ones_like(ids)
    hidden = enc.run(None, {"input_ids": ids, "attention_mask": mask})[0]

    cur = [PAD]
    out = []
    for _ in range(96):
        cur_arr = np.array([cur], dtype=np.int64)
        logits = dec.run(None, {
            "input_ids": cur_arr,
            "encoder_hidden_states": hidden,
            "encoder_attention_mask": mask,
        })[0]
        nxt = int(np.argmax(logits[0, -1]))
        if nxt == EOS:
            break
        out.append(nxt)
        cur.append(nxt)

    en = "".join(vocab[i] for i in out if 0 <= i < len(vocab)).replace("\u2581", " ").strip()
    print("SRC:", text)
    print("EN :", en)


if __name__ == "__main__":
    main()
