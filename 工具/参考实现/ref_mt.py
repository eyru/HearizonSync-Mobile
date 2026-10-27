"""opus-mt-zh-en 参考实现（v3）：sentencepiece 编码 + beam search 解码。
用法: python ref_mt.py "中文" [--max 60] [--beam 4] [--lenpen 1.0]
"""
import json
import os
import sys

import numpy as np
import onnxruntime as ort
import sentencepiece as spm

HERE = "D:/Audivox事务/Hearizon Sync手机版/工具/参考实现/opus"
PAD_ID = 65000
EOS_ID = 0


def load():
    os.chdir(HERE)
    with open("tokenizer.json", encoding="utf-8") as f:
        t = json.load(f)
    vocab = t["model"]["vocab"]
    tok2id = {tok: i for i, (tok, _sc) in enumerate(vocab)}
    id2tok = [tok for tok, _sc in vocab]
    src_sp = spm.SentencePieceProcessor(model_file="source.spm")
    return tok2id, id2tok, src_sp


def decode_ids(ids, id2tok):
    return "".join(id2tok[i] for i in ids if i not in (EOS_ID, PAD_ID)).replace("\u2581", " ").strip()


def beam_search(dec, hid, mask, beam=4, max_new=60, lenpen=1.0):
    """返回 (best_ids, best_score)。每步对所有 beam 跑一次完整 decoder。"""
    beams = [([PAD_ID], 0.0)]          # (ids, logprob_sum)
    done = []
    for step in range(max_new):
        cands = []
        for ids, score in beams:
            logits = dec.run(["logits"], {
                "encoder_attention_mask": mask,
                "input_ids": np.array([ids], dtype=np.int64),
                "encoder_hidden_states": hid,
            })[0][0, -1]
            logp = logits - np.log(np.exp(logits - logits.max()).sum()) - logits.max() + logits.max()
            # 数值稳定 log_softmax
            m = logits.max()
            logp = logits - m - np.log(np.exp(logits - m).sum())
            top = np.argsort(-logp)[: beam * 2]
            for tid in top:
                tid = int(tid)
                if tid == PAD_ID:
                    continue
                cands.append((ids + [tid], score + float(logp[tid])))
        if not cands:
            break
        cands.sort(key=lambda x: x[1], reverse=True)
        beams = []
        for ids, score in cands:
            if ids[-1] == EOS_ID:
                done.append((ids, score / ((len(ids) - 1) ** lenpen)))
            else:
                beams.append((ids, score))
            if len(beams) >= beam:
                break
        if not beams:
            break
    for ids, score in beams:
        done.append((ids, score / ((len(ids) - 1) ** lenpen)))
    if not done:
        return [], 0.0
    done.sort(key=lambda x: x[1], reverse=True)
    ids, score = done[0]
    return [i for i in ids if i != EOS_ID], score


def main():
    argv = sys.argv[1:]
    opts = {"--max": 60, "--beam": 4, "--lenpen": 1.0}
    text = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in opts:
            opts[a] = float(argv[i + 1]) if a == "--lenpen" else int(argv[i + 1])
            i += 2
        else:
            text = a
            i += 1
    text = text or "大家好，今天天气不错。"
    max_new, beam, lenpen = int(opts["--max"]), int(opts["--beam"]), float(opts["--lenpen"])

    tok2id, id2tok, src_sp = load()
    pieces = src_sp.encode(text, out_type=str)
    ids = [tok2id[p] for p in pieces] + [EOS_ID]   # Marian 要求源序列以 </s> 收尾
    print(f"输入: {text}")
    print(f"SPM({len(pieces)}): {pieces}")

    enc = ort.InferenceSession(f"{HERE}/encoder_model_quantized.onnx",
                               providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(f"{HERE}/decoder_model_quantized.onnx",
                               providers=["CPUExecutionProvider"])
    src = np.array([ids], dtype=np.int64)
    mask = np.ones_like(src)
    hid = enc.run(None, {"input_ids": src, "attention_mask": mask})[0]

    best, score = beam_search(dec, hid, mask, beam=beam, max_new=max_new, lenpen=lenpen)
    print(f"译文: {decode_ids(best, id2tok)}")
    print(f"(score={score:.3f}, beam={beam}, tokens={len(best)})")


if __name__ == "__main__":
    main()
