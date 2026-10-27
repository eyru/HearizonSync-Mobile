"""为 MarianMT 方向模型导出 Kotlin 侧要用的两张词表。

导出物（与 `app/src/main/assets/mt/` 同构）：
1. `src_spm.txt` —— 三列：`piece <TAB> score <TAB> jsonId`
   sentencepiece 的 piece/score 用于 Viterbi 分词；jsonId 是 tokenizer.json 里的 id，
   喂给模型必须用后者（源 spm 的行号 ≠ tokenizer.json 的 id，实测差 4320）。
2. `vocab.txt` —— 65001 行，第 n 行 = id n 的 token 文本，供解码端拼回字符串。

用法：
    python 导出分词表.py <source.spm> <tokenizer.json> <输出目录>
"""
import json
import os
import sys

import sentencepiece as spm


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    spm_path, tok_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]

    sp = spm.SentencePieceProcessor(model_file=spm_path)
    tok = json.load(open(tok_path, encoding="utf-8"))
    vocab = tok["model"]["vocab"]  # [[piece, score], ...]，顺序即 id

    piece2id = {}
    for idx, item in enumerate(vocab):
        piece2id[item[0]] = idx

    os.makedirs(out_dir, exist_ok=True)

    # ---- src_spm.txt ----
    missing = 0
    with open(os.path.join(out_dir, "src_spm.txt"), "w", encoding="utf-8", newline="\n") as f:
        for i in range(sp.get_piece_size()):
            piece = sp.id_to_piece(i)
            score = sp.get_score(i)
            jid = piece2id.get(piece)
            if jid is None:
                # tokenizer.json 里没有这个 piece → 归到 <unk>(1)
                jid = 1
                missing += 1
            f.write(f"{piece}\t{score}\t{jid}\n")

    # ---- vocab.txt（id → token 文本）----
    with open(os.path.join(out_dir, "vocab.txt"), "w", encoding="utf-8", newline="\n") as f:
        for piece, _score in vocab:
            f.write(piece + "\n")

    print(f"spm pieces   : {sp.get_piece_size()}")
    print(f"tokenizer ids: {len(vocab)}")
    print(f"缺失映射     : {missing} 个 piece → <unk>(1)")
    print(f"输出         : {out_dir}")
    for name in ("src_spm.txt", "vocab.txt"):
        p = os.path.join(out_dir, name)
        print(f"  {name}: {os.path.getsize(p):,} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
