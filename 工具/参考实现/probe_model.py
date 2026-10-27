# -*- coding: utf-8 -*-
"""探查 SenseVoice ONNX 模型的输入输出契约（不猜、直接问模型）。"""
import json
import sys

import onnxruntime as ort

MODEL = r"C:\Program Files\Hearizon Sync\_internal\models\sv\model.int8.onnx"
TOKENS = r"C:\Program Files\Hearizon Sync\_internal\models\sv\tokens.txt"


def main() -> None:
    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta()
    print("== metadata ==")
    print("producer:", meta.producer_name, "| graph:", meta.graph_name)
    print("description:", (meta.description or "")[:300])
    print("custom_metadata_map:", json.dumps(meta.custom_metadata_map, ensure_ascii=False)[:800])

    print("\n== inputs ==")
    for i in sess.get_inputs():
        print(f"  {i.name:24s} {i.type:12s} {i.shape}")

    print("\n== outputs ==")
    for o in sess.get_outputs():
        print(f"  {o.name:24s} {o.type:12s} {o.shape}")

    with open(TOKENS, "r", encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh][:8]
    print("\n== tokens head ==")
    for ln in lines:
        print("  ", ln)


if __name__ == "__main__":
    sys.exit(main())
