# -*- coding: utf-8 -*-
"""把 SenseVoice 的 custom metadata 完整摊开：前端参数、CMVN、语种映射全在里面。"""
import onnxruntime as ort

MODEL = r"C:\Program Files\Hearizon Sync\_internal\models\sv\model.int8.onnx"


def main() -> None:
    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    print("keys:", list(meta.keys()))
    for key, value in meta.items():
        print(f"\n-- {key} (chars={len(value)}) --")
        if len(value) < 700:
            print(value)
        else:
            parts = value.split(",")
            print("count:", len(parts))
            print("head:", parts[:5])
            print("tail:", parts[-3:])


if __name__ == "__main__":
    main()
