#!/usr/bin/env python3
"""拉取 HearizonSync Mobile 需要的全部第三方模型权重，并生成 assets 目录。

用法（在仓库根目录）：
    python 工具/拉取模型.py

可选镜像：默认 hf-mirror.com；设 HF_ENDPOINT=https://huggingface.co 可走官方站。

产物：
    app/src/main/assets/models/sv/model.int8.onnx   239 MB   SenseVoice-Small int8
    app/src/main/assets/models/sv/tokens.txt
    app/src/main/assets/models/mt/{encoder,decoder}.onnx     中→英
    app/src/main/assets/mt/{src_spm,vocab}.txt
    app/src/main/assets/models/mt_en_zh/...                  英→中
    app/src/main/assets/mt_en_zh/...
    app/src/main/assets/models/mt_ja_en/...                  日→英
    app/src/main/assets/mt_ja_en/...
    app/src/main/assets/models/mt_ko_en/...                  韩→英
    app/src/main/assets/mt_ko_en/...

这些权重都是第三方资产（SenseVoice: Apache-2.0；opus-mt: CC-BY-4.0），体积超过
GitHub 单文件上限，所以不入库，由本脚本按需拉取。
"""
import os
import subprocess
import sys
import urllib.request

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "app", "src", "main", "assets")
TMP = os.path.join(ROOT, "工具", "参考实现")

SV_REPO = "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
MT = {
    "zh_en": "Xenova/opus-mt-zh-en",
    "en_zh": "Xenova/opus-mt-en-zh",
    "ja_en": "Xenova/opus-mt-ja-en",
    "ko_en": "Xenova/opus-mt-ko-en",
}


def fetch(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        print(f"  已存在 {os.path.relpath(dest, ROOT)}")
        return dest
    print(f"  下载 {url}")
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                print(f"\r    {pct:3d}%  {done / 1048576:7.1f} / {total / 1048576:.1f} MB", end="")
    print()
    os.replace(tmp, dest)
    return dest


def hf(repo, path):
    return f"{ENDPOINT}/{repo}/resolve/main/{path}"


def main():
    print("== SenseVoice ==")
    sv_dir = os.path.join(ASSETS, "models", "sv")
    fetch(hf(SV_REPO, "model.int8.onnx"), os.path.join(sv_dir, "model.int8.onnx"))
    fetch(hf(SV_REPO, "tokens.txt"), os.path.join(sv_dir, "tokens.txt"))

    exporter = os.path.join(TMP, "导出分词表.py")
    if not os.path.exists(exporter):
        sys.exit(f"缺少 {exporter}（生成分词表要用）")

    for key, repo in MT.items():
        print(f"== {repo} ==")
        work = os.path.join(TMP, f"opus_{key}")
        enc = fetch(hf(repo, "onnx/encoder_model_quantized.onnx"), os.path.join(work, "encoder.onnx"))
        fetch(hf(repo, "onnx/decoder_model_quantized.onnx"), os.path.join(work, "decoder.onnx"))
        fetch(hf(repo, "source.spm"), os.path.join(work, "source.spm"))
        fetch(hf(repo, "tokenizer.json"), os.path.join(work, "tokenizer.json"))

        if key == "zh_en":
            model_dir, tok_dir = os.path.join(ASSETS, "models", "mt"), os.path.join(ASSETS, "mt")
        else:
            model_dir, tok_dir = os.path.join(ASSETS, "models", f"mt_{key}"), os.path.join(ASSETS, f"mt_{key}")
        os.makedirs(model_dir, exist_ok=True)
        for name in ("encoder.onnx", "decoder.onnx"):
            dst = os.path.join(model_dir, name)
            if not os.path.exists(dst):
                os.replace(os.path.join(work, name), dst)

        subprocess.run(
            [sys.executable, exporter, os.path.join(work, "source.spm"), os.path.join(work, "tokenizer.json"), tok_dir],
            check=True,
        )

    print("\n完成。可以开始构建：工具/构建.sh release")


if __name__ == "__main__":
    main()
