#!/usr/bin/env bash
# Hearizon Sync 手机版 · 一键构建
# 用法：工具/构建.sh [debug|release|install|clean]
set -euo pipefail

export JAVA_HOME="C:/Users/hrcsn/android-toolchain/jdk-17.0.2"
export ANDROID_HOME="C:/Users/hrcsn/android-toolchain/sdk"
export ANDROID_SDK_ROOT="$ANDROID_HOME"
export PATH="$JAVA_HOME/bin:$PATH"
GRADLE="/c/Users/hrcsn/android-toolchain/gradle-8.9/bin/gradle"

cd "$(dirname "$0")/.."

version_name() {
  grep -oE 'versionName = "[^"]+"' app/build.gradle.kts | head -1 | cut -d'"' -f2
}

case "${1:-debug}" in
  debug)
    "$GRADLE" --no-daemon :app:assembleDebug
    ;;
  release)
    "$GRADLE" --no-daemon :app:assembleRelease
    APK="app/build/outputs/apk/release/app-release.apk"
    VER="$(version_name)"
    OUT="dist/HearizonSync-${VER}.apk"
    mkdir -p dist
    cp "$APK" "$OUT"
    echo
    echo "=== 发布包 ==="
    ls -l "$OUT"
    cmd //c "$(cygpath -w "$ANDROID_HOME/build-tools/34.0.0/apksigner.bat") verify --print-certs $(cygpath -w "$OUT")" | head -4
    sha256sum "$OUT"
    ;;
  install)
    "$GRADLE" --no-daemon :app:installDebug
    ;;
  clean)
    "$GRADLE" --no-daemon clean
    ;;
  *)
    echo "用法: 工具/构建.sh [debug|release|install|clean]" >&2
    exit 2
    ;;
esac
