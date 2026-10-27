# HearizonSync Mobile

离线实时语音字幕 + 中英日韩四方向翻译。全部推理在手机本地完成，**不联网也能用**。

- 版本：2.0.0（versionCode 20000）
- 包名：`com.hearizon.sync`
- 最低系统：Android 8.0（API 26），目标 API 34；真机已在 Android 16 验证
- 发布包：`dist/HearizonSync-2.0.0.apk`（734 MB，模型全内置）

> 作者：**何瑞程** · [个人网站](https://eyru.github.io/personalweb/) · [GitHub](https://github.com/eyru/HearizonSync-Mobile)

## 功能

| 功能 | 说明 |
|---|---|
| 实时字幕 | 说话即出字幕，停顿约 0.3 秒自动断句 |
| 四方向翻译 | 中→英、英→中、日→中、韩→中，全部离线；日/韩先到英文再转中文 |
| 顶栏语言胶囊 | 不进设置页就能切方向；切方向会同步锁定识别语言（否则韩语会被判成中文） |
| 噪声幻觉抑制 | 静音/音乐/键盘环境不再刷「嗯。」「我。」「Yeah.」「一」这类假字幕 |
| 句尾降噪 | 帧级噪声门，**默认关闭**——实测对识别率无稳定收益，只在底噪极大时手动开 |
| 字幕导出 | SRT 字幕 / 纯文本 / 中英对照，写入「下载/HearizonSync」，不需要存储权限 |
| 设置页 | 译文开关、字号四档、降噪开关、翻译方向、识别语言、导出时间戳、翻译自测 |
| 整段自测 / 回放验证 | 内置示例音频跑完整链路，或按 20ms 帧喂进真实 VAD 链路 |
| 完全离线 | 模型内置在 APK，首次启动解包一次，之后不再需要网络 |

## 技术栈

- 语言/UI：Kotlin 2.0.21 + Jetpack Compose（Material 3）
- 构建：Gradle 8.9 + AGP 8.7.3，compileSdk 34
- 推理：ONNX Runtime Android 1.19.2
- 语音识别：SenseVoice-Small（int8，239 MB），**自研** kaldi 风格 fbank + LFR + CMVN + CTC 解码
- 翻译：opus-mt 四方向（每个约 110 MB），**自研** sentencepiece 分词 + 贪心生成
- VAD：**自研**滑动窗口 10 分位噪声底（指数平滑会把连续语音切碎，实测 CER 0.514 → 0.000）

> 前端与解码没有使用 sherpa-onnx 等现成运行时，全部按训练侧参数自行实现（见 `docs/架构决策.md`）。

## 目录

```
app/src/main/java/com/hearizon/sync/
├─ MainActivity.kt            入口
├─ audio/AudioCapture.kt      AudioRecord 采集（MIC 音源 + 硬件/软件 AGC，16k/16bit/20ms 帧）
├─ audio/AudioDsp.kt          FFT + mel + fbank
├─ audio/VadEnergy.kt         分位统计噪声底 VAD
├─ audio/Denoiser.kt          帧级噪声门（默认关）
├─ audio/WavReader.kt         WAV 读取（自测/回放）
├─ asr/SenseVoiceEngine.kt    LFR + CMVN + ORT + CTC（顺带取模型自判语种）
├─ asr/TranscriptFilter.kt    幻觉字幕拦截（8 条规则）
├─ asr/LangDetect.kt          文本语种判定
├─ asr/LiveSession.kt         采集→VAD→切句→识别队列
├─ mt/SpmTokenizer.kt         sentencepiece 等价分词 + id 映射
├─ mt/MarianTranslator.kt     encoder/decoder + 贪心（pad 可配，ja-en 是 60715）
├─ store/ModelInstaller.kt    assets → filesDir 解包
├─ store/Prefs.kt             设置持久化
├─ store/Exporter.kt          SRT / TXT / 中英对照导出
└─ ui/                        界面、设置页、主题
```

## 构建

需要 JDK 17 + Android SDK（platforms;android-34、build-tools;34.0.0）+ Gradle 8.9。
`local.properties` 里写 `sdk.dir=...`；本机工具链默认在 `C:\Users\hrcsn\android-toolchain\`。

```bash
python 工具/拉取模型.py    # 首次：下载模型权重到 assets（约 660 MB，不入库）
工具/构建.sh debug         # 调试包
工具/构建.sh release       # 签名发布包 → dist/HearizonSync-<版本>.apk，打印签名与 SHA-256
工具/构建.sh install       # 装到已连接设备
工具/构建.sh clean
```

发布签名配置在同目录 `keystore.properties`（`storeFile` 指向 `工具/keystore/hearizon-release.jks`）。**这两个文件都不可外传**；keystore 丢失后将无法给已安装用户升级。

## 安装与首次启动

1. 安装 APK（约 734 MB，含全部模型）
2. 首次启动会解包约 660 MB 模型到应用私有目录，界面显示进度百分比
3. 授予麦克风权限后点「开始」即可

> 2.0.0 的签名与旧 1.1.0 不同，手机上若还装着 1.1.0，**必须先卸载再装**。

## 模型来源与许可

| 模型 | 来源 | 许可 |
|---|---|---|
| SenseVoice-Small int8 | iic / k2-fsa（FunAudioLLM） | Apache-2.0 |
| opus-mt-zh-en / en-zh / ja-en / ko-en | Helsinki-NLP，Xenova 的 ONNX 量化转换 | CC-BY-4.0 |

模型权重属第三方资产，不入库；本仓库只包含自研代码。

## 已知限制

- 日→中、韩→中要两跳（→英→中），个别句子译文会走样；「小语种→中文」的直译模型公开渠道没有可用转换
- 单字真语音（「好。」「对。」）会被幻觉抑制误杀——宁漏不错是有意取舍
- 不做说话人分离，多人同时说话会混段
- 不做历史记录（导出即归档）
- 体积 734 MB，模型按需下载/更激进量化是后续优化方向

## 文档

- `docs/架构决策.md`：技术选型与理由（ADR-001 ~ ADR-016）
- `docs/进度-恢复点.md`：当前进度、验证记录、踩坑清单
