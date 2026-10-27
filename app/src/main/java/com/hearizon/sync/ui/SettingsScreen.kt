package com.hearizon.sync.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.hearizon.sync.BuildConfig
import com.hearizon.sync.store.Exporter
import com.hearizon.sync.store.Prefs

/**
 * 设置页。
 *
 * 这里每一项都写清「为什么默认是这个值」，因为默认值全部来自实测，
 * 不是拍脑袋：降噪默认关（实测无收益且会放大键盘噪声），
 * 翻译默认开（异步补，不拖慢字幕），字号默认标准。
 */
@Composable
fun SettingsScreen(
    showTranslation: Boolean,
    enhance: Boolean,
    fontScale: Float,
    exportWithTimestamp: Boolean,
    direction: String,
    asrLanguage: Int,
    enZhLoaded: Boolean,
    lines: List<CaptionLine>,
    lastExportPath: String?,
    notice: String,
    selfTestText: String,
    onShowTranslationChange: (Boolean) -> Unit,
    onEnhanceChange: (Boolean) -> Unit,
    onFontScaleChange: (Float) -> Unit,
    onExportTimestampChange: (Boolean) -> Unit,
    onDirectionChange: (String) -> Unit,
    onAsrLanguageChange: (Int) -> Unit,
    onSelfTest: () -> Unit,
    onExport: (Exporter.Format) -> Unit,
    onBack: () -> Unit,
) {
    Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
        Column(modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
            // ---------------- 顶栏
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .statusBarsPadding()
                    .padding(horizontal = 8.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                TextButton(onClick = onBack) { Text("← 返回") }
                Spacer(Modifier.width(4.dp))
                Text(
                    text = "设置",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                )
            }

            // ---------------- 字幕
            SectionTitle("字幕")
            SwitchRow(
                title = "显示译文",
                subtitle = "识别完异步翻译，不拖慢出字速度；中文→英文、英文→中文都会",
                checked = showTranslation,
                onChange = onShowTranslationChange,
            )
            Row(
                modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 6.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text("翻译方向", style = MaterialTheme.typography.bodyLarge)
                    Text(
                        if (enZhLoaded) "自动 = 按这句话的语种选方向" else "自动 = 按语种选方向（英→中首次使用需装载模型）",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
            Column(modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp)) {
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    listOf(
                        Prefs.DIR_AUTO to "自动",
                        Prefs.DIR_ZH2EN to "中→英",
                        Prefs.DIR_EN2ZH to "英→中",
                    ).forEach { (v, name) ->
                        FilterChip(
                            selected = direction == v,
                            onClick = { onDirectionChange(v) },
                            label = { Text(name, style = MaterialTheme.typography.labelSmall) },
                        )
                    }
                }
                Row(
                    modifier = Modifier.padding(top = 2.dp),
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    listOf(
                        Prefs.DIR_JA2ZH to "日→中",
                        Prefs.DIR_KO2ZH to "韩→中",
                    ).forEach { (v, name) ->
                        FilterChip(
                            selected = direction == v,
                            onClick = { onDirectionChange(v) },
                            label = { Text(name, style = MaterialTheme.typography.labelSmall) },
                        )
                    }
                }
            }
            Row(
                modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 6.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text("字号", style = MaterialTheme.typography.bodyLarge)
                    Text(
                        "只影响字幕显示，不影响导出",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    listOf(0.85f to "小", 1.0f to "标准", 1.25f to "大", 1.5f to "特大").forEach { (v, name) ->
                        FilterChip(
                            selected = kotlin.math.abs(fontScale - v) < 0.01f,
                            onClick = { onFontScaleChange(v) },
                            label = { Text(name, style = MaterialTheme.typography.labelSmall) },
                        )
                    }
                }
            }

            Divider()

            // ---------------- 识别
            SectionTitle("识别")
            SwitchRow(
                title = "句尾降噪",
                subtitle = "实测对识别率没有稳定收益，默认关闭；只在底噪极大的环境才可能有用",
                checked = enhance,
                onChange = onEnhanceChange,
            )
            Row(
                modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 6.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text("识别语言", style = MaterialTheme.typography.bodyLarge)
                    Text(
                        "固定成中文会把英语硬解；自动由模型自己判（中/英/日/韩/粤）",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
            Row(
                modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                listOf(0 to "自动", 3 to "中文", 4 to "英文", 11 to "日文", 12 to "韩文").forEach { (v, name) ->
                    FilterChip(
                        selected = asrLanguage == v,
                        onClick = { onAsrLanguageChange(v) },
                        label = { Text(name, style = MaterialTheme.typography.labelSmall) },
                    )
                }
            }
            InfoRow(
                title = "幻觉抑制",
                subtitle = "已内置且不可关闭：噪声里 SenseVoice 会吐出「嗯。」「我。」这类字幕，" +
                    "按段长 + 语气词 + 字数三条规则拦掉。实测音乐 40 条 → 0 条。",
            )
            InfoRow(
                title = "VAD 噪声底",
                subtitle = "滑动窗口 3 秒取 10 分位（不是指数平滑）。同一段 10.3 秒语音：" +
                    "指数平滑切 8 段、CER 0.514；分位法切 4 段、CER 0.029。",
            )

            Divider()

            // ---------------- 导出
            SectionTitle("导出")
            Text(
                text = if (lines.isEmpty()) "还没有字幕，先录一段再导出" else "当前共 ${lines.size} 条字幕",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(horizontal = 20.dp, vertical = 4.dp),
            )
            Row(
                modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 6.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                ExportButton("SRT 字幕", lines.isNotEmpty()) { onExport(Exporter.Format.SRT) }
                ExportButton("纯文本", lines.isNotEmpty()) { onExport(Exporter.Format.TXT) }
                ExportButton("中英对照", lines.isNotEmpty()) { onExport(Exporter.Format.BILINGUAL) }
            }
            SwitchRow(
                title = "文本带时间戳",
                subtitle = "纯文本 / 中英对照导出时，每行前面加 [时:分:秒]；SRT 自带时间轴，不受此开关影响",
                checked = exportWithTimestamp,
                onChange = onExportTimestampChange,
            )
            if (!lastExportPath.isNullOrBlank()) {
                Text(
                    text = "上次导出：$lastExportPath",
                    style = MaterialTheme.typography.labelSmall,
                    fontFamily = FontFamily.Monospace,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(horizontal = 20.dp, vertical = 4.dp),
                )
            }
            if (notice.isNotBlank()) {
                Text(
                    text = notice,
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.padding(horizontal = 20.dp, vertical = 6.dp),
                )
            }

            Divider()

            // ---------------- 翻译自测
            SectionTitle("翻译自测")
            Text(
                text = "四个方向各跑一句固定样例，用来确认模型都在、译文说得通；出问题会把原因原文显示出来。",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(horizontal = 20.dp),
            )
            Row(modifier = Modifier.padding(horizontal = 20.dp, vertical = 6.dp)) {
                OutlinedButton(onClick = onSelfTest) { Text("跑一遍") }
            }
            if (selfTestText.isNotBlank()) {
                Text(
                    text = selfTestText,
                    style = MaterialTheme.typography.bodySmall,
                    fontFamily = FontFamily.Monospace,
                    modifier = Modifier.padding(horizontal = 20.dp, vertical = 6.dp),
                )
            }

            Divider()

            // ---------------- 关于
            SectionTitle("关于")
            InfoRow("版本", "v${BuildConfig.VERSION_NAME}（versionCode ${BuildConfig.VERSION_CODE}）")
            InfoRow(
                "识别模型",
                "SenseVoiceSmall int8 · 239MB · 完全离线 · 中英日韩粤" ,
            )
            InfoRow(
                "翻译模型",
                "Opus-MT 四方向全离线：中→英、英→中、日→英、韩→英（日韩先到英文再转中文）",
            )
            InfoRow(
                "已知限制",
                "① 单字真语音（「好。」「对。」）会被幻觉抑制误杀" +
                    "② 不做说话人分离，多人同时说话会混段" +
                    "③ 日→中、韩→中要两跳（先到英文再转中文），个别句子会走样",
            )

            Divider()

            // ---------------- 作者
            SectionTitle("作者")
            InfoRow("作者", "何瑞程")
            InfoRow("个人网站", "https://eyru.github.io/personalweb/")
            InfoRow("项目地址", "https://github.com/eyru/HearizonSync-Mobile")

            Spacer(Modifier.height(16.dp))
            Text(
                text = "HearizonSync Mobile · 2.0.0",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.outline,
                modifier = Modifier.fillMaxWidth().padding(bottom = 20.dp),
                textAlign = androidx.compose.ui.text.style.TextAlign.Center,
            )
            Spacer(Modifier.navigationBarsPadding())
        }
    }
}

@Composable
private fun SectionTitle(text: String) {
    Text(
        text = text,
        style = MaterialTheme.typography.labelLarge,
        color = MaterialTheme.colorScheme.primary,
        fontWeight = FontWeight.SemiBold,
        modifier = Modifier.padding(start = 20.dp, end = 20.dp, top = 18.dp, bottom = 6.dp),
    )
}

@Composable
private fun Divider() {
    HorizontalDivider(
        modifier = Modifier.padding(horizontal = 20.dp, vertical = 8.dp),
        color = MaterialTheme.colorScheme.surfaceVariant,
    )
}

@Composable
private fun SwitchRow(
    title: String,
    subtitle: String,
    checked: Boolean,
    onChange: (Boolean) -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(modifier = Modifier.weight(1f).padding(end = 12.dp)) {
            Text(title, style = MaterialTheme.typography.bodyLarge)
            Text(
                subtitle,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                lineHeight = MaterialTheme.typography.labelSmall.lineHeight * 1.4f,
            )
        }
        Switch(checked = checked, onCheckedChange = onChange)
    }
}

@Composable
private fun InfoRow(title: String, subtitle: String) {
    Column(modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 6.dp)) {
        Text(title, style = MaterialTheme.typography.bodyLarge)
        Text(
            subtitle,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            lineHeight = MaterialTheme.typography.labelSmall.lineHeight * 1.4f,
        )
    }
}

@Composable
private fun ExportButton(label: String, enabled: Boolean, onClick: () -> Unit) {
    Surface(
        onClick = onClick,
        enabled = enabled,
        shape = RoundedCornerShape(10.dp),
        color = if (enabled) MaterialTheme.colorScheme.surfaceVariant
        else MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.4f),
    ) {
        Box(modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp)) {
            Text(
                text = label,
                style = MaterialTheme.typography.labelMedium,
                color = if (enabled) MaterialTheme.colorScheme.onSurfaceVariant
                else MaterialTheme.colorScheme.outline,
            )
        }
    }
}
