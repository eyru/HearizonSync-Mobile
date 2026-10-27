package com.hearizon.sync.ui

import com.hearizon.sync.store.Exporter

/**
 * 一行字幕。
 *
 * [startMs] / [durationMs] 是**相对本次会话开始**的偏移，由
 * [com.hearizon.sync.asr.LiveSession] 切句时算出，导出 SRT 时要用。
 */
data class CaptionLine(
    val id: Long,
    val time: String,
    val text: String,
    val tag: String = "",
    val en: String = "",
    val startMs: Long = 0L,
    val durationMs: Long = 0L,
)

/**
 * 转成导出用的 cue。时间轴以最早一条为 0 点，
 * 这样 SRT 从 `00:00:00,000` 开始，符合播放器预期。
 */
fun List<CaptionLine>.toCues(): List<Exporter.Cue> {
    if (isEmpty()) return emptyList()
    val base = minOf { it.startMs }
    return map {
        Exporter.Cue(
            text = it.text,
            startMs = (it.startMs - base).coerceAtLeast(0L),
            durationMs = it.durationMs,
            translation = it.en,
        )
    }
}
