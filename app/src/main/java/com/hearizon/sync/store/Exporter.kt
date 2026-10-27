package com.hearizon.sync.store

import android.content.ContentValues
import android.content.Context
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * 字幕导出：SRT 字幕 / 纯文本 / 双语文本。
 *
 * 一段字幕对应 [Exporter.Cue]：
 * - SRT 需要序号 + `HH:MM:SS,mmm --> HH:MM:SS,mmm` + 文本（可选双语两行）
 * - TXT 只要文本，可选前缀 `[HH:MM:SS]`
 *
 * 写文件分两条路：
 * - Android 10+ 走 MediaStore，落到 `Movies/HearizonSync/`，**不需要任何存储权限**；
 * - Android 8/9 落到 app 私有外部目录 `Android/data/com.hearizon.sync/files/exports/`，
 *   同样不需要权限（但文件管理器里不太显眼，所以路径要在 UI 上写清楚）。
 */
object Exporter {

    /** 一条字幕。时间相对录音开始。 */
    data class Cue(
        val text: String,
        val startMs: Long,
        val durationMs: Long,
        val translation: String = "",
    )

    enum class Format(val ext: String, val label: String) {
        SRT("srt", "SRT 字幕"),
        TXT("txt", "纯文本"),
        BILINGUAL("txt", "中英对照"),
    }

    /** 导出结果，UI 用它显示成功路径或错误。 */
    data class Result(val ok: Boolean, val path: String, val message: String)

    private val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US)

    // ------------------------------------------------------------ 内容生成

    /** `00:01:23,456` */
    fun srtTime(ms: Long): String {
        val safe = ms.coerceAtLeast(0)
        val h = safe / 3_600_000
        val m = safe % 3_600_000 / 60_000
        val s = safe % 60_000 / 1000
        val f = safe % 1000
        return String.format(Locale.US, "%02d:%02d:%02d,%03d", h, m, s, f)
    }

    /** `00:01:23` */
    fun clockTime(ms: Long): String {
        val safe = ms.coerceAtLeast(0)
        return String.format(
            Locale.US, "%02d:%02d:%02d",
            safe / 3_600_000, safe % 3_600_000 / 60_000, safe % 60_000 / 1000,
        )
    }

    fun buildSrt(cues: List<Cue>, bilingual: Boolean = false): String {
        val sb = StringBuilder(cues.size * 64)
        cues.forEachIndexed { i, cue ->
            val end = cue.startMs + cue.durationMs.coerceAtLeast(300)
            sb.append(i + 1).append('\n')
            sb.append(srtTime(cue.startMs)).append(" --> ").append(srtTime(end)).append('\n')
            sb.append(cue.text.trim()).append('\n')
            if (bilingual && cue.translation.isNotBlank()) {
                sb.append(cue.translation.trim()).append('\n')
            }
            sb.append('\n')
        }
        return sb.toString()
    }

    fun buildText(cues: List<Cue>, withTimestamp: Boolean, bilingual: Boolean = false): String {
        val sb = StringBuilder(cues.size * 32)
        for (cue in cues) {
            if (withTimestamp) sb.append('[').append(clockTime(cue.startMs)).append("] ")
            sb.append(cue.text.trim())
            if (bilingual && cue.translation.isNotBlank()) {
                sb.append(" / ").append(cue.translation.trim())
            }
            sb.append('\n')
        }
        return sb.toString()
    }

    fun build(cues: List<Cue>, format: Format, withTimestamp: Boolean): String = when (format) {
        Format.SRT -> buildSrt(cues, bilingual = false)
        Format.TXT -> buildText(cues, withTimestamp, bilingual = false)
        Format.BILINGUAL -> buildText(cues, withTimestamp, bilingual = true)
    }

    fun fileName(format: Format): String =
        "HearizonSync-${stamp.format(Date())}.${format.ext}"

    // ------------------------------------------------------------ 落盘

    /**
     * @return 成功时 path 是用户能看懂的位置描述
     */
    fun export(context: Context, cues: List<Cue>, format: Format, withTimestamp: Boolean): Result {
        if (cues.isEmpty()) return Result(false, "", "还没有字幕可以导出")
        val name = fileName(format)
        val content = build(cues, format, withTimestamp)
        return runCatching {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                saveToMediaStore(context, name, format, content)
            } else {
                saveToAppDir(context, name, content)
            }
        }.getOrElse { e ->
            Result(false, "", "导出失败：${e.message ?: e}")
        }
    }

    private fun saveToMediaStore(
        context: Context,
        name: String,
        format: Format,
        content: String,
    ): Result {
        val resolver = context.contentResolver
        val mime = if (format == Format.SRT) "application/x-subrip" else "text/plain"
        val values = ContentValues().apply {
            put(MediaStore.Downloads.DISPLAY_NAME, name)
            put(MediaStore.Downloads.MIME_TYPE, mime)
            put(MediaStore.Downloads.RELATIVE_PATH, "${Environment.DIRECTORY_DOWNLOADS}/HearizonSync")
            put(MediaStore.Downloads.IS_PENDING, 1)
        }
        val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
            ?: return Result(false, "", "系统没给分配文件位置")
        resolver.openOutputStream(uri)?.use { out ->
            out.write(content.toByteArray(Charsets.UTF_8))
        } ?: return Result(false, "", "打不开输出流")
        values.clear()
        values.put(MediaStore.Downloads.IS_PENDING, 0)
        resolver.update(uri, values, null, null)
        return Result(true, "下载/HearizonSync/$name", "已导出到「下载/HearizonSync」")
    }

    private fun saveToAppDir(context: Context, name: String, content: String): Result {
        val base = context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS)
            ?: File(context.filesDir, "exports")
        val dir = File(base, "exports").apply { mkdirs() }
        val file = File(dir, name)
        file.writeText(content, Charsets.UTF_8)
        return Result(true, file.absolutePath, "已导出到 $name")
    }
}
