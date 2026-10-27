package com.hearizon.sync.store

import android.content.Context

/**
 * 用户偏好。用 SharedPreferences 就够，不值得引入 DataStore。
 *
 * 所有默认值都按「实测最优」定，而不是按「功能最多」定：
 * - [enhance] 默认 false —— 降噪对识别率没有稳定收益，见 Denoiser 注释里的实测表
 * - [showTranslation] 默认 true —— 翻译是产品卖点，且不拖慢字幕（异步补）
 * - [fontScale] 默认 1.0
 */
class Prefs(context: Context) {

    private val sp = context.applicationContext.getSharedPreferences("hearizon", Context.MODE_PRIVATE)

    /** 显示英文译文 */
    var showTranslation: Boolean
        get() = sp.getBoolean(KEY_TRANSLATION, true)
        set(v) = sp.edit().putBoolean(KEY_TRANSLATION, v).apply()

    /** 句尾降噪 + 增益归一 */
    var enhance: Boolean
        get() = sp.getBoolean(KEY_ENHANCE, false)
        set(v) = sp.edit().putBoolean(KEY_ENHANCE, v).apply()

    /** 字幕字号倍率：0.85 / 1.0 / 1.25 / 1.5 */
    var fontScale: Float
        get() = sp.getFloat(KEY_FONT, 1.0f)
        set(v) = sp.edit().putFloat(KEY_FONT, v).apply()

    /** 导出时是否带时间戳 */
    var exportWithTimestamp: Boolean
        get() = sp.getBoolean(KEY_EXPORT_TS, true)
        set(v) = sp.edit().putBoolean(KEY_EXPORT_TS, v).apply()

    /** 最近一次导出路径，设置页显示「上次导出到…」 */
    var lastExportPath: String?
        get() = sp.getString(KEY_LAST_EXPORT, null)
        set(v) = sp.edit().putString(KEY_LAST_EXPORT, v).apply()

    /**
     * 翻译方向：
     * - `auto`（默认）按识别结果的语言自动选中→英 / 英→中
     * - `zh2en` 强制中→英
     * - `en2zh` 强制英→中
     */
    var translateDirection: String
        get() = sp.getString(KEY_DIRECTION, DIR_AUTO) ?: DIR_AUTO
        set(v) = sp.edit().putString(KEY_DIRECTION, v).apply()

    /**
     * 识别语言（SenseVoice 的 language 参数）：
     * 0=自动、3=中文、4=英文、7=粤语、11=日文、12=韩文。
     * 默认自动——否则说英语会被中文模型硬解。
     */
    var asrLanguage: Int
        get() = sp.getInt(KEY_ASR_LANG, 0)
        set(v) = sp.edit().putInt(KEY_ASR_LANG, v).apply()

    companion object {
        private const val KEY_TRANSLATION = "show_translation"
        private const val KEY_ENHANCE = "enhance"
        private const val KEY_FONT = "font_scale"
        private const val KEY_EXPORT_TS = "export_timestamp"
        private const val KEY_LAST_EXPORT = "last_export"
        private const val KEY_DIRECTION = "translate_direction"
        private const val KEY_ASR_LANG = "asr_language"

        const val DIR_AUTO = "auto"
        const val DIR_ZH2EN = "zh2en"
        const val DIR_EN2ZH = "en2zh"
        const val DIR_JA2ZH = "ja2zh"
        const val DIR_KO2ZH = "ko2zh"
    }
}
