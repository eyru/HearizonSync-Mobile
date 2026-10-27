package com.hearizon.sync.asr

/**
 * 判断一句识别结果是什么语言，用来选翻译方向。
 *
 * 为什么要有它：SenseVoice 是多语言模型，说英文时它就会输出英文原文；
 * 如果不管三七二十一都塞给「中→英」模型，英文会被当成中文源文再翻一遍，
 * 出现「说英语、输出英语」的现象。
 *
 * 判据只用字符类别，不引入额外模型（快、不占内存、离线）：
 * - 假名 → 日语；谚文 → 韩语
 * - 汉字占比 ≥15% → 中文（容忍「我们用 Python 写的」这种夹杂）
 * - 拉丁字母占比 ≥50% → 英文
 */
object LangDetect {

    enum class Lang { ZH, EN, JA, KO, OTHER }

    fun detect(text: String): Lang {
        var han = 0
        var kana = 0
        var hangul = 0
        var latin = 0
        var other = 0
        for (ch in text) {
            when {
                ch in '\u3040'..'\u30FF' -> kana++
                ch in '\uAC00'..'\uD7AF' || ch in '\u1100'..'\u11FF' -> hangul++
                ch in '\u4E00'..'\u9FFF' || ch in '\u3400'..'\u4DBF' -> han++
                ch in 'a'..'z' || ch in 'A'..'Z' -> latin++
                ch.isLetter() -> other++
            }
        }
        val total = han + kana + hangul + latin + other
        if (total == 0) return Lang.OTHER
        if (kana > 0 && kana.toFloat() / total > 0.05f) return Lang.JA
        if (hangul > 0 && hangul.toFloat() / total > 0.05f) return Lang.KO
        if (han.toFloat() / total >= 0.15f) return Lang.ZH
        if (latin.toFloat() / total >= 0.5f) return Lang.EN
        return Lang.OTHER
    }
}
