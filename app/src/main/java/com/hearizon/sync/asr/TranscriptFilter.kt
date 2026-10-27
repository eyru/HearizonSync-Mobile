package com.hearizon.sync.asr

/**
 * 字幕幻觉过滤器。
 *
 * ## 为什么必须有这一层
 *
 * SenseVoice 是 CTC 模型，输出分布上**不存在「什么都不输出」这个选项**——
 * 即使输入是纯静音，argmax 也会挑出概率最高的 token 拼成字。
 * 实测（`工具/参考实现/eval_hallucination.py`、`eval_speechlike.py`、`eval_final.py` 场景 B）：
 *
 * | 输入 | 20 秒内产生的字幕 |
 * |---|---|
 * | 纯白噪 -30 dB | 「我。」「我。」「我。」… |
 * | 全零静音 | 「我。」 |
 * | 音乐（类语音） | **40 段全部「嗯。」**（每段 0.32–0.60 s） |
 * | 电视人声 | 「嗯嗯。」「嗯。」「我。」 |
 * | 键盘敲击 | 「嗯。」「对。」「对。」 |
 *
 * 而真语音产物里，语气词占比为 **0.00**，且段长普遍 ≥ 0.98 s。
 * 这个反差就是过滤器的全部依据。
 *
 * ## 真机补充（主人反馈：不说话时蹦 «yeah»、«一»、日语韩语单字）
 *
 * 第一版只拦「核心字符 < 2 且段长短于 1.2 秒」+「全是中文语气词」，
 * 真机上漏了三类：
 * 1. 英文填充词——`yeah` / `oh` / `okay` 这类，字面长度够，中文语气词表管不到；
 * 2. 单字但段长被拉长——呼吸/键盘尾音把段拖到 1.2 秒以上，单字规则就失效了；
 * 3. 日语/韩语单字符——`い`、`어` 不在中文语气词表里。
 *
 * 现在改成「不管段长，先按内容强度判」，并有模型自判语种做旁证。
 */
object TranscriptFilter {

    /** 短段阈值：噪声幻觉段长实测 0.32–0.60 s，真人拖长音一般 > 1.2 s */
    const val SHORT_SEGMENT_MS = 1_200

    /** 极短段：低于这个长度还说不出 3 个核心字符，基本是噪声 */
    const val VERY_SHORT_MS = 700

    /** 少于这个核心字符数就判定为噪声产物 */
    const val MIN_CORE_CHARS = 2

    /** 汉语语气词 / 感叹词。全由这些字构成的短句在噪声里高频出现，在正常语音里几乎不出现。 */
    private val FILLERS: Set<Char> = setOf(
        // 单韵母感叹
        '嗯', '啊', '哦', '呃', '唉', '诶', '呀', '哈', '嘿', '噢', '喔', '呐', '哪',
        // 语气助词
        '吧', '嘛', '呗', '咧', '咯', '啦', '嘞', '哟', '嗷', '唔', '哼', '咦', '哇',
    )

    /**
     * 英文填充词（单个词成段时一律视为噪声产物）。
     *
     * 注意只收「不承载信息」的词：`hi` / `hello` / `no` / `yes` 这类能独立成立的应答没有收，
     * 免得把真人一句「Hello.」也吃掉。
     */
    private val FILLER_WORDS: Set<String> = setOf(
        "yeah", "yeh", "ya", "yep", "yup", "oh", "ah", "uh", "um", "hmm", "mm", "mhm",
        "mmm", "ohh", "ahh", "ok", "okay", "wow", "hey", "huh", "eh", "er", "ha", "haha",
        "hehe", "huh", "thanks", "thank", "well", "so", "and", "the", "a", "i",
    )

    /** 去掉标点、空白、符号，只留「信息字符」。 */
    fun core(text: String): String {
        val sb = StringBuilder(text.length)
        for (ch in text) {
            if (ch.isLetterOrDigit()) sb.append(ch)
        }
        return sb.toString()
    }

    /**
     * @param text 识别结果
     * @param segmentMs 该段音频时长（毫秒）
     * @param expectLang 本次识别请求的语种（`zh`/`en`/`ja`/`ko`/`yue`，null 表示自动）
     * @param detectedLang 模型自己判定的语种（输出首部 `<|zh|>` 之类标签）
     * @return true 表示这是幻觉，应当丢弃
     */
    fun isHallucination(
        text: String,
        segmentMs: Int,
        expectLang: String? = null,
        detectedLang: String? = null,
    ): Boolean {
        val c = core(text)
        if (c.isEmpty()) return true                       // 只有标点

        // 1. 单字符产物（「我」「一」「い」「어」）—— 不论段长多长，一律丢
        if (c.length < MIN_CORE_CHARS) return true

        // 2. 同一个字反复（「一一一」「呃呃呃」）→ 去重后就没信息了
        if (c.toSet().size < 2) return true

        // 3. 全中文语气词
        if (c.all { it in FILLERS }) return true

        // 4. 英文填充词（单个词成段）
        if (c.lowercase() in FILLER_WORDS) return true

        // 5. 模型自判语种与请求语种不符，且内容很短 → 多半是噪声被硬解成了别的语言
        if (expectLang != null && detectedLang != null && detectedLang.isNotEmpty() &&
            expectLang != detectedLang && c.length < 4
        ) {
            return true
        }

        // 6. 两字产物里夹着语气词（「啊一」「嗯呀」「一哦」）→ 噪声
        if (segmentMs < 2_000 && c.length == 2 && c.any { it in FILLERS }) return true

        // 7. 极短段（< 0.7 s）还只有两个字 → 噪声
        if (segmentMs < VERY_SHORT_MS && c.length < 3) return true

        // 8. 短段（< 1.2 s）且核心字符 < 3 且全是语气词 + 标点 → 噪声
        if (segmentMs < SHORT_SEGMENT_MS && c.length < 3) {
            return c.all { it in FILLERS }
        }
        return false
    }
}
