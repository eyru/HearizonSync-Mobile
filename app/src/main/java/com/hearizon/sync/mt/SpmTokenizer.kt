package com.hearizon.sync.mt

import android.content.Context

/**
 * 自研 SentencePiece（Unigram）编码器。
 *
 * 词表来自方向模型的 source.spm（32000 片），已用 sentencepiece 库导出成
 * assets/<tokenizerDir>/src_spm.txt（每行 `piece<TAB>score<TAB>jsonId`）。
 * 编码流程 = 归一化 → Metaspace(空格与句首换成 ▁) → Viterbi 最优切分 → 末尾补 </s>。
 *
 * @param assetPath 编码表在 assets 里的路径；中→英为 `mt/src_spm.txt`，英→中为 `mt_en_zh/src_spm.txt`
 */
class SpmTokenizer(context: Context, assetPath: String = "mt/src_spm.txt") : AutoCloseable {

    private val pieces: Array<String>
    private val scores: FloatArray
    private val jsonIds: IntArray
    private val pieceIndex: HashMap<String, Int>
    private var maxPieceLen: Int = 16

    init {
        val tmp = ArrayList<Triple<String, Float, Int>>(32768)
        context.assets.open(assetPath).bufferedReader(Charsets.UTF_8).use { r ->
            var line: String? = r.readLine()
            while (line != null) {
                if (line.isNotEmpty()) {
                    val parts = line.split('\t')
                    if (parts.size >= 3) {
                        tmp.add(
                            Triple(
                                parts[0],
                                parts[1].toFloat(),
                                parts[2].toIntOrNull() ?: UNK,
                            )
                        )
                    }
                }
                line = r.readLine()
            }
        }
        pieces = Array(tmp.size) { tmp[it].first }
        scores = FloatArray(tmp.size) { tmp[it].second }
        jsonIds = IntArray(tmp.size) { tmp[it].third }
        // piece -> 词表行号（Viterbi 用），行号再经 jsonIds 映射到模型 id
        val idx = HashMap<String, Int>(tmp.size * 2)
        var maxLen = 1
        for (i in pieces.indices) {
            idx[pieces[i]] = i
            if (pieces[i].length > maxLen) maxLen = pieces[i].length
        }
        pieceIndex = idx
        maxPieceLen = maxLen
    }

    /** 文本 → token id 列表（含结尾 </s>=0）。 */
    fun encode(text: String): LongArray {
        val norm = normalize(text)
        val s = "\u2581" + norm.replace(' ', '\u2581')
        val n = s.length
        if (n == 0) return longArrayOf(EOS.toLong())

        val NEG = -1e18f
        val best = FloatArray(n + 1) { NEG }
        val back = IntArray(n + 1) { -1 }
        best[0] = 0f
        val maxL = minOf(maxPieceLen, n)
        for (i in 0 until n) {
            if (best[i] == NEG) continue
            val lim = minOf(maxL, n - i)
            for (len in 1..lim) {
                val idx = pieceIndex[s.substring(i, i + len)] ?: continue
                val sc = best[i] + scores[idx]
                if (sc > best[i + len]) {
                    best[i + len] = sc
                    back[i + len] = i
                }
            }
        }

        val out = ArrayList<Long>(n)
        var pos = n
        while (pos > 0) {
            var prev = back[pos]
            if (prev < 0) prev = pos - 1
            val piece = s.substring(prev, pos)
            val idx = pieceIndex[piece]
            if (idx != null) {
                out.add(jsonIds[idx].toLong())
            } else {
                for (b in piece.toByteArray(Charsets.UTF_8)) {
                    out.add(UNK.toLong())
                }
            }
            pos = prev
        }
        out.reverse()
        out.add(EOS.toLong())
        return out.toLongArray()
    }

    fun piece(id: Int): String = if (id in pieces.indices) pieces[id] else "<unk>"

    override fun close() {}

    companion object {
        const val PAD = 65000
        const val EOS = 0
        const val UNK = 1

        /** 与 source.spm 的 normalizer 对齐：全角 → 半角；中文标点保留；… 展开成三个点。 */
        fun normalize(text: String): String {
            val sb = StringBuilder(text.length + 8)
            for (ch in text) {
                when (ch) {
                    '，' -> sb.append(',')
                    '！' -> sb.append('!')
                    '？' -> sb.append('?')
                    '；' -> sb.append(';')
                    '：' -> sb.append(':')
                    '（' -> sb.append('(')
                    '）' -> sb.append(')')
                    '％' -> sb.append('%')
                    '＃' -> sb.append('#')
                    '＆' -> sb.append('&')
                    '＊' -> sb.append('*')
                    '＋' -> sb.append('+')
                    '／' -> sb.append('/')
                    '＝' -> sb.append('=')
                    '＠' -> sb.append('@')
                    '｜' -> sb.append('|')
                    '…' -> sb.append("...")
                    in 'Ａ'..'Ｚ' -> sb.append(('A' + (ch - 'Ａ')))
                    in 'ａ'..'ｚ' -> sb.append(('a' + (ch - 'ａ')))
                    in '０'..'９' -> sb.append(('0' + (ch - '０')))
                    '\u3000' -> sb.append(' ')
                    '\n', '\r', '\t' -> sb.append(' ')
                    else -> sb.append(ch)
                }
            }
            return sb.toString()
        }
    }
}
