package com.hearizon.sync.asr

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.hearizon.sync.audio.AudioDsp
import java.io.File
import java.nio.FloatBuffer
import java.nio.IntBuffer
import kotlin.math.ceil

/**
 * SenseVoice-small 的端侧推理引擎（自研前端 + CTC 解码，不依赖 sherpa-onnx）。
 *
 * 契约（由 `工具/参考实现/probe_meta.py` 实测）：
 *  - 输入 x: float32 [1, T, 560]（80 fbank × LFR 7）
 *  - 输入 x_length / language / text_norm: int32 [1]
 *  - 输出 logits: float32 [1, T, 25055]
 *  - metadata 里带 LFR 参数与 CMVN 的 neg_mean / inv_stddev
 */
class SenseVoiceEngine(
    modelPath: String,
    tokensPath: String,
) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val lfrM: Int
    private val lfrN: Int
    private val negMean: FloatArray
    private val invStd: FloatArray
    private val tokens: Map<Int, String>

    init {
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(modelPath, options)

        val meta = session.metadata.customMetadata
        lfrM = meta["lfr_window_size"]?.trim()?.toInt() ?: 7
        lfrN = meta["lfr_window_shift"]?.trim()?.toInt() ?: 6
        negMean = parseVec(meta["neg_mean"] ?: error("模型缺少 neg_mean"))
        invStd = parseVec(meta["inv_stddev"] ?: error("模型缺少 inv_stddev"))
        require(negMean.size == invStd.size) { "CMVN 维度不一致" }
        tokens = loadTokens(File(tokensPath))
    }

    private fun parseVec(raw: String): FloatArray =
        raw.split(',').map { it.trim().toFloat() }.toFloatArray()

    private fun loadTokens(file: File): Map<Int, String> {
        val map = HashMap<Int, String>(26000)
        file.forEachLine { line ->
            if (line.isNotEmpty()) {
                val idx = line.lastIndexOf(' ')
                if (idx > 0) {
                    val id = line.substring(idx + 1).trim().toIntOrNull()
                    if (id != null) map[id] = line.substring(0, idx)
                }
            }
        }
        return map
    }

    // ------------------------------------------------------------ LFR + CMVN
    /** 低帧率：每 lfrN 帧取 lfrM 帧拼成一行；左填 (lfrM-1)/2 帧，尾部复制末帧。 */
    private fun applyLfr(feats: Array<DoubleArray>): Array<FloatArray> {
        val t = feats.size
        val dim = feats[0].size
        val tLfr = ceil(t.toDouble() / lfrN).toInt()
        val left = (lfrM - 1) / 2

        // padded = [首帧 × left] + 原帧
        val padded = Array(t + left) { i -> if (i < left) feats[0] else feats[i - left] }
        val total = padded.size

        return Array(tLfr) { i ->
            val start = i * lfrN
            val row = FloatArray(dim * lfrM)
            if (start + lfrM <= total) {
                for (m in 0 until lfrM) {
                    val src = padded[start + m]
                    for (d in 0 until dim) row[m * dim + d] = src[d].toFloat()
                }
            } else {
                var off = 0
                var m = start
                while (m < total) {
                    val src = padded[m]
                    for (d in 0 until dim) row[off++] = src[d].toFloat()
                    m++
                }
                val last = padded[total - 1]
                while (off < row.size) {
                    row[off++] = last[off % dim].toFloat()
                }
            }
            row
        }
    }

    private fun applyCmvn(feats: Array<FloatArray>): Array<FloatArray> =
        Array(feats.size) { i ->
            val src = feats[i]
            FloatArray(src.size) { k -> (src[k] + negMean[k]) * invStd[k] }
        }

    // ------------------------------------------------------------- 推理 + CTC
    /** 识别结果：文本 + 模型自判语种（来自输出首部 `<|zh|>` 之类的标签）+ 帧数。 */
    data class Result(val text: String, val lang: String, val frames: Int)

    /** 对一段 16k 单声道 PCM16 做识别；language: 3=中文 4=英文 11=日文 12=韩文 0=自动。 */
    fun recognize(samples: ShortArray, sampleRate: Int = 16000, language: Int = 3, textNorm: Int = 14): String =
        recognizeEx(samples, sampleRate, language, textNorm).text

    fun recognizeEx(samples: ShortArray, sampleRate: Int = 16000, language: Int = 3, textNorm: Int = 14): Result {
        if (samples.size < sampleRate / 2) return Result("", "", 0)      // 少于 0.5 秒不识别
        val fbank = AudioDsp.fbank(samples, sampleRate)
        if (fbank.isEmpty()) return Result("", "", 0)

        val lfr = applyLfr(fbank)
        val feats = applyCmvn(lfr)
        val frames = feats.size
        val dim = feats[0].size

        val flat = FloatArray(frames * dim)
        for (i in 0 until frames) feats[i].copyInto(flat, i * dim)

        var out = Result("", "", frames)
        OnnxTensor.createTensor(env, FloatBuffer.wrap(flat), longArrayOf(1, frames.toLong(), dim.toLong())).use { x ->
            OnnxTensor.createTensor(env, IntBuffer.wrap(intArrayOf(frames)), longArrayOf(1)).use { xLen ->
                OnnxTensor.createTensor(env, IntBuffer.wrap(intArrayOf(language)), longArrayOf(1)).use { lang ->
                    OnnxTensor.createTensor(env, IntBuffer.wrap(intArrayOf(textNorm)), longArrayOf(1)).use { norm ->
                        val feeds = mapOf(
                            "x" to x,
                            "x_length" to xLen,
                            "language" to lang,
                            "text_norm" to norm,
                        )
                        session.run(feeds).use { result ->
                            @Suppress("UNCHECKED_CAST")
                            val logits = (result.get(0) as OnnxTensor).value as Array<Array<FloatArray>>
                            out = decode(logits[0])
                        }
                    }
                }
            }
        }
        return out
    }

    /**
     * CTC 贪心解码：逐帧 argmax → 去重 → 去 blank(0) → 拼 token。
     *
     * 顺带把输出开头的 `<|zh|>` / `<|en|>` / `<|ja|>` / `<|ko|>` 标签读出来——
     * 这是模型自己判定的语种，噪声段经常和请求语言不一致，用来做幻觉的旁证。
     */
    private fun decode(frames: Array<FloatArray>): Result {
        val sb = StringBuilder()
        var lang = ""
        var prev = -1
        for (frame in frames) {
            var best = 0
            var bestScore = frame[0]
            for (i in 1 until frame.size) {
                if (frame[i] > bestScore) {
                    bestScore = frame[i]
                    best = i
                }
            }
            if (best != prev && best != 0) {
                val tok = tokens[best]
                if (!tok.isNullOrEmpty()) {
                    if (tok.startsWith("<|") && tok.endsWith("|>")) {
                        if (lang.isEmpty()) {
                            lang = when (tok) {
                                "<|zh|>" -> "zh"
                                "<|en|>" -> "en"
                                "<|ja|>" -> "ja"
                                "<|ko|>" -> "ko"
                                "<|yue|>" -> "yue"
                                else -> ""
                            }
                        }
                    } else if (tok.startsWith("\u2581")) {
                        sb.append(' ').append(tok.substring(1))
                    } else {
                        sb.append(tok)
                    }
                }
            }
            prev = best
        }
        return Result(sb.toString().replace(Regex("<\\|[^|]*\\|>"), "").trim(), lang, frames.size)
    }

    override fun close() {
        session.close()
    }
}
