package com.hearizon.sync.mt

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * 自研 MarianMT 译文生成器（encoder-decoder + 贪心解码）。
 *
 * 同一套代码吃多个方向的模型：
 * - `中→英`：`opus-mt-zh-en`，模型在 files/models/mt/，词表在 assets/mt/
 * - `英→中`：`opus-mt-en-zh`，模型在 files/models/mt_en_zh/，词表在 assets/mt_en_zh/
 * - `日→英`：`opus-mt-ja-en`，模型在 files/models/mt_ja_en/，词表在 assets/mt_ja_en/
 * - `韩→英`：`opus-mt-ko-en`，模型在 files/models/mt_ko_en/，词表在 assets/mt_ko_en/
 *
 * @param tokenizerDir 词表所在的 assets 目录（`mt` / `mt_en_zh` / `mt_ja_en` / `mt_ko_en`）
 * @param padToken 解码起始 token，各模型不同：绝大多数是 65000，
 *   **日→英是 60715**（该模型 vocab_size=60716）。传错会解码出乱码或空串。
 */
class MarianTranslator(
    context: Context,
    modelDir: File,
    tokenizerDir: String = "mt",
    private val padToken: Int = SpmTokenizer.PAD,
) : AutoCloseable {

    private val env = OrtEnvironment.getEnvironment()
    private val enc: OrtSession
    private val dec: OrtSession
    private val tok = SpmTokenizer(context, "$tokenizerDir/src_spm.txt")
    private val id2piece: Array<String>

    init {
        val list = ArrayList<String>(65536)
        context.assets.open("$tokenizerDir/vocab.txt").bufferedReader(Charsets.UTF_8).use { r ->
            while (true) {
                list.add(r.readLine() ?: break)
            }
        }
        if (list.isNotEmpty() && list.last().isEmpty()) list.removeAt(list.size - 1)
        id2piece = list.toTypedArray()

        val so = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        val encFile = File(modelDir, "encoder.onnx")
        val decFile = File(modelDir, "decoder.onnx")
        enc = try {
            env.createSession(encFile.absolutePath, so)
        } catch (e: Exception) {
            throw IllegalStateException(
                "encoder 加载失败 exists=${encFile.exists()} len=${encFile.length()} " +
                    "path=${encFile.absolutePath} err=${e.message}", e
            )
        }
        dec = try {
            env.createSession(decFile.absolutePath, so)
        } catch (e: Exception) {
            throw IllegalStateException(
                "decoder 加载失败 exists=${decFile.exists()} len=${decFile.length()} err=${e.message}", e
            )
        }
    }

    /** 中文 → 英文。空输入返回空串。 */
    fun translate(zh: String, maxNew: Int = 96): String {
        val text = zh.trim()
        if (text.isEmpty()) return ""
        val ids = tok.encode(text)
        val n = ids.size
        val mask = LongArray(n) { 1L }

        val hidden: FloatArray
        val hidDim: Int
        OnnxTensor.createTensor(env, LongBuffer.wrap(ids), longArrayOf(1, n.toLong())).use { inputIds ->
            OnnxTensor.createTensor(env, LongBuffer.wrap(mask), longArrayOf(1, n.toLong())).use { attn ->
                enc.run(mapOf("input_ids" to inputIds, "attention_mask" to attn)).use { res ->
                    val t = res[0] as OnnxTensor
                    hidDim = t.info.shape[2].toInt()
                    hidden = readFloats(t)
                }
            }
        }

        val out = ArrayList<Int>(maxNew)
        val cur = ArrayList<Long>(maxNew + 1).apply { add(padToken.toLong()) }
        var repeat = 1
        var last = -1
        OnnxTensor.createTensor(env, LongBuffer.wrap(mask), longArrayOf(1, n.toLong())).use { attn ->
            OnnxTensor.createTensor(env, FloatBuffer.wrap(hidden), longArrayOf(1, n.toLong(), hidDim.toLong())).use { hid ->
                for (step in 0 until maxNew) {
                    val curArr = LongArray(cur.size) { cur[it] }
                    var next: Int
                    OnnxTensor.createTensor(env, LongBuffer.wrap(curArr), longArrayOf(1, cur.size.toLong())).use { inputIds ->
                        dec.run(
                            mapOf(
                                "input_ids" to inputIds,
                                "encoder_hidden_states" to hid,
                                "encoder_attention_mask" to attn,
                            )
                        ).use { res ->
                            val logits = res[0] as OnnxTensor
                            next = argmaxLastPosition(logits)
                        }
                    }
                    if (next == SpmTokenizer.EOS) break
                    if (next == last) {
                        repeat++
                        if (repeat >= 3) break      // 贪心卡死保护（连续吐同一个 token 就停）
                    } else {
                        repeat = 1
                        last = next
                    }
                    out.add(next)
                    cur.add(next.toLong())
                }
            }
        }
        return render(out)
    }

    private fun argmaxLastPosition(logits: OnnxTensor): Int {
        val shape = logits.info.shape
        val len = shape[1].toInt()
        val vocab = shape[2].toInt()
        val buf = logits.floatBuffer.duplicate()
        val base = (len - 1) * vocab
        var best = 0
        var bestV = Float.NEGATIVE_INFINITY
        for (i in 0 until vocab) {
            val v = buf.get(base + i)
            if (v > bestV) {
                bestV = v
                best = i
            }
        }
        return best
    }

    private fun readFloats(t: OnnxTensor): FloatArray {
        val buf = t.floatBuffer.duplicate()
        val arr = FloatArray(buf.remaining())
        buf.get(arr)
        return arr
    }

    private fun render(ids: List<Int>): String {
        val sb = StringBuilder()
        for (id in ids) {
            // UNK 直接丢弃：模型偶尔连吐一串 <unk>，渲染出来是「<unk><unk>…」这样的噪声
            if (id == SpmTokenizer.EOS || id == padToken || id == SpmTokenizer.UNK) continue
            if (id in id2piece.indices) sb.append(id2piece[id])
        }
        return sb.toString().replace('\u2581', ' ').trim()
    }

    override fun close() {
        runCatching { enc.close() }
        runCatching { dec.close() }
        runCatching { tok.close() }
    }
}
