package com.hearizon.sync.audio

import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * 轻量降噪 + 远场增益归一。
 *
 * 算法移植自桌面版 v2.5.6 的 `denoise()` / `normalize_gain()`：
 * 按 20 ms 分帧算 RMS，取第 10 百分位当噪声底，低于「噪声底 × 1.1」的帧衰减到 0.30，语音帧原样保留。
 * 好处是不做频谱变换，不会引入音乐噪声（musical noise），也不会破坏语音的相位结构。
 *
 * ## 实测结论：对 SenseVoice 识别率没有稳定收益，因此**默认关闭**
 *
 * 桌面版的「底噪 −55.8%、语音保留 100%」是**能量指标**，不是识别率指标。
 * 在 SenseVoice 上重跑（`工具/参考实现/eval_final2.py` 场景 C，同一段 10.3 s 语音 + 白噪）：
 *
 * | SNR | 关闭降噪 CER | 开启降噪 CER |
 * |---|---|---|
 * | −5 dB | 0.914 | **0.886** |
 * | 0 dB | **0.657** | 0.714 |
 * | 5 dB | 0.429 | **0.371** |
 * | 10 dB | **0.086** | 0.143 |
 * | 15 dB | 0.000 | 0.000 |
 *
 * 平均下来关闭略好（0.417 vs 0.423），且噪声场景（`eval_final2.py` 场景 B）里
 * 开启降噪会把键盘声从 0 条幻觉**放大成 3 条「嗯。」**——因为增益归一抬高了底噪。
 *
 * 所以：默认 `enhance = false`，只在用户明确打开「增强」时才走这条路。
 *
 * **为什么不在采集端做**：VAD 要靠原始电平判断「有没有人说话」，
 * 先降噪会把远场小声一起压掉，反而切不出句。所以只在「句尾、送识别前」处理。
 */
object Denoiser {

    /** 与 SenseVoice 前端一致的分帧：20 ms @ 16 kHz。 */
    private const val FRAME = 320

    /** 低于噪声底的帧衰减倍数（桌面版实测值）。 */
    private const val QUIET_GAIN = 0.30f

    /** 噪声底分位数。 */
    private const val NOISE_PERCENTILE = 10.0

    /** 远场增益归一的目标电平（相对满量程）。 */
    private const val TARGET_RMS = 0.12f

    /** 增益上限，防止把底噪一起放大。 */
    private const val MAX_GAIN = 6.0f

    /** 峰值保护，避免削波失真。 */
    private const val PEAK_LIMIT = 32767.0 * 0.97

    /**
     * 帧级噪声门：把明显低于噪声底的帧衰减，语音帧不动。
     *
     * @param strength 越大门限越低（压得更狠）；0.3 以下等同于关闭，桌面版默认 1.0
     */
    fun denoise(samples: ShortArray, strength: Float = 1.0f): ShortArray {
        if (samples.size < 512) return samples
        val n = samples.size / FRAME
        if (n < 2) return samples

        val energy = DoubleArray(n)
        for (i in 0 until n) {
            var sum = 0.0
            val off = i * FRAME
            for (j in 0 until FRAME) {
                val v = samples[off + j].toDouble()
                sum += v * v
            }
            energy[i] = sqrt(sum / FRAME)
        }

        val noise = percentile(energy, NOISE_PERCENTILE)
        if (noise <= 1e-6) return samples

        val gate = noise * 1.1 / max(0.3f, strength)
        var anyQuiet = false
        for (i in 0 until n) {
            if (energy[i] < gate) {
                anyQuiet = true
                break
            }
        }
        if (!anyQuiet) return samples

        val out = samples.copyOf()
        for (i in 0 until n) {
            if (energy[i] >= gate) continue
            val off = i * FRAME
            for (j in 0 until FRAME) {
                out[off + j] = (samples[off + j] * QUIET_GAIN).toInt().toShort()
            }
        }
        return out
    }

    /**
     * 远场增益归一：说话人离得远时音量小、识别率会掉，把整句拉到统一目标电平。
     * 只放大不缩小；已经够响的句子原样返回。带峰值保护。
     */
    fun normalizeGain(
        samples: ShortArray,
        target: Float = TARGET_RMS,
        maxGain: Float = MAX_GAIN,
    ): ShortArray {
        if (samples.isEmpty()) return samples
        var sum = 0.0
        for (s in samples) {
            val v = s / 32768.0
            sum += v * v
        }
        val rms = sqrt(sum / samples.size)
        if (rms <= 1e-5) return samples

        var gain = target / rms
        if (gain <= 1.0) return samples
        gain = min(gain, maxGain.toDouble())

        var peak = 0.0
        for (s in samples) {
            val v = abs(s * gain)
            if (v > peak) peak = v
        }
        val scale = if (peak > PEAK_LIMIT) PEAK_LIMIT / peak else 1.0
        val k = gain * scale

        return ShortArray(samples.size) {
            (samples[it] * k).toInt().coerceIn(-32768, 32767).toShort()
        }
    }

    /** 降噪 + 增益归一，一步到位（与桌面版句尾处理顺序一致）。 */
    fun enhance(samples: ShortArray, strength: Float = 1.0f): ShortArray =
        normalizeGain(denoise(samples, strength))

    /** 线性插值分位数（与 numpy.percentile 默认行为一致）。 */
    private fun percentile(values: DoubleArray, p: Double): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.copyOf()
        sorted.sort()
        val rank = (p / 100.0) * (sorted.size - 1)
        val lo = floor(rank).toInt()
        val hi = ceil(rank).toInt()
        return if (lo == hi) sorted[lo] else sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - lo)
    }

    /** 帧级 RMS（dBFS），给「降噪对比」界面显示噪声底用。 */
    fun frameDbfs(samples: ShortArray): FloatArray {
        val n = samples.size / FRAME
        return FloatArray(n) { i ->
            var sum = 0.0
            val off = i * FRAME
            for (j in 0 until FRAME) {
                val v = samples[off + j].toDouble()
                sum += v * v
            }
            val rms = sqrt(sum / FRAME)
            if (rms < 1e-6) -90f else (20.0 * kotlin.math.log10(rms / 32768.0)).toFloat().coerceIn(-90f, 0f)
        }
    }
}
