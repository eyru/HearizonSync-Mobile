package com.hearizon.sync.audio

import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.sqrt

/**
 * 能量 VAD：滑动窗口分位噪声底 + 起止滞后判决。
 *
 * ## 为什么噪声底必须用「窗口分位」而不是指数平滑
 *
 * 同一段 10.3 秒连续语音，只换噪声底估计方式，识别结果天差地别
 * （实测脚本 `工具/参考实现/eval_vad_noise.py`、`eval_final.py` 场景 A）：
 *
 * | 噪声底估计 | 切段 | 覆盖时长 | CER | 结果 |
 * |---|---|---|---|---|
 * | 指数平滑 mult=2.2 | 9 段 | 4.54s | 0.486 | 大家。/好。/我们。/现在测。/以下是。… |
 * | 指数平滑 mult=2.6 | 8 段 | 3.84s | 0.514 | 我们。/现在测。/以下是。/看看。… |
 * | 指数平滑 mult=3.0 | 6 段 | 2.88s | 0.657 | 现在测。/以下是。/看看。… |
 * | **窗口 3s 取 10 分位** | **4 段** | **9.12s** | **0.029** | 大家好。/今天天气不错。/我们现在测试一下实时语音识别系统。/看看段句和标点准不准。 |
 *
 * 根因：指数平滑在**持续说话**时会跟着语音电平一起爬升（因为每帧都比上一帧高一点点，
 * 平滑值就被一路拖上去），等句中稍弱的音节（「识」「别」「标」）出现时，
 * 噪声底已经爬到它们之上，于是被判成静音 → 句子被切碎、字被吞。
 * 分位数天然抗这个：只要窗口里还有 10% 的帧是静音，噪声底就不会被语音带跑。
 *
 * ## 其它参数依据
 * - `noiseMult=2.6`：阈值为噪声底的 2.6 倍（约 +8.3 dB）。实测 2.2 / 2.6 / 3.0 在分位法下结果相同，
 *   取中间值给真机（有 AGC、回声消除）留余量。
 * - `enterFrames=2`：连续 2 帧（40 ms）超阈值才算开口，挡单帧爆音；配合 200 ms 预卷不丢首字。
 * - `exitFrames=12`：连续 12 帧（240 ms）静音收尾，与 [com.hearizon.sync.asr.LiveSession] 的尾巴对齐。
 */
class VadEnergy(
    /** 高于噪声底多少倍才算语音 */
    private val noiseMult: Float = 2.6f,
    /** 绝对地板：低于此 RMS 一律不算语音，防止安静环境把底噪当人声。
     *  初版取 0.003（约 −50 dBFS），真机上非要大声说话才触发；配合 [AudioCapture] 的
     *  软件 AGC（目标峰值 0.45）后降到 0.0012（约 −58 dBFS），小声也能开口。 */
    private val absFloor: Float = 0.0012f,
    /** 噪声底统计窗口（帧）。150 帧 = 3 秒 */
    private val windowFrames: Int = 150,
    /** 噪声底分位数 */
    private val percentile: Float = 10f,
    /** 连续多少帧超阈值才认定开口 */
    private val enterFrames: Int = 2,
    /** 连续多少帧低于阈值才认定收尾 */
    private val exitFrames: Int = 12,
) {

    enum class State { SILENCE, SPEECH }

    var state: State = State.SILENCE
        private set

    /** 当前估计的噪声底（dBFS），给 UI 自检用 */
    var noiseFloorDb: Float = -60f
        private set

    /** 当前判决阈值（线性 RMS），给 UI 画电平条参考线用 */
    var thresholdRms: Float = absFloor
        private set

    private val window = FloatArray(windowFrames)
    private var count = 0
    private val scratch = FloatArray(windowFrames)

    private var voicedRun = 0
    private var silentRun = 0

    /**
     * @return true 表示这一帧属于语音段，应送进识别管线
     */
    fun process(frame: ShortArray): Boolean {
        val rms = frameRms(frame)
        pushWindow(rms)

        val floorRms = if (count < 20) {
            absFloor
        } else {
            max(percentileOf(percentile), absFloor)
        }
        val threshold = max(floorRms * noiseMult, absFloor)
        thresholdRms = threshold
        noiseFloorDb = if (floorRms < 1e-6f) {
            -90f
        } else {
            (20.0 * log10(floorRms.toDouble())).toFloat().coerceIn(-90f, 0f)
        }

        val voiced = rms > threshold
        if (voiced) {
            voicedRun++
            silentRun = 0
        } else {
            silentRun++
            voicedRun = 0
        }

        when (state) {
            State.SILENCE -> if (voicedRun >= enterFrames) state = State.SPEECH
            State.SPEECH -> if (silentRun >= exitFrames) state = State.SILENCE
        }

        return state == State.SPEECH
    }

    fun reset() {
        state = State.SILENCE
        voicedRun = 0
        silentRun = 0
        count = 0
        noiseFloorDb = -60f
        thresholdRms = absFloor
    }

    // ------------------------------------------------------------ 内部

    private fun pushWindow(v: Float) {
        if (count < windowFrames) {
            window[count++] = v
        } else {
            System.arraycopy(window, 1, window, 0, windowFrames - 1)
            window[windowFrames - 1] = v
        }
    }

    /** 线性插值分位（与 numpy.percentile 默认行为一致）。 */
    private fun percentileOf(p: Float): Float {
        if (count == 0) return 0f
        System.arraycopy(window, 0, scratch, 0, count)
        java.util.Arrays.sort(scratch, 0, count)
        val rank = (p / 100f) * (count - 1)
        val lo = floor(rank).toInt()
        val hi = ceil(rank).toInt()
        return if (lo == hi) scratch[lo] else scratch[lo] + (scratch[hi] - scratch[lo]) * (rank - lo)
    }

    companion object {
        /** 帧 RMS（归一化到 0..1，与 SenseVoice 前端一致）。 */
        fun frameRms(frame: ShortArray): Float {
            if (frame.isEmpty()) return 0f
            var sum = 0.0
            for (sample in frame) {
                val v = sample / 32768.0
                sum += v * v
            }
            return sqrt(sum / frame.size).toFloat()
        }

        /** 帧能量（dBFS），-90 为地板。 */
        fun frameDbfs(frame: ShortArray): Float {
            val rms = frameRms(frame)
            if (rms < 1e-6f) return -90f
            return (20.0 * log10(rms.toDouble())).toFloat().coerceIn(-90f, 0f)
        }
    }
}
