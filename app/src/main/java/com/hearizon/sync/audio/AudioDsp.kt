package com.hearizon.sync.audio

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln

/**
 * 声学前端：kaldi 风格 fbank(80)。
 *
 * 与 `工具/参考实现/ref_asr.py` 的 fbank() 逐参数对齐：
 * - 25ms 帧长 / 10ms 帧移 / snip_edges=true
 * - povey 窗（(0.5-0.5cos)^0.85）
 * - 预加重 0.97、逐帧去直流
 * - 20Hz ~ (Nyquist-400Hz) 的 80 个三角 mel 滤波器
 * - 自然对数，下限 1.1920929e-07
 *
 * 这条链路是自研的（不依赖 sherpa-onnx），所有数学都用 Double 计算，
 * 以便与 Python 参考实现逐帧比对。
 */
object AudioDsp {

    const val N_MELS = 80
    private const val FRAME_LEN_MS = 25.0
    private const val FRAME_SHIFT_MS = 10.0
    private const val PREEMPH = 0.97
    private const val LOW_FREQ = 20.0
    /** 相对 Nyquist 的上限偏移，与 kaldi 的 high_freq=-400 等价。 */
    private const val HIGH_FREQ_OFFSET = -400.0
    private const val EPS = 1.1920929e-07

    // ------------------------------------------------------------------ FFT
    /** 就地 radix-2 复数 FFT，长度须为 2 的幂。 */
    fun fft(re: DoubleArray, im: DoubleArray) {
        val n = re.size
        require(n and (n - 1) == 0) { "FFT 长度必须是 2 的幂：$n" }

        // 位反转置换
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j or bit
            if (i < j) {
                val tr = re[i]; re[i] = re[j]; re[j] = tr
                val ti = im[i]; im[i] = im[j]; im[j] = ti
            }
        }

        var len = 2
        while (len <= n) {
            val ang = -2.0 * PI / len
            val wRe = cos(ang)
            val wIm = kotlin.math.sin(ang)
            val half = len shr 1
            var i = 0
            while (i < n) {
                var curRe = 1.0
                var curIm = 0.0
                for (k in 0 until half) {
                    val a = i + k
                    val b = a + half
                    val vRe = re[b] * curRe - im[b] * curIm
                    val vIm = re[b] * curIm + im[b] * curRe
                    val uRe = re[a]
                    val uIm = im[a]
                    re[a] = uRe + vRe
                    im[a] = uIm + vIm
                    re[b] = uRe - vRe
                    im[b] = uIm - vIm
                    val nextRe = curRe * wRe - curIm * wIm
                    curIm = curRe * wIm + curIm * wRe
                    curRe = nextRe
                }
                i += len
            }
            len = len shl 1
        }
    }

    // ------------------------------------------------------------------ mel
    private fun melScale(freq: Double): Double = 1127.0 * ln(1.0 + freq / 700.0)

    private fun inverseMelScale(mel: Double): Double = 700.0 * (exp(mel / 1127.0) - 1.0)

    /** 三角 mel 滤波器组，形状 (N_MELS, nFft/2+1)。 */
    fun melFilterBank(sampleRate: Int, nFft: Int): Array<DoubleArray> {
        val nBins = nFft / 2 + 1
        val binWidth = sampleRate.toDouble() / nFft
        val highFreq = sampleRate / 2.0 + HIGH_FREQ_OFFSET
        val melLow = melScale(LOW_FREQ)
        val melHigh = melScale(highFreq)

        val hz = DoubleArray(N_MELS + 2) { idx ->
            inverseMelScale(melLow + (melHigh - melLow) * idx / (N_MELS + 1))
        }

        return Array(N_MELS) { m ->
            val left = hz[m]
            val center = hz[m + 1]
            val right = hz[m + 2]
            val upSpan = (center - left).coerceAtLeast(1e-9)
            val downSpan = (right - center).coerceAtLeast(1e-9)
            DoubleArray(nBins) { k ->
                val freq = k * binWidth
                val up = (freq - left) / upSpan
                val down = (right - freq) / downSpan
                kotlin.math.min(up, down).coerceIn(0.0, 1.0)
            }
        }
    }

    /** povey 窗（kaldi 默认）。 */
    private fun poveyWindow(length: Int): DoubleArray = DoubleArray(length) { i ->
        val x = 2.0 * PI * i / (length - 1)
        Math.pow(0.5 - 0.5 * cos(x), 0.85)
    }

    // ---------------------------------------------------------------- fbank
    /**
     * @param samples 16kHz 单声道 PCM16
     * @return nFrames × 80 的 log-mel 特征
     */
    fun fbank(samples: ShortArray, sampleRate: Int): Array<DoubleArray> {
        val frameLength = Math.round(sampleRate * FRAME_LEN_MS / 1000.0).toInt()   // 400
        val frameShift = Math.round(sampleRate * FRAME_SHIFT_MS / 1000.0).toInt()  // 160
        var nFft = 1
        while (nFft < frameLength) nFft = nFft shl 1                              // 512

        if (samples.size < frameLength) return emptyArray()
        val nFrames = 1 + (samples.size - frameLength) / frameShift
        val window = poveyWindow(frameLength)
        val mel = melFilterBank(sampleRate, nFft)

        // 预加重：整段信号做一次（与 kaldi 一致，跨帧连续）
        val x = DoubleArray(samples.size) { samples[it] / 32768.0 }
        for (i in x.size - 1 downTo 1) {
            x[i] -= PREEMPH * x[i - 1]
        }

        val re = DoubleArray(nFft)
        val im = DoubleArray(nFft)
        val out = Array(nFrames) { DoubleArray(N_MELS) }

        for (f in 0 until nFrames) {
            val start = f * frameShift
            var dc = 0.0
            for (i in 0 until frameLength) dc += x[start + i]
            dc /= frameLength

            java.util.Arrays.fill(re, 0.0)
            java.util.Arrays.fill(im, 0.0)
            for (i in 0 until frameLength) {
                re[i] = (x[start + i] - dc) * window[i]
            }
            fft(re, im)

            val power = DoubleArray(nFft / 2 + 1) { k -> re[k] * re[k] + im[k] * im[k] }
            for (m in 0 until N_MELS) {
                val weights = mel[m]
                var acc = 0.0
                for (k in weights.indices) acc += power[k] * weights[k]
                out[f][m] = ln(acc.coerceAtLeast(EPS))
            }
        }
        return out
    }
}
