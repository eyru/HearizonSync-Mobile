package com.hearizon.sync.audio

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.media.audiofx.AutomaticGainControl
import android.media.audiofx.NoiseSuppressor
import kotlin.concurrent.thread

/**
 * 麦克风直采：16 kHz / 单声道 / PCM16，按固定帧长回调。
 *
 * 为什么不用 MediaRecorder：那玩意会先编码成 AAC，识别端还得解回来，白折损一层，
 * 而且拿不到稳定的 20 ms 帧边界。
 *
 * ## 音源选择（真机实测教训）
 * 最初用 `VOICE_RECOGNITION`，在 vivo（Android 16）上输入电平被压得极低，必须对着手机喊才触发 VAD。
 * 那个源面向「通话近讲」场景，国产 ROM 会叠加强降噪和距离压制。
 * 改成 `MIC`（原始麦克风）后灵敏度正常；再叠一层软件 AGC，让小声说话也能被 VAD 抓住。
 *
 * ## 关于软件 AGC 的边界
 * 软件增益**不改变信噪比**，它只是把整段电平抬到 VAD 与模型更舒服的区间，
 * 所以它不能替代降噪——真正的噪声抑制在 `Denoiser`（默认关闭）与文本级过滤里。
 */
class AudioCapture(
    private val frameMillis: Int = 20,
    private val onFrame: (ShortArray) -> Unit,
    private val onLevel: (Float) -> Unit = {},
    private val autoGain: Boolean = true,
) {

    companion object {
        const val SAMPLE_RATE = 16_000
        private const val CHANNEL = AudioFormat.CHANNEL_IN_MONO
        private const val ENCODING = AudioFormat.ENCODING_PCM_16BIT

        /** 软件 AGC 的目标峰值（约 -7 dBFS），再高容易在近讲时削波 */
        private const val TARGET_PEAK = 0.45f

        /** 增益上限：把极小的输入（如 0.01 峰值）抬到可用范围 */
        private const val MAX_GAIN = 8f

        /** 峰值包络的每帧衰减，20 ms 一帧 ≈ 300 ms 半衰，避免句间把底噪越抬越高 */
        private const val PEAK_DECAY = 0.94f
    }

    val frameSamples: Int = SAMPLE_RATE * frameMillis / 1000

    @Volatile
    private var running = false
    private var record: AudioRecord? = null
    private var worker: Thread? = null
    private var hwAgc: AutomaticGainControl? = null
    private var hwNs: NoiseSuppressor? = null

    @SuppressLint("MissingPermission")
    fun start(): Boolean {
        if (running) return true

        val minBuffer = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL, ENCODING)
        if (minBuffer <= 0) return false

        // 至少留 8 帧，避免 read 因缓冲太小被切碎
        val bufferBytes = maxOf(minBuffer, frameSamples * 2 * 8)

        val recorder = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            CHANNEL,
            ENCODING,
            bufferBytes,
        )
        if (recorder.state != AudioRecord.STATE_INITIALIZED) {
            recorder.release()
            return false
        }

        record = recorder

        // 硬件 AGC / 降噪：各家 ROM 支持不一，拿不到就静默跳过，不因此拒绝启动
        runCatching {
            hwAgc = AutomaticGainControl.create(recorder.audioSessionId)?.apply { setEnabled(true) }
        }
        runCatching {
            hwNs = NoiseSuppressor.create(recorder.audioSessionId)?.apply { setEnabled(true) }
        }

        running = true
        recorder.startRecording()

        worker = thread(name = "audio-capture", isDaemon = true) {
            val frame = ShortArray(frameSamples)
            var filled = 0
            var peakEnv = 0.02f
            var gain = 1f

            while (running) {
                val read = recorder.read(frame, filled, frame.size - filled)
                if (read <= 0) continue
                filled += read
                if (filled < frame.size) continue

                val chunk = frame.copyOf()
                filled = 0

                var peak = 0
                for (sample in chunk) {
                    val abs = if (sample < 0) -sample.toInt() else sample.toInt()
                    if (abs > peak) peak = abs
                }
                val peakNorm = peak / 32768f

                if (autoGain) {
                    // 峰值包络：快升慢降，避免一句话里增益来回跳
                    peakEnv = maxOf(peakNorm, peakEnv * PEAK_DECAY)
                    val want = (TARGET_PEAK / maxOf(peakEnv, 0.008f)).coerceIn(1f, MAX_GAIN)
                    gain = gain * 0.85f + want * 0.15f
                    if (gain > 1.02f) {
                        for (i in chunk.indices) {
                            val v = (chunk[i] * gain).toInt()
                            chunk[i] = (if (v > 32767) 32767 else if (v < -32768) -32768 else v).toShort()
                        }
                    }
                }

                // 电平条用增益**之后**的值，才能反映「现在能不能被听见」
                onLevel((peakNorm * (if (autoGain) gain else 1f)).coerceIn(0f, 1f))
                onFrame(chunk)
            }
        }
        return true
    }

    fun stop() {
        running = false
        worker?.let { runCatching { it.join(600) } }
        worker = null
        hwAgc?.let { runCatching { it.enabled = false; it.release() } }
        hwAgc = null
        hwNs?.let { runCatching { it.enabled = false; it.release() } }
        hwNs = null
        record?.let {
            runCatching { it.stop() }
            runCatching { it.release() }
        }
        record = null
    }
}
