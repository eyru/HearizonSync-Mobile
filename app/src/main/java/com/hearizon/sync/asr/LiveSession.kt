package com.hearizon.sync.asr

import com.hearizon.sync.audio.AudioCapture
import com.hearizon.sync.audio.Denoiser
import com.hearizon.sync.audio.VadEnergy
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * 一次「边听边出字幕」的会话：麦克风 → VAD 切句 → 单线程识别 → 回调。
 *
 * 设计取舍：
 * - 采集与识别分线程，识别慢的时候由单线程队列自然排队，不会丢音频；
 * - 段落两端各留一点静音（前 200 ms 预卷 + 后 240 ms 尾巴），避免吞掉开头的字；
 * - 超过 15 秒强制切段，防止长句同时占内存和让字幕迟迟不出；
 * - 引擎是同一个 OrtSession，串行调用即可（ORT 的 run 是线程安全的，但串行更稳）；
 * - 识别结果先过 [TranscriptFilter]，把噪声触发的「嗯。」「我。」挡在字幕外面。
 */
class LiveSession(
    private val engine: SenseVoiceEngine,
    private val onSegment: (Segment) -> Unit,
    private val onLevel: (Float) -> Unit,
    private val onSpeaking: (Boolean) -> Unit,
    private val onError: (String) -> Unit = {},
    /** false = 不启动麦克风，由外部 feed() 喂帧（回放/自动化测试用） */
    private val microphone: Boolean = true,
    /** 句尾降噪 + 增益归一。实测对识别率无明显收益（见 [Denoiser] 注释），默认关闭 */
    private val enhance: Boolean = false,
    /**
     * 识别语言（SenseVoice language）：0=自动、3=中文、4=英文、7=粤语、11=日文、12=韩文。
     * 默认自动——固定成中文会把英语硬解成中文（或原样输出英文），中英混说也不自然。
     */
    private val asrLanguage: Int = 0,
) {

    /** 一段识别结果。时间相对会话开始，供 SRT 导出使用。 */
    data class Segment(
        val text: String,
        val startMs: Long,
        val durationMs: Long,
    )

    companion object {
        /** 20 ms 帧 */
        private const val FRAME_MS = 20
        private const val PREROLL_FRAMES = 10      // 200 ms 预卷
        private const val TAIL_FRAMES = 12         // 段尾静音保留 240 ms（与 VAD 的 exitFrames 对齐）
        private const val MIN_SEGMENT_MS = 320     // 短于此不识别（咳一声、碰麦）
        private const val MAX_SEGMENT_MS = 15_000
    }

    private val vad = VadEnergy()
    private val worker = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "asr-worker").apply { isDaemon = true }
    }
    private val preroll = ArrayDeque<ShortArray>()
    private val current = ArrayList<ShortArray>()
    private val running = AtomicBoolean(false)

    /** 已处理过的总帧数，用来算段落起点 */
    private var frameIndex = 0L
    /** 当前段第一帧的全局帧号 */
    private var segmentStartFrame = 0L
    private var lastSpeaking = false
    private var silentFrames = 0

    private val capture = AudioCapture(
        frameMillis = FRAME_MS,
        onFrame = { frame -> handleFrame(frame) },
        onLevel = onLevel,
    )

    fun start(): Boolean {
        if (!running.compareAndSet(false, true)) return true
        vad.reset()
        current.clear()
        preroll.clear()
        lastSpeaking = false
        silentFrames = 0
        frameIndex = 0
        segmentStartFrame = 0
        val ok = if (microphone) capture.start() else true
        if (!ok) running.set(false)
        return ok
    }

    fun stop() {
        if (!running.compareAndSet(true, false)) return
        if (microphone) capture.stop()   // 内部会 join 采集线程，之后 current / preroll 只由本线程访问
        flush()
        onSpeaking(false)
    }

    /** 回放/自测：直接把一帧 20ms 音频喂进同一套 VAD + 切句逻辑。 */
    fun feed(frame: ShortArray) = handleFrame(frame)

    /** 等识别队列排空，保证回放结束时最后几段都出完。 */
    fun awaitIdle(timeoutMs: Long = 30_000): Boolean {
        val latch = java.util.concurrent.CountDownLatch(1)
        worker.execute { latch.countDown() }
        return latch.await(timeoutMs, java.util.concurrent.TimeUnit.MILLISECONDS)
    }

    // ------------------------------------------------------------ 帧处理
    private fun handleFrame(frame: ShortArray) {
        if (!running.get()) return
        val speaking = vad.process(frame)

        if (current.isEmpty()) {
            if (speaking) {
                while (preroll.isNotEmpty()) current.add(preroll.removeFirst())
                current.add(frame)
                segmentStartFrame = frameIndex - (current.size - 1)
                silentFrames = 0
            } else {
                preroll.addLast(frame)
                while (preroll.size > PREROLL_FRAMES) preroll.removeFirst()
            }
        } else {
            // 段内一律收帧：静音帧也留着，段尾才能带上 240ms 尾巴，
            // 否则边界上的字（「语音|识别」的「识」）会被当成静音丢掉。
            current.add(frame)
            if (speaking) silentFrames = 0 else silentFrames++
            if (silentFrames >= TAIL_FRAMES || current.size * FRAME_MS >= MAX_SEGMENT_MS) {
                flush()
                silentFrames = 0
            }
        }

        frameIndex++

        if (speaking != lastSpeaking) {
            lastSpeaking = speaking
            onSpeaking(speaking)
        }
    }

    private fun flush() {
        if (current.isEmpty()) return
        val total = current.sumOf { it.size }
        val durationMs = total * 1000L / AudioCapture.SAMPLE_RATE
        val startMs = segmentStartFrame * FRAME_MS
        if (durationMs < MIN_SEGMENT_MS) {
            current.clear()
            return
        }
        var pcm = ShortArray(total)
        var offset = 0
        for (chunk in current) {
            chunk.copyInto(pcm, offset)
            offset += chunk.size
        }
        current.clear()
        preroll.clear()

        if (enhance) pcm = Denoiser.enhance(pcm)

        val audio = pcm
        worker.execute {
            runCatching { engine.recognizeEx(audio, language = asrLanguage) }
                .onSuccess { r ->
                    if (r.text.isNotBlank() &&
                        !TranscriptFilter.isHallucination(
                            r.text,
                            durationMs.toInt(),
                            expectLang = expectLang,
                            detectedLang = r.lang,
                        )
                    ) {
                        onSegment(Segment(r.text, startMs, durationMs))
                    }
                }
                .onFailure { onError(it.message ?: it.toString()) }
        }
    }

    /** 请求的识别语种转成 `zh`/`en`/… 形式，给幻觉过滤器做旁证。 */
    private val expectLang: String? = when (asrLanguage) {
        3 -> "zh"
        4 -> "en"
        7 -> "yue"
        11 -> "ja"
        12 -> "ko"
        else -> null
    }
}
