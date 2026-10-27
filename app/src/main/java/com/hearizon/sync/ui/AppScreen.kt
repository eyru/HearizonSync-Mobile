package com.hearizon.sync.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.hearizon.sync.BuildConfig
import com.hearizon.sync.asr.LiveSession
import com.hearizon.sync.asr.SenseVoiceEngine
import com.hearizon.sync.audio.WavReader
import com.hearizon.sync.asr.LangDetect
import com.hearizon.sync.mt.MarianTranslator
import com.hearizon.sync.store.Exporter
import com.hearizon.sync.store.ModelInstaller
import com.hearizon.sync.store.Prefs
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

private sealed interface EngineState {
    data class Loading(val progress: Float = -1f) : EngineState
    data class Ready(val loadMs: Long) : EngineState
    data class Failed(val message: String) : EngineState
}

private val clock: SimpleDateFormat = SimpleDateFormat("HH:mm:ss", Locale.US)

@Composable
fun AppScreen() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val mainHandler = remember { Handler(Looper.getMainLooper()) }
    val prefs = remember { Prefs(context) }

    var engine by remember { mutableStateOf<SenseVoiceEngine?>(null) }
    var engineState by remember { mutableStateOf<EngineState>(EngineState.Loading()) }
    /** 中→英 */
    var translator by remember { mutableStateOf<MarianTranslator?>(null) }
    /** 英→中（懒加载：第一次说到英语才装载，省 113MB 与几秒启动时间） */
    var translatorEnZh by remember { mutableStateOf<MarianTranslator?>(null) }
    /** 日→英、韩→英：小语种走「→英→中」两跳 */
    var translatorJaEn by remember { mutableStateOf<MarianTranslator?>(null) }
    var translatorKoEn by remember { mutableStateOf<MarianTranslator?>(null) }
    var mtReady by remember { mutableStateOf(false) }
    val mtMutex = remember { Mutex() }
    val enZhLock = remember { Mutex() }
    val jaEnLock = remember { Mutex() }
    val koEnLock = remember { Mutex() }
    var session by remember { mutableStateOf<LiveSession?>(null) }
    var isRunning by remember { mutableStateOf(false) }
    var speaking by remember { mutableStateOf(false) }
    var busy by remember { mutableStateOf(false) }
    var level by remember { mutableFloatStateOf(0f) }
    var notice by remember { mutableStateOf("") }

    // ---- 设置（本地状态镜像 prefs，改动即写回）
    var showSettings by remember { mutableStateOf(false) }
    var showTranslation by remember { mutableStateOf(prefs.showTranslation) }
    var enhance by remember { mutableStateOf(prefs.enhance) }
    var fontScale by remember { mutableFloatStateOf(prefs.fontScale) }
    var exportTs by remember { mutableStateOf(prefs.exportWithTimestamp) }
    var exportNotice by remember { mutableStateOf("") }
    var lastExport by remember { mutableStateOf(prefs.lastExportPath) }
    /** 翻译方向 auto / zh2en / en2zh */
    var direction by remember { mutableStateOf(prefs.translateDirection) }
    /** 识别语言 0=自动 3=中 4=英 7=粤 11=日 12=韩 */
    var asrLang by remember { mutableStateOf(prefs.asrLanguage) }

    val lines = remember { mutableStateListOf<CaptionLine>() }
    val listState = rememberLazyListState()

    var hasMic by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED
        )
    }
    val micLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> hasMic = granted }

    // 引擎加载（内置模型首次启动先落盘 350MB，之后秒开）
    LaunchedEffect(Unit) {
        val started = SystemClock.elapsedRealtime()
        val loaded = withContext(Dispatchers.IO) {
            runCatching {
                val total = ModelInstaller.totalAssetsBytes(context).coerceAtLeast(1L)
                var done = 0L
                fun stage(path: String): File {
                    val f = ModelInstaller.ensure(context, path) { copied, _ ->
                        engineState = EngineState.Loading(((done + copied).toFloat() / total).coerceIn(0f, 1f))
                    }
                    done += f.length()
                    return f
                }
                val asr = stage(ModelInstaller.Paths.ASR_MODEL)
                val tok = stage(ModelInstaller.Paths.ASR_TOKENS)
                SenseVoiceEngine(
                    modelPath = asr.absolutePath,
                    tokensPath = tok.absolutePath,
                )
            }
        }
        loaded.onSuccess {
            engine = it
            engineState = EngineState.Ready(SystemClock.elapsedRealtime() - started)
            // 翻译模型（110MB）随后装载；失败只是没有英文，不影响字幕
            val mtLoaded = withContext(Dispatchers.IO) {
                runCatching {
                    val enc = ModelInstaller.ensure(context, ModelInstaller.Paths.MT_ENCODER)
                    ModelInstaller.ensure(context, ModelInstaller.Paths.MT_DECODER)
                    MarianTranslator(context, enc.parentFile ?: error("翻译模型目录缺失"))
                }
            }
            mtLoaded.onSuccess { tr ->
                translator = tr
                mtReady = true
            }.onFailure { e ->
                notice = "翻译未启用：${e.message ?: e}"
            }
        }.onFailure {
            engineState = EngineState.Failed(it.message ?: it.toString())
        }
    }

    // 新字幕出现时滚到底
    LaunchedEffect(lines.size) {
        if (lines.isNotEmpty()) listState.animateScrollToItem(lines.lastIndex)
    }

    fun stopListening() {
        session?.stop()
        session = null
        isRunning = false
        speaking = false
        level = 0f
    }

    /** 装载一对 ONNX 模型（落盘 + 建会话）；失败只影响译文，不影响字幕。 */
    suspend fun loadPair(
        lock: Mutex,
        encPath: String,
        decPath: String,
        tokenizerDir: String,
        label: String,
        padToken: Int = 65000,
    ): MarianTranslator? = lock.withLock {
        val loaded = withContext(Dispatchers.IO) {
            runCatching {
                val enc = ModelInstaller.ensure(context, encPath)
                ModelInstaller.ensure(context, decPath)
                MarianTranslator(context, enc.parentFile ?: error("$label 目录缺失"), tokenizerDir, padToken)
            }
        }
        loaded.onFailure { e -> notice = "$label 未启用：${e.message ?: e}" }
        loaded.getOrNull()
    }

    suspend fun ensureEnZh(): MarianTranslator? =
        translatorEnZh ?: loadPair(
            enZhLock, ModelInstaller.Paths.MT2_ENCODER, ModelInstaller.Paths.MT2_DECODER, "mt_en_zh", "英→中"
        )?.also { translatorEnZh = it }

    suspend fun ensureJaEn(): MarianTranslator? =
        translatorJaEn ?: loadPair(
            jaEnLock, ModelInstaller.Paths.MT_JA_ENCODER, ModelInstaller.Paths.MT_JA_DECODER,
            "mt_ja_en", "日→英", padToken = 60715,
        )?.also { translatorJaEn = it }

    suspend fun ensureKoEn(): MarianTranslator? =
        translatorKoEn ?: loadPair(
            koEnLock, ModelInstaller.Paths.MT_KO_ENCODER, ModelInstaller.Paths.MT_KO_DECODER, "mt_ko_en", "韩→英"
        )?.also { translatorKoEn = it }

    /**
     * 译文清洗：
     * 1. 去掉模型偶尔吐出的 `<unk>` 串；
     * 2. 整串是同一片段的 3 次以上重复（「早早早」「好的好的好的」）→ 只留一次；
     * 3. 按逗号去掉完全重复的片段（韩→中会出现「早上好,早,早安,早安」）。
     */
    fun tidy(s: String): String {
        var t = s.replace("<unk>", "").replace("<pad>", "").trim()
        if (t.isEmpty()) return ""
        val n = t.length
        for (unit in 1..(n / 3)) {
            if (n % unit != 0) continue
            val piece = t.substring(0, unit)
            if (piece.repeat(n / unit) == t) return piece
        }
        val parts = t.split(',', '，').map { it.trim() }.filter { it.isNotEmpty() }
        if (parts.size < 2) return t
        val kept = ArrayList<String>(parts.size)
        for (p in parts) if (!kept.contains(p)) kept.add(p)
        return kept.joinToString("，")
    }

    suspend fun translateOne(tr: MarianTranslator?, src: String): String? {
        val t = tr ?: return null
        val out = withContext(Dispatchers.IO) {
            runCatching { mtMutex.withLock { t.translate(src) } }.getOrElse { "" }
        }
        return out.ifBlank { null }?.let { tidy(it) }?.ifBlank { null }
    }

    /** 日语/韩语先翻成英语，再由「英→中」落到中文。 */
    suspend fun viaEnglish(first: suspend () -> MarianTranslator?, src: String): String? {
        val en = translateOne(first(), src) ?: return null
        return translateOne(translatorEnZh ?: ensureEnZh(), en)
    }

    suspend fun autoRoute(src: String, lang: LangDetect.Lang = LangDetect.detect(src)): String? = when (lang) {
        LangDetect.Lang.ZH -> translateOne(translator, src)
        LangDetect.Lang.EN -> translateOne(translatorEnZh ?: ensureEnZh(), src)
        LangDetect.Lang.JA -> viaEnglish({ ensureJaEn() }, src)
        LangDetect.Lang.KO -> viaEnglish({ ensureKoEn() }, src)
        else -> null
    }

    /**
     * 按当前设置 + 这句话本身的语种决定怎么翻。
     *
     * 为什么不能只用一个模型：说英语时 SenseVoice 输出的就是英文，交给「中→英」
     * 等于把英语当中文再翻一遍，结果还是英语——这就是用户看到的「说英语输出英语」。
     *
     * 为什么还要兜一道语种判断：识别语言判断偶尔出错（选「韩→中」却把韩语解成中文/日文），
     * 死板按方向喂模型只会得到更离谱的结果，语种不符就回退到自动选路。
     */
    suspend fun translateText(src: String): String? {
        val lang = LangDetect.detect(src)
        return when (direction) {
            Prefs.DIR_ZH2EN -> if (lang == LangDetect.Lang.ZH) translateOne(translator, src) else autoRoute(src, lang)
            Prefs.DIR_EN2ZH -> if (lang == LangDetect.Lang.EN) translateOne(translatorEnZh ?: ensureEnZh(), src) else autoRoute(src, lang)
            Prefs.DIR_JA2ZH -> if (lang == LangDetect.Lang.JA) viaEnglish({ ensureJaEn() }, src) else autoRoute(src, lang)
            Prefs.DIR_KO2ZH -> if (lang == LangDetect.Lang.KO) viaEnglish({ ensureKoEn() }, src) else autoRoute(src, lang)
            else -> autoRoute(src, lang)
        }
    }

    // 翻译自测（设置页）：四方向各跑一句固定样例，出错也照样显示，用来排查模型问题
    var selfTest by remember { mutableStateOf("") }
    fun runMtSelfTest() {
        selfTest = "测试中…（首次会落盘 100MB 级模型，请稍候）"
        scope.launch {
            val sb = StringBuilder()
            suspend fun line(name: String, block: suspend () -> String?) {
                val r = runCatching { block() }.getOrElse { "出错：" + (it.message ?: it.toString()) }
                sb.append(name).append("：").append(r ?: "（无译文）").append('\n')
            }
            line("中→英") { translateOne(translator, "今天天气不错。") }
            line("英→中") { translateOne(translatorEnZh ?: ensureEnZh(), "What time is it?") }
            line("日→中") { viaEnglish({ ensureJaEn() }, "こんにちは、今日はいい天気ですね。") }
            line("韩→中") { viaEnglish({ ensureKoEn() }, "안녕하세요. 오늘 날씨가 좋네요.") }
            selfTest = sb.toString().trim()
        }
    }

    /**
     * 加一行字幕。
     *
     * @param startMs 相对会话开始的偏移（SRT 导出用）
     * @param durationMs 该段音频时长（幻觉过滤与 SRT 结束时间都要用）
     */
    fun addCaption(text: String, tag: String = "", startMs: Long = 0L, durationMs: Long = 0L) {
        val id = System.nanoTime()
        lines.add(CaptionLine(id, clock.format(Date()), text, tag, startMs = startMs, durationMs = durationMs))
        if (!showTranslation) return
        scope.launch {
            val out = translateText(text) ?: return@launch
            if (out.isNotEmpty()) {
                val idx = lines.indexOfFirst { it.id == id }
                if (idx >= 0) lines[idx] = lines[idx].copy(en = out)
            }
        }
    }

    fun startListening() {
        val eng = engine ?: return
        val live = LiveSession(
            engine = eng,
            onSegment = { seg ->
                mainHandler.post {
                    addCaption(seg.text, "", seg.startMs, seg.durationMs)
                }
            },
            onLevel = { v -> mainHandler.post { level = v } },
            onSpeaking = { sp -> mainHandler.post { speaking = sp } },
            onError = { msg -> mainHandler.post { notice = "识别出错：$msg" } },
            enhance = enhance,
            asrLanguage = asrLang,
        )
        if (live.start()) {
            session = live
            isRunning = true
            notice = ""
        } else {
            notice = "麦克风打不开（设备被占用或模拟器无输入）"
        }
    }

    /**
     * 切翻译方向时同步「识别语言」。
     *
     * SenseVoice 的 language 参数是每段识别时喂进去的 token：不给它（0=auto）时它自己猜，
     * 韩语这种和中文/日语音系接近的语言很容易被判成 zh/ja，输出就成了中文或日文。
     * 所以选「韩→中」就固定 language=ko(12)，选「日→中」固定 ja(11)。
     * 会话创建时语言已固定，正在监听时就重开一次。
     */
    fun applyDirection(dir: String) {
        direction = dir
        prefs.translateDirection = dir
        val lang = when (dir) {
            Prefs.DIR_ZH2EN -> 3      // zh
            Prefs.DIR_EN2ZH -> 4      // en
            Prefs.DIR_JA2ZH -> 11     // ja
            Prefs.DIR_KO2ZH -> 12     // ko
            else -> 0                 // auto
        }
        if (asrLang != lang) {
            asrLang = lang
            prefs.asrLanguage = lang
            if (isRunning) {
                stopListening()
                startListening()
            }
        }
    }

    fun selfTest() {        if (busy || engine == null) return
        busy = true
        scope.launch {
            val outcome = withContext(Dispatchers.IO) {
                runCatching {
                    val pcm = WavReader.read(ModelInstaller.ensure(context, "demo/zh_16k.wav"))
                    val t0 = SystemClock.elapsedRealtime()
                    val text = engine!!.recognize(pcm.samples)
                    Triple(text, SystemClock.elapsedRealtime() - t0, pcm.samples.size / 16000.0)
                }
            }
            outcome.onSuccess { (text, ms, secs) ->
                // 耗时放 tag（只用于界面显示），正文保持纯净 —— 导出 SRT/TXT 时不能带上这种调试信息
                addCaption(
                    text,
                    tag = "自测 ${ms}ms / ${String.format(Locale.US, "%.1f", secs)}s",
                    durationMs = (secs * 1000).toLong(),
                )
            }.onFailure {
                notice = "自测失败：${it.message}"
            }
            busy = false
        }
    }

    /** 回放测试：把测试音频按 20ms 帧喂进真实链路（VAD→切句→识别），验证不止「整段识别」。 */
    fun replayTest() {
        if (busy || engine == null) return
        busy = true
        scope.launch {
            withContext(Dispatchers.IO) {
                val pcm = WavReader.read(ModelInstaller.ensure(context, "demo/zh_16k.wav")).samples
                val live = LiveSession(
                    engine = engine!!,
                    onSegment = { seg ->
                        mainHandler.post {
                            addCaption(seg.text, "回放", seg.startMs, seg.durationMs)
                        }
                    },
                    onLevel = {},
                    onSpeaking = {},
                    microphone = false,
                    enhance = enhance,
                    asrLanguage = asrLang,
                )
                live.start()
                var offset = 0
                while (offset + 320 <= pcm.size) {
                    live.feed(pcm.copyOfRange(offset, offset + 320))
                    offset += 320
                    delay(4)      // 5 倍速回放
                }
                live.stop()
                live.awaitIdle()
            }
            busy = false
        }
    }

    fun doExport(format: Exporter.Format) {
        val snapshot = lines.toList()
        scope.launch {
            val result = withContext(Dispatchers.IO) {
                Exporter.export(context, snapshot.toCues(), format, exportTs)
            }
            exportNotice = if (result.ok) result.message else result.message
            if (result.ok) {
                lastExport = result.path
                prefs.lastExportPath = result.path
            }
        }
    }

    if (showSettings) {
        SettingsScreen(
            showTranslation = showTranslation,
            enhance = enhance,
            fontScale = fontScale,
            exportWithTimestamp = exportTs,
            direction = direction,
            asrLanguage = asrLang,
            enZhLoaded = translatorEnZh != null,
            lines = lines,
            lastExportPath = lastExport,
            notice = exportNotice,
            selfTestText = selfTest,
            onShowTranslationChange = {
                showTranslation = it
                prefs.showTranslation = it
            },
            onEnhanceChange = {
                enhance = it
                prefs.enhance = it
            },
            onFontScaleChange = {
                fontScale = it
                prefs.fontScale = it
            },
            onExportTimestampChange = {
                exportTs = it
                prefs.exportWithTimestamp = it
            },
            onDirectionChange = { applyDirection(it) },
            onAsrLanguageChange = {
                asrLang = it
                prefs.asrLanguage = it
            },
            onSelfTest = { runMtSelfTest() },
            onExport = { doExport(it) },
            onBack = { showSettings = false },
        )
        return
    }

    Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
        Column(modifier = Modifier.fillMaxSize()) {
            TopBar(
                state = engineState,
                isRunning = isRunning,
                speaking = speaking,
                mtReady = mtReady,
                direction = direction,
                onDirectionChange = { applyDirection(it) },
                onSettings = {
                    exportNotice = ""
                    showSettings = true
                },
            )

            Box(modifier = Modifier.fillMaxWidth().weight(1f).padding(horizontal = 20.dp)) {
                when {
                    busy -> Column(
                        modifier = Modifier.fillMaxSize(),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.Center,
                    ) {
                        CircularProgressIndicator(modifier = Modifier.size(32.dp), strokeWidth = 3.dp)
                        Spacer(Modifier.height(12.dp))
                        Text("正在识别…", color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }

                    lines.isEmpty() -> EmptyHint(engineState, isRunning)

                    else -> LazyColumn(
                        state = listState,
                        modifier = Modifier.fillMaxSize(),
                        verticalArrangement = Arrangement.spacedBy(10.dp),
                        contentPadding = androidx.compose.foundation.layout.PaddingValues(vertical = 12.dp),
                    ) {
                        items(lines) { line -> CaptionRow(line, fontScale, showTranslation) }
                    }
                }
            }

            BottomBar(
                isRunning = isRunning,
                speaking = speaking,
                level = level,
                hasMic = hasMic,
                busy = busy,
                engineReady = engine != null,
                notice = notice,
                onToggle = {
                    if (!hasMic) {
                        micLauncher.launch(Manifest.permission.RECORD_AUDIO)
                    } else if (isRunning) {
                        stopListening()
                    } else {
                        startListening()
                    }
                },
                onSelfTest = { selfTest() },
                onReplay = { replayTest() },
            )
        }
    }
}

@Composable
private fun TopBar(
    state: EngineState,
    isRunning: Boolean,
    speaking: Boolean,
    mtReady: Boolean,
    direction: String,
    onDirectionChange: (String) -> Unit,
    onSettings: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .statusBarsPadding()
            .padding(start = 20.dp, end = 8.dp, top = 14.dp, bottom = 14.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(
            modifier = Modifier
                .size(10.dp)
                .clip(CircleShape)
                .background(
                    when {
                        state is EngineState.Failed -> MaterialTheme.colorScheme.error
                        isRunning -> MaterialTheme.colorScheme.primary
                        else -> MaterialTheme.colorScheme.outline
                    }
                )
        )
        Spacer(Modifier.width(10.dp))
        Column {
            Text(
                text = "HearizonSync Mobile",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                text = "v${BuildConfig.VERSION_NAME} · 离线字幕" + if (mtReady) " + 翻译" else "",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Spacer(Modifier.weight(1f))
        LanguageChip(direction = direction, onChange = onDirectionChange)
        Spacer(Modifier.width(6.dp))
        val badge = when {
            state is EngineState.Loading -> "引擎加载中"
            state is EngineState.Failed -> "引擎失败"
            speaking -> "识别中"
            isRunning -> "监听中"
            else -> "就绪"
        }
        Surface(shape = RoundedCornerShape(50), color = MaterialTheme.colorScheme.surfaceVariant) {
            Text(
                text = badge,
                modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        TextButton(onClick = onSettings, contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 10.dp)) {
            Text("设置", style = MaterialTheme.typography.labelMedium)
        }
    }
}

@Composable
private fun LanguageChip(direction: String, onChange: (String) -> Unit) {
    var open by remember { mutableStateOf(false) }
    val label = when (direction) {
        Prefs.DIR_ZH2EN -> "中→英"
        Prefs.DIR_EN2ZH -> "英→中"
        Prefs.DIR_JA2ZH -> "日→中"
        Prefs.DIR_KO2ZH -> "韩→中"
        else -> "自动"
    }
    Box {
        Surface(
            shape = RoundedCornerShape(50),
            color = MaterialTheme.colorScheme.primaryContainer,
            onClick = { open = true },
        ) {
            Text(
                text = "$label ⌄",
                modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onPrimaryContainer,
            )
        }
        DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
            listOf(
                Prefs.DIR_AUTO to "自动判断语种",
                Prefs.DIR_ZH2EN to "中文 → 英文",
                Prefs.DIR_EN2ZH to "英文 → 中文",
                Prefs.DIR_JA2ZH to "日文 → 中文",
                Prefs.DIR_KO2ZH to "韩文 → 中文",
            ).forEach { (value, name) ->
                DropdownMenuItem(
                    text = {
                        Text(
                            text = (if (direction == value) "✓ " else "   ") + name,
                            style = MaterialTheme.typography.bodyMedium,
                        )
                    },
                    onClick = {
                        open = false
                        onChange(value)
                    },
                )
            }
        }
    }
}

@Composable
private fun EmptyHint(state: EngineState, isRunning: Boolean) {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(
            text = when (state) {
                is EngineState.Loading -> "正在准备离线模型"
                is EngineState.Failed -> "引擎没能加载"
                else -> if (isRunning) "请开始说话" else "准备好了就开始"
            },
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.Medium,
        )
        Spacer(Modifier.height(8.dp))
        Text(
            text = when (state) {
                is EngineState.Loading -> if (state.progress >= 0f) {
                    "首次启动解包模型 ${(state.progress * 100).toInt()}%"
                } else {
                    "首次启动要解包 350MB 模型，大约十几秒"
                }
                is EngineState.Failed -> state.message
                else -> if (isRunning) {
                    "说话即出字幕；停顿 0.3 秒自动断句"
                } else {
                    "点开始边听边出字幕；也可以先点「整段自测」验证链路"
                }
            },
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )
    }
}

@Composable
private fun CaptionRow(line: CaptionLine, fontScale: Float, showTranslation: Boolean) {
    val body = MaterialTheme.typography.bodyLarge.copy(
        fontSize = (MaterialTheme.typography.bodyLarge.fontSize.value * fontScale).sp,
        lineHeight = (MaterialTheme.typography.bodyLarge.lineHeight.value * fontScale).sp,
    )
    val sub = MaterialTheme.typography.bodyMedium.copy(
        fontSize = (MaterialTheme.typography.bodyMedium.fontSize.value * fontScale * 0.92f).sp,
        lineHeight = (MaterialTheme.typography.bodyMedium.lineHeight.value * fontScale * 0.92f).sp,
    )
    Row(verticalAlignment = Alignment.Top) {
        Text(
            text = line.time,
            style = MaterialTheme.typography.labelSmall,
            fontFamily = FontFamily.Monospace,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.width(58.dp).padding(top = 4.dp),
        )
        Column(modifier = Modifier.weight(1f)) {
            if (line.tag.isNotEmpty()) {
                Text(
                    text = line.tag,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.primary,
                )
            }
            Text(text = line.text, style = body, lineHeight = body.lineHeight * 1.35f)
            if (line.en.isNotEmpty() && showTranslation) {
                Spacer(Modifier.height(3.dp))
                Text(
                    text = line.en,
                    style = sub,
                    color = MaterialTheme.colorScheme.primary,
                    lineHeight = sub.lineHeight * 1.3f,
                )
            }
        }
    }
}

@Composable
private fun BottomBar(
    isRunning: Boolean,
    speaking: Boolean,
    level: Float,
    hasMic: Boolean,
    busy: Boolean,
    engineReady: Boolean,
    notice: String,
    onToggle: () -> Unit,
    onSelfTest: () -> Unit,
    onReplay: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .navigationBarsPadding()
            .padding(horizontal = 24.dp, vertical = 12.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        if (notice.isNotEmpty()) {
            Text(
                text = notice,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.error,
                textAlign = TextAlign.Center,
            )
            Spacer(Modifier.height(6.dp))
        }

        // 电平条：只在监听时显示，给「麦在工作」一个直观反馈
        if (isRunning) {
            Box(
                modifier = Modifier
                    .fillMaxWidth(0.5f)
                    .height(4.dp)
                    .clip(RoundedCornerShape(2.dp))
                    .background(MaterialTheme.colorScheme.surfaceVariant)
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth(level.coerceIn(0f, 1f))
                        .height(4.dp)
                        .clip(RoundedCornerShape(2.dp))
                        .background(
                            if (speaking) MaterialTheme.colorScheme.primary
                            else MaterialTheme.colorScheme.outline
                        )
                )
            }
            Spacer(Modifier.height(10.dp))
        }

        Text(
            text = when {
                !hasMic -> "需要麦克风权限"
                isRunning -> if (speaking) "正在听你说…" else "监听中 · 点击停止"
                else -> "点击开始"
            },
            style = MaterialTheme.typography.labelMedium,
            fontFamily = FontFamily.Monospace,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(Modifier.height(10.dp))
        Surface(
            onClick = onToggle,
            shape = CircleShape,
            color = if (isRunning) MaterialTheme.colorScheme.error
            else MaterialTheme.colorScheme.primary,
            modifier = Modifier.size(72.dp),
        ) {
            Box(contentAlignment = Alignment.Center) {
                Text(
                    text = if (isRunning) "停止" else "开始",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onPrimary,
                )
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
            TextButton(onClick = onSelfTest, enabled = engineReady && !busy) {
                Text("整段自测")
            }
            TextButton(onClick = onReplay, enabled = engineReady && !busy) {
                Text("回放·走VAD切句")
            }
        }
    }
}
