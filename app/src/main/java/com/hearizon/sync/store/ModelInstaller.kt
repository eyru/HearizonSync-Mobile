package com.hearizon.sync.store

import android.content.Context
import java.io.File
import java.io.FileOutputStream

/**
 * 模型安装器：把 assets 里的模型在首次启动时复制到应用私有目录。
 *
 * 为什么不在运行时直接读 assets：
 *  - ONNX Runtime 的 Java 接口按文件路径加载，assets 只能整块读进内存；
 *  - 大模型（ASR 239MB）整块读入会在低端机上 OOM；
 *  - 复制一次后后续启动零成本，且卸载应用即清理。
 */
object ModelInstaller {

    private const val BUF = 1 shl 20

    fun file(context: Context, assetPath: String): File = File(context.filesDir, assetPath)

    /** 资产是否已在私有目录就绪（大小一致即认为就绪，不做逐字节校验以省启动时间）。 */
    fun isReady(context: Context, assetPath: String): Boolean {
        val f = file(context, assetPath)
        if (!f.isFile || f.length() <= 0L) return false
        return runCatching {
            context.assets.openFd(assetPath).use { it.length == f.length() }
        }.getOrDefault(f.length() > 0L)
    }

    /**
     * 确保资产已在私有目录，返回本地文件。
     * @param onProgress 已复制字节 / 总字节，可在工作线程回调。
     */
    fun ensure(context: Context, assetPath: String, onProgress: ((Long, Long) -> Unit)? = null): File {
        val dest = file(context, assetPath)
        if (dest.isFile && dest.length() > 0L) return dest

        dest.parentFile?.mkdirs()
        val total = runCatching { context.assets.openFd(assetPath).length }.getOrDefault(0L)
        val tmp = File(dest.parentFile, dest.name + ".part")
        context.assets.open(assetPath).use { input ->
            FileOutputStream(tmp).use { out ->
                val buf = ByteArray(BUF)
                var written = 0L
                while (true) {
                    val n = input.read(buf)
                    if (n <= 0) break
                    out.write(buf, 0, n)
                    written += n
                    onProgress?.invoke(written, total)
                }
                out.flush()
                out.fd.sync()
            }
        }
        if (dest.exists()) dest.delete()
        if (!tmp.renameTo(dest)) {
            tmp.copyTo(dest, overwrite = true)
            tmp.delete()
        }
        return dest
    }

    /** 模型文件的资产路径常量。 */
    object Paths {
        const val ASR_MODEL = "models/sv/model.int8.onnx"
        const val ASR_TOKENS = "models/sv/tokens.txt"

        /** 中→英（opus-mt-zh-en） */
        const val MT_ENCODER = "models/mt/encoder.onnx"
        const val MT_DECODER = "models/mt/decoder.onnx"
        const val MT_SRC_SPM = "mt/src_spm.txt"
        const val MT_VOCAB = "mt/vocab.txt"

        /** 英→中（opus-mt-en-zh），首次需要时再落盘，不占启动时间 */
        const val MT2_ENCODER = "models/mt_en_zh/encoder.onnx"
        const val MT2_DECODER = "models/mt_en_zh/decoder.onnx"
        const val MT2_SRC_SPM = "mt_en_zh/src_spm.txt"
        const val MT2_VOCAB = "mt_en_zh/vocab.txt"

        /** 日→英、韩→英（小语种先到英文，再由英→中落到中文） */
        const val MT_JA_ENCODER = "models/mt_ja_en/encoder.onnx"
        const val MT_JA_DECODER = "models/mt_ja_en/decoder.onnx"
        const val MT_JA_SRC_SPM = "mt_ja_en/src_spm.txt"
        const val MT_JA_VOCAB = "mt_ja_en/vocab.txt"

        const val MT_KO_ENCODER = "models/mt_ko_en/encoder.onnx"
        const val MT_KO_DECODER = "models/mt_ko_en/decoder.onnx"
        const val MT_KO_SRC_SPM = "mt_ko_en/src_spm.txt"
        const val MT_KO_VOCAB = "mt_ko_en/vocab.txt"
    }

    /**
     * 启动阶段需要落盘的字节总数（只算语音识别 + 中→英；英→中按需懒加载，
     * 不计入，否则首次启动进度会停在 100% 之前）。
     */
    fun totalAssetsBytes(context: Context): Long {
        val paths = listOf(Paths.ASR_MODEL, Paths.ASR_TOKENS, Paths.MT_ENCODER, Paths.MT_DECODER)
        return paths.sumOf { p -> runCatching { context.assets.openFd(p).length }.getOrDefault(0L) }
    }
}
