package com.hearizon.sync.audio

import java.io.File

/** 极简 WAV 读取：只接受 PCM16 单声道，频率不符则记录原始值交给调用方重采样判断。 */
object WavReader {

    class Pcm(val samples: ShortArray, val sampleRate: Int)

    fun read(file: File): Pcm {
        val bytes = file.readBytes()
        require(bytes.size > 44) { "WAV 太短：${bytes.size}" }
        require(String(bytes, 0, 4, Charsets.US_ASCII) == "RIFF") { "不是 RIFF 文件" }

        var pos = 12
        var sampleRate = 16000
        var channels = 1
        var bits = 16
        var dataStart = -1
        var dataLen = 0

        while (pos + 8 <= bytes.size) {
            val id = String(bytes, pos, 4, Charsets.US_ASCII)
            val size = le32(bytes, pos + 4)
            val body = pos + 8
            when (id) {
                "fmt " -> {
                    channels = le16(bytes, body + 2)
                    sampleRate = le32(bytes, body + 4)
                    bits = le16(bytes, body + 14)
                }
                "data" -> {
                    dataStart = body
                    dataLen = minOf(size, bytes.size - body)
                }
            }
            pos = body + size + (size and 1)
        }
        require(dataStart >= 0) { "WAV 缺少 data 块" }
        require(bits == 16) { "只支持 16bit，实际 $bits" }

        val count = dataLen / 2
        val samples = ShortArray(count) { i ->
            val lo = bytes[dataStart + i * 2].toInt() and 0xFF
            val hi = bytes[dataStart + i * 2 + 1].toInt()
            ((hi shl 8) or lo).toShort()
        }
        if (channels == 2) {
            // 立体声下混成单声道
            val mono = ShortArray(count / 2) { i ->
                ((samples[i * 2].toInt() + samples[i * 2 + 1].toInt()) / 2).toShort()
            }
            return Pcm(mono, sampleRate)
        }
        return Pcm(samples, sampleRate)
    }

    private fun le16(b: ByteArray, o: Int): Int =
        (b[o].toInt() and 0xFF) or ((b[o + 1].toInt() and 0xFF) shl 8)

    private fun le32(b: ByteArray, o: Int): Int =
        (b[o].toInt() and 0xFF) or
            ((b[o + 1].toInt() and 0xFF) shl 8) or
            ((b[o + 2].toInt() and 0xFF) shl 16) or
            ((b[o + 3].toInt() and 0xFF) shl 24)
}
