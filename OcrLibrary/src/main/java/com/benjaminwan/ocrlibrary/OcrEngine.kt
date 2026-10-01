package com.benjaminwan.ocrlibrary

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap

class OcrEngine(
    context: Context,
    detName: String = "ch_PP-OCRv3_det_infer.onnx",
    clsName: String = "ch_ppocr_mobile_v2.0_cls_infer.onnx",
    recName: String = "ch_PP-OCRv3_rec_infer.onnx",
    keysName: String = "ppocr_keys_v1.txt",
) {
    companion object {
        const val numThread: Int = 4
    }

    /**
     * Native OcrLite handle owned by this instance. Each OcrEngine carries its own
     * models, so a second engine (e.g. a Devanagari recognizer) can coexist with the
     * default one in the same process.
     */
    private var nativeHandle: Long = 0

    init {
        System.loadLibrary("RapidOcr")
        nativeHandle = create(numThread, context.assets, detName, clsName, recName, keysName)
        if (nativeHandle == 0L) throw IllegalArgumentException("Failed to create native OcrLite")
    }

    @Suppress("FinalizerSuppression")
    protected fun finalize() {
        if (nativeHandle != 0L) {
            destroy(nativeHandle)
            nativeHandle = 0L
        }
    }

    var padding: Int = 50
    var boxScoreThresh: Float = 0.2f
    var boxThresh: Float = 0.45f
    var unClipRatio: Float = 1.4f
    var doAngle: Boolean = true
    var mostAngle: Boolean = true

    fun detect(input: Bitmap, output: Bitmap, maxSideLen: Int) =
        detect(
            nativeHandle,
            input, output, padding, maxSideLen,
            boxScoreThresh, boxThresh,
            unClipRatio, doAngle, mostAngle
        )

    private external fun create(
        numThread: Int, assetManager: AssetManager,
        detName: String, clsName: String, recName: String, keysName: String,
    ): Long

    private external fun destroy(handle: Long)

    private external fun detect(
        handle: Long,
        input: Bitmap, output: Bitmap, padding: Int, maxSideLen: Int,
        boxScoreThresh: Float, boxThresh: Float,
        unClipRatio: Float, doAngle: Boolean, mostAngle: Boolean,
    ): OcrResult
}
