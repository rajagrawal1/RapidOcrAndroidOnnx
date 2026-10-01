#include "OcrResultUtils.h"
#include "BitmapUtils.h"
#include "OcrLite.h"
#include "OcrUtils.h"

// Each Java OcrEngine owns its own native OcrLite instance, referenced by the
// long handle returned from create() — multiple engines with different
// recognition models can coexist in one process.

static OcrLite *handleToOcrLite(jlong handle) {
    return reinterpret_cast<OcrLite *>(handle);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_benjaminwan_ocrlibrary_OcrEngine_create(JNIEnv *env, jobject thiz, jint numThread,
                                                 jobject assetManager, jstring detName, jstring clsName,
                                                 jstring recName, jstring keysName) {
    auto *lite = new OcrLite();
    std::string modelDetName = jstringTostring(env, detName);
    std::string modelClsName = jstringTostring(env, clsName);
    std::string modelRecName = jstringTostring(env, recName);
    std::string modelKeysName = jstringTostring(env, keysName);
    lite->init(env, assetManager, numThread, modelDetName, modelClsName, modelRecName, modelKeysName);
    return reinterpret_cast<jlong>(lite);
}

extern "C" JNIEXPORT void JNICALL
Java_com_benjaminwan_ocrlibrary_OcrEngine_destroy(JNIEnv *env, jobject thiz, jlong handle) {
    if (handle != 0) {
        delete handleToOcrLite(handle);
    }
}

cv::Mat makePadding(cv::Mat &src, const int padding) {
    if (padding <= 0) return src;
    cv::Scalar paddingScalar = {255, 255, 255};
    cv::Mat paddingSrc;
    cv::copyMakeBorder(src, paddingSrc, padding, padding, padding, padding, cv::BORDER_ISOLATED,
                       paddingScalar);
    return paddingSrc;
}

extern "C"
JNIEXPORT jobject JNICALL
Java_com_benjaminwan_ocrlibrary_OcrEngine_detect(JNIEnv *env, jobject thiz, jlong handle,
                                                 jobject input, jobject output,
                                                 jint padding, jint maxSideLen, jfloat boxScoreThresh, jfloat boxThresh,
                                                 jfloat unClipRatio, jboolean doAngle, jboolean mostAngle) {
    OcrLite *ocrLite = handleToOcrLite(handle);
    Logger("padding(%d),maxSideLen(%d),boxScoreThresh(%f),boxThresh(%f),unClipRatio(%f),doAngle(%d),mostAngle(%d)",
           padding, maxSideLen, boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);
    cv::Mat imgRGBA, imgBGR, imgOut;
    bitmapToMat(env, input, imgRGBA);
    cv::cvtColor(imgRGBA, imgBGR, cv::COLOR_RGBA2BGR);
    int originMaxSide = (std::max)(imgBGR.cols, imgBGR.rows);
    int resize;
    if (maxSideLen <= 0 || maxSideLen > originMaxSide) {
        resize = originMaxSide;
    } else {
        resize = maxSideLen;
    }
    resize += 2*padding;
    cv::Rect paddingRect(padding, padding, imgBGR.cols, imgBGR.rows);
    cv::Mat paddingSrc = makePadding(imgBGR, padding);
    //按比例缩小图像，减少文字分割时间
    ScaleParam s = getScaleParam(paddingSrc, resize);//例：按长或宽缩放 src.cols=不缩放，src.cols/2=长度缩小一半
    OcrResult ocrResult = ocrLite->detect(paddingSrc, paddingRect, s, boxScoreThresh, boxThresh,
                                          unClipRatio, doAngle, mostAngle);

    cv::cvtColor(ocrResult.boxImg, imgOut, cv::COLOR_BGR2RGBA);
    matToBitmap(env, imgOut, output);

    return OcrResultUtils(env, ocrResult, output).getJObject();
}
