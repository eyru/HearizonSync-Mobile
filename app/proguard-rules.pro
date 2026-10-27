# ONNX Runtime 反射调用 JNI 层，别裁
-keep class ai.onnxruntime.** { *; }
-keepclasseswithmembernames class * {
    native <methods>;
}
