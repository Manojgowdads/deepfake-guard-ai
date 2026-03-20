import tensorflow as tf

print("⏳ Loading the heavy Cloud Model...")
model = tf.keras.models.load_model('deepfake_detector_model.h5')

print("🗜️ Compressing into Mobile Edge format...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# This line does the magic: it shrinks the math so the phone's battery doesn't drain
converter.optimizations = [tf.lite.Optimize.DEFAULT] 

tflite_model = converter.convert()

with open('deepfake_mobile.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ SUCCESS: 'deepfake_mobile.tflite' is ready for Android Studio!")
