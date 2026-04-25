import tensorflow as tf

print("Loading model...")

model = tf.keras.models.load_model("model.h5", compile=False)

print("Saving new model...")

model.save("model_fixed.keras")

print("DONE: model_fixed.keras created successfully")