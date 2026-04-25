import tensorflow as tf

# load original model
model = tf.keras.models.load_model("model.keras", compile=False)

# re-save in SAFE legacy-compatible format
model.save("model_fixed.h5")