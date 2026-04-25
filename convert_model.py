import tensorflow as tf

model = tf.keras.models.load_model("model.h5")
model.save("model.keras")

print("DONE: model.keras created")