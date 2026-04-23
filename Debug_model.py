from tensorflow.keras.models import load_model

model = load_model("model.h5")

print("INPUT SHAPE:")
print(model.input_shape)

print("OUTPUT SHAPE:")
print(model.output_shape)