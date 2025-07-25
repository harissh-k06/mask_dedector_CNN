import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load your trained model
model = load_model('maskdedector.keras')

# Function to make predictions and return image + result
def predict_mask(img):
    # Prediction
    img_resized = img.resize((64, 64))
    arr = image.img_to_array(img_resized)
    arr = np.expand_dims(arr, axis=0)
    prediction = model.predict(arr)
    # Class logic
    result = "This person is not wearing a mask !!!! " if prediction[0][0] > 0.5 else "This person is wearing a mask !!!!"
    return result, img  # Return BOTH result string and original uploaded image

# Gradio interface setup
iface = gr.Interface(
    fn=predict_mask,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Image(type="pil", label="Uploaded Image")
    ],
    title="Mask Detection",
    description="Upload an image to check if a person is wearing a mask. "
)

iface.launch(share= True)
