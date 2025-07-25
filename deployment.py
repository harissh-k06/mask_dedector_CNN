import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


model = load_model('maskdedector.keras')
def predict_mask(img):
    img_resized = img.resize((64, 64))
    arr = image.img_to_array(img_resized)
    arr = np.expand_dims(arr, axis=0)
    prediction = model.predict(arr)
    result = "This person is not wearing a mask !!!! " if prediction[0][0] > 0.5 else "This person is wearing a mask !!!!"
    return result, img 
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
