from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import requests
import io
from keras.models import load_model

app = FastAPI()

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), image_type: str = None, model_type: str = None):
    try:
        if image_type == "brain_mri" and model_type == "ML_model":
            image_array = np.array(Image.open(io.BytesIO(await file.read())))
            model = load_model("alzahimer_keras_model.h5")  # Specify your model
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            class_names = ["ModerateDemented", "NonDemented", "VeryMildDemented", "MildDemented"]
            target_size = (176, 176)

            # Call the external API
            api_url = "https://oluwatomisinayodabo.us-east-1.modelbit.com/v1/predict_image_class/latest"
            payload = {"data": [image_array, model, class_names, target_size]}
            response = requests.post(api_url, json=payload)
            result = response.json()

            return JSONResponse(content=result)
        else:
            return JSONResponse(content={"error": "Invalid request"}, status_code=400)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)