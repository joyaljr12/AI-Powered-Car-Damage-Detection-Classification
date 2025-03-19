from fastapi import FastAPI, File, UploadFile
from typing import Union
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_helper import predict

app = FastAPI()

    
@app.post('/predict')
async def get_prediction(file: Union[UploadFile, None] = File(None)):  # Optional file
    try:
        if file is None:
            return {"error": "No file uploaded"}
        
        image_bytes = await file.read()
        image_path = 'temp_file.jpg'

        with open(image_path, 'wb') as f:
            f.write(image_bytes)

        prediction = predict(image_path)
        return {'Prediction': prediction}

    except Exception as e:
        return {'error': str(e)}




