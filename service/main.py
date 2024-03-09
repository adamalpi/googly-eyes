from fastapi import FastAPI, UploadFile
import numpy as np
import cv2

from model import Model, find_eyes


app = FastAPI()
face_segmentation_model = Model("weights.pt")


@app.post("/find_eyes/")
async def create_upload_file(file: UploadFile):
    bytes = await file.read()

    # read image as an numpy array 
    image = np.asarray(bytearray(bytes), dtype="uint8") 
      
    # use imdecode function 
    image = cv2.imdecode(image, cv2.IMREAD_COLOR) 
    pred = face_segmentation_model.predict(image)
    eyes = find_eyes(pred)
    
    return {"eyes": eyes}
