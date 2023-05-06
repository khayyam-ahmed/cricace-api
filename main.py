import fastapi
from fastapi import UploadFile, File
# from io import BytesIO
from model.battingShotClassification.model import classifyBattingShot
import cv2
import os


# Create a FastAPI app.
app = fastapi.FastAPI()

@app.get("/")
def index():
    return {"lol": "hahah"}

def read_video(file) -> cv2.VideoCapture:
    video = cv2.VideoCapture(file)
    return video

@app.post("/predict/video")
async def predict(video: UploadFile = File(...)):
    with open("temp.mp4", "wb") as buffer:
        buffer.write(await video.read())
    video = read_video("temp.mp4")
    prediction = classifyBattingShot(video)
    video.release()
    os.remove("temp.mp4")
    return prediction

# Run the app.
if __name__ == "__main__":
    app.run(debug=True)