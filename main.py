import fastapi
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
# from io import BytesIO
from model.battingShotClassification.model import classifyBattingShot
from model.bowlingTypeClassificationModels.model import classifyBowlingType
import cv2
import os
import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# import uvicorn

origins = ["*"]

# Create a FastAPI app.
app = fastapi.FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


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

    battingType = classifyBattingShot(video)
    bowlingType = classifyBowlingType()

    prediction = {}
    prediction.update(bowlingType)
    prediction.update(battingType)

    video.release()
    os.remove("temp.mp4")
    return prediction

app.include_router(ToolsRoutes.router)

# Run the app.
# if __name__ == "__main__":
    # uvicorn.run(app, port=8000, host="0.0.0.0")
