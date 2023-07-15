## FASTAPI
import sys
from Cyberbullying.pipeline.train_pipeline import TrainPipeline
from fastapi import FastAPI
import uvicorn
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from Cyberbullying.pipeline.predict_pipeline import PredictionPipeline
from Cyberbullying.exception import CustomException
from Cyberbullying.constants import *


text: str = "This is a beautiful text"

app = FastAPI()

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def training():
    try:
        train_pipleine = TrainPipeline()
        train_pipleine.run_pipeline()

        return Response("Training Successfully done!!")
    
    except Exception as e:
        raise Response(f"Error Occurred while training!! {e}")
    
@app.get("/predict")
async def predict_route(text):
    try:
        obj = PredictionPipeline()
        text = obj.run_pipeline(text)

        return text 
    except Exception as e:
        raise CustomException(e,sys) from e
    

## Start the app

if __name__ == "__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)