from functions import *
from fastapi import FastAPI
from starlette.requests import Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


app = FastAPI()
origins= ["http://localhost:8888"]
app.add_middleware(CORSMiddleware, allow_origins=origins)
@app.get('/index/{para}')
def result(para: str):
    return paraphraser(para)


# Press the green button in the gutter to run the script.
if _name_ == '_main_':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    