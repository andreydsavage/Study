import PIL 
from fastapi import FastAPI, File, UploadFile 
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from utils.model_func import class_id_to_label, load_model, transform_image

app = FastAPI()


@app.get('/') # запрос гет просто спрашивает что есть по адресу
def return_info():
    return 'There is simplest FastAPI app'


@app.post('/classify') # запрос пост отправляет данные по адресу
def classify(file: UploadFile = File(...)):
    image = PIL.Image.open(file.file)
    adapted_image = transform_image(image)
    model = load_model()
    pred_index = model(adapted_image.unsqueeze(0)).detach().cpu().numpy().argmax()
    result = jsonable_encoder(
        {
            'prediction': class_id_to_label(pred_index)
        }
    )
    return JSONResponse(content=result)
