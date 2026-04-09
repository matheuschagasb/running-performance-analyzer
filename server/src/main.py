from fastapi import FastAPI, File, UploadFile, HTTPException
import shutil
import os

app = FastAPI()

if not os.path.exists("videos"):
    os.makedirs("videos")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/uploadvideo/")
async def create_upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.mp4'):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .mp4 files are allowed.")

    file_location = f"videos/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    return {"info": f"File '{file.filename}' saved at '{file_location}'"}
