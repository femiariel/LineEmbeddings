from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import json
from io import BytesIO
#from src.services.clustering import *
import pandas as pd # dataframe manipulation
import numpy as np # linear algebra
from sentence_transformers import SentenceTransformer
#from openai import OpenAI
import os
from pyod.models.ecod import ECOD
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import torch


app = FastAPI()

 
#crée une liste contenant la valeur de chaque ligne pour chaque colonne sous la forme "col 1": "line 1", "col 2":"line 1" , ainsi de suite
def compile_text_generic(df, row_index):
    row = df.iloc[row_index]
    text_parts = []
    for column in row.index:
        text_parts.append(f"{column.capitalize()}: {row[column]}")
    return ",\n".join(text_parts)
 
 
model = SentenceTransformer("./bge-m3")
device = "cuda" if torch.cuda.is_available() else "cpu"
def embeddings_bge(text):
    text = text.replace("\n", " ")
    embeddings = model.encode(text,device=device,batch_size=100)
    return embeddings
 
def get_embedding_from_inputfile(file_path):
    #openai_api_key= os.getenv("openai_api_key")
    file_like_object = BytesIO(file_path)
    df = pd.read_csv(file_like_object, sep = ";")
    # Appliquer la fonction à toutes les lignes du DataFrame
    texts = [compile_text_generic(df, i) for i in range(len(df))]
    embeddings=[embeddings_bge(text) for text in texts[:100]]
    df_embedding = pd.DataFrame(embeddings)
    return df_embedding
 


@app.get("/")
def read_root():
    return {"message": "Welcome to the embedding"}
 
@app.post("/embedding")
async def clust(file: UploadFile = File(...)):
    content = await file.read()
    # Assurez-vous que cette ligne soit adaptée pour correspondre à la logique d'entrée de votre fonction customer_segmentation
    df = get_embedding_from_inputfile(content)  # Adaptez cette ligne selon votre implémentation
 
    # Convertir le DataFrame en format JSON
    result = df.to_json(orient="records", date_format="iso", force_ascii=False)
    parsed_result = json.loads(result)  # Convertit la chaîne JSON en objet Python pour une meilleure compatibilité avec JSONResponse
    return JSONResponse(content=parsed_result)
 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)