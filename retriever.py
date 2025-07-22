
from pymilvus import connections
from langchain_community.vectorstores import Milvus

from langchain_ollama import OllamaEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()  # Carga .env

class BaseRetriever:
    def __init__(self, collection_name: str):
        # Definimos la collecion en la que vamos a buscar
        self.collection_name = collection_name
        # Definimos el modelo de embedding que vamos a utilizar
        self.embeddings = OllamaEmbeddings(
                model=os.getenv("OLLAMA_MODEL"),
                base_url=os.getenv("OLLAMA_BASE_URL")
            )
        connections.connect(
            alias="default",
            host=os.getenv("MILVUS_HOST"),
            port=os.getenv("MILVUS_PORT")
        )
        # Cargar vectorstore existente desde Milvus
        self.vectorstore = Milvus(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    vector_field="embedding",
                    text_field="text"
        )
    
    def __call__(self, query: str) -> str:
        # Hacemos la busqueda del documento que sea mas similar a lo que pide el usuario
        docs = self.vectorstore.similarity_search(query,10)
        # Se llama al metodo de formato para formatear el contexto
        return self.format_context(docs)