import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_milvus import Milvus
from dotenv import load_dotenv

# Cargar variables del .env
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
model_kwargs = {"device": "cpu", "trust_remote_code": True}

# Configuración
PATH_PDFS = "data/"
COLLECTION_NAME = 'data'



def pdfLoader():
    #### Initialize Our Documents
    documents = []
    ## For each document
    for file in os.listdir(PATH_PDFS):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(PATH_PDFS, file)
            print(pdf_path)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v2-base-de",  model_kwargs=model_kwargs)


    vector_store = Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={"host": os.getenv("MILVUS_HOST"), "port": os.getenv("MILVUS_PORT")},
        auto_id=True,  # Permite que Milvus asigne automáticamente los IDs
    )
    # Insertar en Milvus
    print(f"Inserting {len(docs)} documents into Milvus...")
    vector_store.add_documents(docs)

    print("Inserción completada.")
    


#def processPdf(path):





if __name__=='__main__':
    
    #Cargamos los datos de los libros
    pdfLoader(pdf_path)




# Procesar el PDF
print(f"Procesando: {os.path.basename(pdf_path)}")
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()

# Dividir en chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(pages)

# Patrón para detectar encabezados de capítulo
chapter_pattern = re.compile(r'(Capítulo\s+\d+\.?\s+[^\n]*)', re.IGNORECASE)

# Añadir metadatos
for doc in docs:
    doc.metadata['filename'] = os.path.basename(pdf_path)
    doc.metadata['page_number'] = doc.metadata.get('page', None)

    # Buscar encabezado de capítulo en el contenido
    match = chapter_pattern.search(doc.page_content)
    if match:
        doc.metadata['chapter'] = match.group(1).strip()
    else:
        doc.metadata['chapter'] = 'Sin capítulo detectado'

# Insertar en Milvus
print(f"Inserting {len(docs)} documents into Milvus...")
vector_store.add_documents(docs)

print("Inserción completada.")
