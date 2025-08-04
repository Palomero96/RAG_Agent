import streamlit as st
import os, sys
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd

# Configuración de la página
st.set_page_config(
    layout="wide", 
    page_title="Magic Agent", 
    page_icon="🃏",
    initial_sidebar_state="collapsed"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
        .main {padding-top: 1rem;}
        .header {border-bottom: 1px solid #ddd; margin-bottom: 1.5rem;}
        .chat-container {max-height: 65vh; overflow-y: auto; padding-right: 0.5rem;}
        .file-preview {background-color: #f8f9fa; border-radius: 0.5rem; padding: 1rem;}
        .stSpinner > div {text-align: center;}
        .stChatInput {position: fixed; bottom: 2rem; width: 83%;}
    </style>
""", unsafe_allow_html=True)

# Inicializamos el Agente RAG
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag_agent import RAGAgent
RAG_AGENT = RAGAgent()

# Titulo y descripcion
with st.container():
        st.markdown('<div class="header">', unsafe_allow_html=True)
        st.title("Magic Agent")
        st.markdown("**Haz cualquier pregunta dentro del campo de conocimiento de Magic The Gathering.**")
        st.markdown('</div>', unsafe_allow_html=True)

# Apartado para subir archivos
with st.sidebar:
    # Definimos el titulo del apartado de archivos 
    st.subheader("📂 Documentos de referencia")
    uploaded_file = st.file_uploader(
        "Sube reglas, listas de cartas o artículos",
        type=["pdf", "txt", "docx", "xlsx"],
        help="Puedes subir documentos para enriquecer las respuestas"
    )
    
    if uploaded_file:
        with st.spinner("Procesando archivo..."):
            def extract_text_from_file(uploaded_file):
                text = ""
                # Diferentes transformaciones en función del archivo que suba el usuario
                try:
                    if uploaded_file.type == "application/pdf":
                        reader = PdfReader(uploaded_file)
                        text = "\n".join([page.extract_text() for page in reader.pages])
                    elif uploaded_file.type == "text/plain":
                        text = uploaded_file.getvalue().decode("utf-8")
                    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        doc = Document(uploaded_file)
                        text = "\n".join([para.text for para in doc.paragraphs])
                    elif "spreadsheet" in uploaded_file.type:
                        df = pd.read_excel(uploaded_file)
                        text = df.to_string()
                except Exception as e:
                    st.error(f"Error al procesar: {str(e)}")
                return text
            
            uploaded_text = extract_text_from_file(uploaded_file)
            st.success("Documento listo para consulta")
            with st.expander("📝 Vista previa"):
                st.text(uploaded_text[:500] + "...")

# Apartado del chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Contenedor del chat con scroll
with st.container():
    st.subheader("💬 Chat con el experto")
    chat_placeholder = st.empty()
    
    with chat_placeholder.container():
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)

# Input del usuario
if prompt := st.chat_input("Escribe tu pregunta sobre Magic..."):
    # Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(f"**{prompt}**")
    
    # Procesar con RAG
    with st.spinner("🔍 Buscando en el conocimiento de Magic..."):
        state = {
            "input": prompt,
            "chat_history": st.session_state.chat_history,
            "uploaded_text": uploaded_text if 'uploaded_text' in locals() else ""
        }
        result = RAG_AGENT.search_response(state)
        full_response = result["output"]
    
    # Mostrar respuesta
    with st.chat_message("assistant", avatar="🃏"):
        st.markdown(full_response)
    
    # Actualizar historial
    st.session_state.chat_history.extend([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": full_response}
    ])
    
    # Auto-scroll al final del chat
    st.markdown("""
        <script>
            window.scrollTo(0, document.body.scrollHeight);
        </script>
    """, unsafe_allow_html=True)