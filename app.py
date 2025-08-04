import streamlit as st
import os, sys


# Añade el directorio src al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag_agent import RAGAgent

# Atributos de la pagina
st.set_page_config(layout="wide", page_title="Magic Agent", page_icon="🃏")

# Título llamativo
st.title("Magic Agent")
st.markdown("Haz cualquier pregunta dentro del campo de conocimiento de Magic The Gathering.")

RAG_AGENT = RAGAgent()

# Se inicializa el historial de mensajes
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Se muestra todo el historial de mensajes
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# Interaccion entre usuario y asistente
if prompt := st.chat_input("Escribe tu consulta aquí..."):
    # Se muestra el mensaje del usuario 
    with st.chat_message("user"):
        st.markdown(f"<b>{prompt}</b>", unsafe_allow_html=True)
    # Se guarda el mensaje del usuario en el historial
    st.session_state.chat_history.append({"role": "user", "content": f"<b>{prompt}</b>"})

    # Mostramos un mensaje mientras se ejecuta un grafo
    with st.spinner("🤔 Estoy pensando..."):
        result = RAG_AGENT.search_response({"input": prompt,"chat_history": st.session_state.chat_history})
        full_response = result["output"]

    # Se muestra la respuesta del asistente
    with st.chat_message("assistant"):
        st.markdown(full_response, unsafe_allow_html=True)

    # Se guarda el mensaje del asistente en el historial
    st.session_state.chat_history.append({"role": "assistant", "content": full_response})