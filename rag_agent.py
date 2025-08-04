from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
#from src.agents.retriever import BookRetriever
from typing import TypedDict
import re
import os
from dotenv import load_dotenv
load_dotenv()  # Carga .env

# Definimos el estado que ira de agente en agente
class AgentState(TypedDict):
    input: str
    output: str
    context: str
    decision: str
    
class RAGAgent:
    def __init__(self):
        # Definimos el modelo LLM que usaremos
        self.llm = OllamaLLM(
            model=os.getenv("OLLAMA_MODEL"),
            base_url=os.getenv("OLLAMA_BASE_URL"),
            temperature=os.getenv("OLLAMA_TEMPERATURE")
        )
        
        # Definimos el retriever que usaremos indicando la coleccion en la que buscara
        #self.retriever = BookRetriever(collection_name="books") 
        
        # Definimos el prompt que le pasaremos a nuestro modelo LLM para que realize la tarea que se le solicita
        self.prompt = PromptTemplate.from_template("""
            Eres un experto y posees todos los conocimientos Magic The Gathering. Solo tienes conocimiento sobre este campo de conocimiento. No sabes nada mas de otros campos.
        
                                                   
            Las preguntas que te harán los usuarios seran acerca del juego Magic The Gathering. Aqui tienes la pregunta del usuario:
            {input}
                                                   
            Si la pregunta no tiene nada que ver con el juego, no podras responderla ya que no posees conocimientos en otros campos. Para responderla tendras que responder lo siguiente:
            "No puedo responder tu pregunta porque solo soy experto en Magic The Gathering"

            Debes responder en el mismo idioma en el que te haya preguntado el usuario.    

            Para responder las preguntas tienes disponible el chat de interaccion con el usuario:
            {chat_history}        

            El usuario tambien puede subir un archivo relacionado con el campo de conocimiento. En el caso de que lo haga tendras a continuación la transformación a texto del archivo:
            {uploaded_text}                          
            """)

    def search_response(self,  state) -> dict:
        # Obtiene el contexto formateado desde Milvus usando el retriever
        #context = self.retriever(state['input'])
        # Formateamos el historial del chat para que sea mas entendible
        chat_history_str = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" 
            for msg in state['chat_history']
        )
        #chat_history_str = re.sub(r'<[^>]+>', '', chat_history_str).strip()
        state['chat_history'] = chat_history_str 

        # Genera el texto del prompt
        prompt_text = self.prompt.format(input=state['input'],chat_history=state['chat_history'],uploaded_text=state['uploaded_text'])
        # Pasa el prompt al LLM y devuelve la respuesta
        response = self.llm.invoke(prompt_text)
        # Devolvemos la respuesta
        return {**state, "output": response}
 

if __name__ == "__main__":
    agent = RAGAgent()
    initial_state = {
            "input": "que colores existen",
            "output": "",
            "context": "",
            "decision": ""
        }
    respuesta = agent.search_response(initial_state)
    print(respuesta['output'])

