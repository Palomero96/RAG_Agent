from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
#from src.agents.retriever import BookRetriever
from typing import TypedDict

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
            Eres un experto en reglas de Magic. Responde como tal
                                                   
            
            """)

    def search_response(self,  state) -> dict:
        # Obtiene el contexto formateado desde Milvus usando el retriever
        #context = self.retriever(state['input'])
        # Genera el texto del prompt
        #prompt_text = self.prompt.format(context=context, input=state['input'])
        # Pasa el prompt al LLM y devuelve la respuesta
        response = self.llm.invoke(input)
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
    print(agent.search_response(initial_state))