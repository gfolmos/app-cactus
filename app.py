# Proyecto cactus
# Muestra la inforamcion de cactus seleccionado
# Autor: Gerardo Figueroa
# Fecha: 15/06/26
import streamlit as st
import pandas as pd
import os
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# 1. Configuración de la página
st.set_page_config(layout="wide")

API_KEY = st.secrets["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = API_KEY

# Inicializar el modelo de Groq
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2)
fotos_cactus = "img_1.jpg"

# --- Interfaz de usuario ---
st.header("🌵 Analizador de Información Multimodal RAG (Retrieval-Augmented Generation)")
st.write("🚀 Utiliza un agente de IA que es realmente sorprendente!")

# Explicación del programa
with st.expander("Explicación del Programa"):
    st.write("""
        Ahora es posible buscar y analizar información con ayuda de la IA, herramientas como esta, utilizando fotos, bases de datos \n 
        y archivos, podemos realizar cualquier tipo de aplicación para que nuestro trabajo sea mas eficiente y productivo.      
        Nota Final: \n
        Al utilizar la IA por seguridad no se recomendaría analizar los documentos confidenciales.
        Para poder utilizar una herramienta IA como esta corriendo por internet, se puede correr localmente (on-premise)
        en una computadora moderna para que la información no salga por internet y esté segura la información. \n
        """)
#st.write("Selecciona la foto de un cactus para que nuestra IA te de sus detalles, características y región.")

# Directorio de imágenes
IMAGE_DIR = "images"
CSV_FILE = "cactus_data.csv"

# Verificar que existan las carpetas y archivos necesarios
if not os.path.exists(IMAGE_DIR):
    st.error(f"No se encontró la carpeta '{IMAGE_DIR}'. Por favor créala y sube tus fotos ahí.")
    st.stop()

if not os.path.exists(CSV_FILE):
    st.error(f"No se encontró el archivo '{CSV_FILE}' con los datos de los cactus.")
    st.stop()

# Leer imágenes disponibles en la carpeta
fotos_cactus = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not fotos_cactus:
    st.warning("No se encontraron imágenes en la carpeta 'images/'.")
else:
    # Cargar base de datos base desde el CSV
    df_cactus = pd.read_csv(CSV_FILE)

    # Crear dos columnas: Izquierda para selección/previsualización, Derecha para los resultados de la IA
    col_izq, col_der = st.columns([1, 2])

    with col_izq:
        st.subheader("Selección de Especie")
        foto_seleccionada = st.selectbox("Selecciona una foto de cactus:", sorted(fotos_cactus))
        
        # Mostrar el preview pequeño de la foto fija
        ruta_foto = os.path.join(IMAGE_DIR, foto_seleccionada)
        st.image(ruta_foto, caption=f"Vista previa: {foto_seleccionada}", width=250)

    with col_der:
        st.subheader("Análisis e Información del Cactus")
        
        # Buscar si el archivo seleccionado existe en nuestro CSV de datos
        datos_encontrados = df_cactus[df_cactus['archivo'] == foto_seleccionada]
        
        if datos_encontrados.empty:
            st.info("ℹ️ Esta imagen no tiene información.")
        else:
            # Obtener el texto descriptivo básico del CSV para indexarlo en el RAG
            row = datos_encontrados.iloc[0]
            texto_contexto = f"Archivo: {row['archivo']}. Nombre Común: {row['nombre_comun']}. Detalles base: {row['detalles']}"
            
            with st.spinner("La IA está consultando la enciclopedia de cactáceas..."):
                try:
                    # Crear un Documento dinámico para pasarle a Chroma
                    documento = Document(page_content=texto_contexto, metadata={"source": foto_seleccionada})
                    
                    # Inicializar embeddings y Base de datos Vectorial en memoria
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    vectorstore = Chroma.from_documents(documents=[documento], embedding=embeddings)
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
                    
                    # Prompt del sistema para que Llama expanda la información de manera elegante
                    system_prompt = (
                        "Eres un bot experto en botánica y cactáceas.\n"
                        "A partir del siguiente contexto, genera una ficha informativa clara, breve y atractiva.\n"
                        "Debes incluir obligatoriamente las siguientes secciones usando negritas:\n"
                        "- **Nombre Común y/o Científico**\n"
                        "- **Características principales**\n"
                        "- **Región u Origen geográfico**\n\n"
                        "Si el contexto es muy corto, utiliza tus propios conocimientos amplios sobre cactáceas para complementar la respuesta de forma verídica.\n"
                        "Responde siempre en Español.\n\n"
                        "Contexto:\n{context}"
                    )
                    
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "Genera la ficha informativa para el cactus asociado a: {input}"),
                    ])
                    
                    # Crear la cadena RAG
                    question_answer_chain = create_stuff_documents_chain(llm, prompt)
                    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                    
                    # Invocar al modelo pasando el nombre del cactus encontrado
                    resultado = rag_chain.invoke({"input": row['nombre_comun']})
                    
                    # Desplegar la información en un recuadro verde limpio
                    st.success("¡Ficha generada con éxito!")
                    st.markdown(resultado["answer"])
                    
                except Exception as e:
                    st.error(f"Hubo un inconveniente al procesar los datos de la IA: {e}")