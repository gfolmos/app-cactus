# Programa rag cataceas optimizado
# Muestra la inforamcion de cactus seleccionado
# Autor: Gerardo Figueroa
# Fecha: 17/06/26
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

# =====================================================================
# 1. CONFIGURACIÓN E INFRAESTRUCTURA SEGURA
# =====================================================================
st.set_page_config(layout="wide", page_title="Cactaceae RAG-Analyzer")

try:
    API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("Falta la credencial GROQ_API_KEY en los secretos de Streamlit.")
    st.stop()

IMAGE_DIR = "images"
CSV_FILE = "cactus_data.csv"

# =====================================================================
# 2. CAPA DE IA AUTOMATIZADA CON CACHÉ (Se ejecuta una sola vez)
# =====================================================================
@st.cache_resource
def inicializar_componentes_ia(ruta_csv: str):
    """Inicializa el LLM, genera los Embeddings e indexa TODO el CSV en Chroma."""
    # Inicializar LLM
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2, groq_api_key=API_KEY)
    
    # Inicializar Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Cargar y parsear todo el catálogo a Documentos de LangChain
    df = pd.read_csv(ruta_csv)
    documentos = []
    for _, row in df.iterrows():
        texto_contexto = f"Archivo: {row['archivo']}. Nombre Común: {row['nombre_comun']}. Detalles base: {row['detalles']}"
        doc = Document(page_content=texto_contexto, metadata={"source": row['archivo'], "nombre": row['nombre_comun']})
        documentos.append(doc)
    
    # Crear la base de datos vectorial en memoria con todo el catálogo
    vectorstore = Chroma.from_documents(documents=documentos, embedding=embeddings)
    
    return llm, vectorstore

# Validaciones de archivos de sistema antes de iniciar la IA
if not os.path.exists(IMAGE_DIR) or not os.path.exists(CSV_FILE):
    st.error("Faltan recursos esenciales (directorio de imágenes o archivo CSV).")
    st.stop()

# Instanciación única a través de la caché
llm_global, vectorstore_global = inicializar_componentes_ia(CSV_FILE)

# =====================================================================
# 3. INTERFAZ GRÁFICA Y LÓGICA DE USUARIO
# =====================================================================
def main():
    st.header("🌵 Analizador de Información Multimodal RAG")
    st.caption("Ecosistema inteligente para la generación de fichas botánicas")

    with st.expander("ℹ️ Explicación del Programa e Infraestructura"):
        st.write("Selecciona una muestra botánica para generar su ficha técnica automatizada.")

    fotos_cactus = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not fotos_cactus:
        st.warning("No se encontraron especímenes en la carpeta de imágenes.")
        return

    col_izq, col_der = st.columns([1, 2])

    with col_izq:
        st.subheader("Muestra Seleccionada")
        foto_seleccionada = st.selectbox("Selecciona un cactus de la lista:", sorted(fotos_cactus))
        st.image(os.path.join(IMAGE_DIR, foto_seleccionada), caption=f"ID: {foto_seleccionada}", width=280)

    with col_der:
        st.subheader("Análisis de la Enciclopedia de Cactáceas")
        
        # Filtro rápido para saber el nombre común antes de interrogar al RAG
        df_cactus = pd.read_csv(CSV_FILE)
        match_datos = df_cactus[df_cactus['archivo'] == foto_seleccionada]
        
        if match_datos.empty:
            st.info("ℹ️ Esta muestra de imagen no cuenta con metadatos asociados en el CSV.")
            return
            
        nombre_comun = match_datos.iloc[0]['nombre_comun']

        with st.spinner("Consultando base de conocimiento vectorial..."):
            try:
                # El Retriever ahora busca de forma inteligente sobre toda la base indexada
                retriever = vectorstore_global.as_retriever(
                    search_kwargs={"filter": {"source": foto_seleccionada}}
                )
                
                system_prompt = (
                    "Eres un bot experto en botánica y cactáceas.\n"
                    "A partir del siguiente contexto, genera una ficha informativa clara, breve y atractiva.\n"
                    "Debes incluir obligatoriamente las siguientes secciones usando negritas:\n"
                    "- **Nombre Común y/o Científico**\n"
                    "- **Características principales**\n"
                    "- **Región u Origen geográfico**\n\n"
                    "Si el contexto es muy corto, utiliza tus propios conocimientos amplios sobre cactáceas para complementar de forma verídica.\n"
                    "Responde siempre en Español.\n\n"
                    "Contexto:\n{context}"
                )
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "Genera la ficha informativa para el cactus asociado a: {input}"),
                ])
                
                # Construcción y ejecución de la cadena
                question_answer_chain = create_stuff_documents_chain(llm_global, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                
                resultado = rag_chain.invoke({"input": nombre_comun})
                
                st.success("¡Ficha técnica procesada!")
                st.markdown(resultado["answer"])
                
            except Exception as e:
                st.error(f"Error en el pipeline de IA: {e}")

if __name__ == "__main__":
    main()