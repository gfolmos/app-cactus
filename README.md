# 🌵 Cactácea RAG-Analyzer: Analizador de Información Botánica con IA
Cactaceae RAG-Analyzer es una aplicación web interactiva que combina una base de conocimiento vectorial e Inteligencia Artificial para generar fichas botánicas
detalladas y enriquecidas. Utilizando una arquitectura **RAG (Retrieval-Augmented Generation)** impulsada por LangChain y Groq, el sistema indexa dinámicamente metadatos
de imágenes y registros tabulares para proveer un análisis inmediato sobre taxonomías de cactus, morfología y orígenes geográficos.
---
## ✨ Características Principales

* **Arquitectura RAG Dinámica:** Indexación en tiempo real de registros tabulares estructurados convirtiéndolos en embeddings semánticos a través de `HuggingFaceEmbeddings`(`all-MiniLM-L6-v2`).
* **Base de Datos Vectorial Efímera:** Uso de `Chroma` en memoria para el almacenamiento y recuperación ágil del contexto exacto asociado a la especie seleccionada.
* **Orquestación Avanzada de LLM:** Integración de `ChatGroq` utilizando el modelo de alta velocidad `llama-3.3-70b-versatile`, configurado con baja temperatura ($0.2$)
  para evitar alucinaciones botánicas y asegurar veracidad técnica.
* **Expansión de Conocimiento Híbrido:** Instrucciones de prompts del sistema parametrizadas para fusionar el contexto local estructurado del archivo CSV con el conocimiento masivo del LLM.
* **Interfaz de Usuario Multimodal:** Entorno gráfico dividido en columnas nativas de Streamlit que permite previsualizar la fotografía física del espécimen a la par que se despliega la ficha sintética.
---

## 📂 Estructura del Proyecto

El repositorio cuenta con una distribución desacoplada ideal para el despliegue directo en plataformas cloud:

```text
├── /images/                # Directorio contenedor de las imágenes de cactus (.png, .jpg, .jpeg)
├── app.py                  # Script principal: Interfaz de usuario, Vector Store y Cadena RAG
├── cactus_data.csv         # Base de datos indexable: Relación de archivos, nombres y detalles base
└── requirements.txt        # Manifiesto de dependencias técnicas obligatorias del sistema
```
🛠️ Requisitos Previos e Instalación
Sigue este flujo de comandos para aislar el entorno operativo y desplegar la aplicación de manera local:
1. Clonar el repositorio
Bash
git clone [https://github.com/tu-usuario/tu-repositorio.git](https://github.com/tu-usuario/tu-repositorio.git)
cd tu-repositorio
2. Configurar el Entorno Virtual
Se recomienda utilizar un entorno virtual para mitigar conflictos de dependencias en las librerías de Machine Learning:
3. Instalacion de depandencias
Bash 
pip install -r requirements.txt

4. Inyección de Credenciales (Streamlit Secrets)
Para consumir el motor de inferencia de Groq, genera un archivo confidencial local de variables dentro de la ruta raíz:

Bash
mkdir .streamlit
touch .streamlit/secrets.toml
Edita el archivo .streamlit/secrets.toml y añade tu clave de acceso:

Ini, TOML
GROQ_API_KEY = "tu_groq_api_key_autenticada"

💻 Cómo Ejecutar la Aplicación
Para levantar el servidor web local y compilar los pipelines de LangChain, ejecuta:

Bash
streamlit run app.py
El aplicativo web se desplegará de forma automática en tu navegador de internet asignado por defecto bajo el puerto local indexado: http://localhost:8501.

⚠️ Aviso Importante: Al interactuar con APIs de modelos de lenguaje en nubes públicas, se desaconseja procesar documentación empresarial confidencial.
Para arquitecturas de producción bajo estrictas normativas de gobierno corporativo, este ecosistema de software está diseñado para migrarse a entornos 
On-Premise (Locales) mediante servidores de inferencia internos con herramientas como Ollama o vLLM, garantizando la privacidad absoluta del conocimiento del negocio.

