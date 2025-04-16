import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import tempfile

# Configurar la página
st.set_page_config(page_title="Asistente de CV", page_icon="📄")
st.title("📄 Chat con tu CV usando Groq")

# Mostrar nombre de archivo subido
uploaded_file = st.file_uploader("Subí tu CV en formato PDF", type=["pdf"])

# Inicializar modelo de embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Ruta donde se guarda la base vectorial
persist_dir = "./chroma_db"

# Si ya existe una base, cargarla
vectorstore = None
if os.path.exists(persist_dir):
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)

# Si subís un nuevo PDF, lo procesamos e indexamos (sin pisar lo anterior)
if uploaded_file is not None:
    filename = uploaded_file.name
    st.success(f"📎 Archivo cargado: {filename}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        loader = PyPDFLoader(tmp.name)
        new_docs = loader.load()

    if vectorstore:
        vectorstore.add_documents(new_docs)
    else:
        vectorstore = Chroma.from_documents(new_docs, embedding=embedding_model, persist_directory=persist_dir)
    vectorstore.persist()

# Si ya hay base, mostrar campo de preguntas
if vectorstore:
    st.subheader("✍️ Escribí tu consulta sobre el contenido del PDF")
    user_question = st.text_input("Pregunta:")

    if user_question:
        # Recuperar contexto relevante
        docs = vectorstore.similarity_search(user_question, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        # Preparar prompt
        prompt = ChatPromptTemplate.from_template("""
        Usá el siguiente contexto para responder la pregunta de forma clara y precisa:

        Contexto:
        {context}

        Pregunta:
        {question}

        Respuesta:
        """)

        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-8b-8192"
        )

        chain = LLMChain(llm=llm, prompt=prompt)

        # Ejecutar consulta
        respuesta = chain.run({"context": context, "question": user_question})
        st.markdown("### 🤖 Respuesta:")
        st.write(respuesta)
else:
    st.info("Subí un archivo PDF para comenzar.")
