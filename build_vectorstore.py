from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import logging
import pickle

# ✅ Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("faiss").setLevel(logging.ERROR)

# ✅ Load environment variables
load_dotenv()

def load_documents():
    docs = []
    data_dir = "data"
    supported_extensions = ('.pdf', '.txt')

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ Data directory '{data_dir}' not found")

    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        try:
            if file.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file.endswith('.txt'):
                loader = TextLoader(file_path)
            else:
                continue

            raw_docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len,
                is_separator_regex=False
            )

            chunks = splitter.split_documents(raw_docs)

            for doc in chunks:
                doc.metadata.update({
                    "source": file,
                    "file_type": os.path.splitext(file)[1],
                    "chunk_length": len(doc.page_content.split())
                })

            docs.extend(chunks)
            logger.info(f"✅ Loaded {file} with {len(chunks)} chunks")

        except Exception as e:
            logger.error(f"❌ Failed to process {file}: {str(e)}")

    if not docs:
        raise ValueError("❌ No valid documents found after processing all files")

    logger.info(f"📚 Total documents loaded: {len(docs)}")
    return docs

def create_vectorstore():
    logger.info("🔄 Creating vectorstore...")

    docs = load_documents()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
    )

    db = FAISS.from_documents(docs, embeddings)
    save_path = "vectorstore"
    db.save_local(save_path)

    # ✅ Save the docstore.pkl explicitly (to match your app.py needs)
    with open(os.path.join(save_path, "docstore.pkl"), "wb") as f:
        pickle.dump(docs, f)

    logger.info(f"✅ Vectorstore and docstore saved with {db.index.ntotal} embeddings")

if __name__ == "__main__":
    try:
        create_vectorstore()
    except Exception as e:
        logger.error(f"❌ Fatal error in vectorstore creation: {str(e)}")
        raise
