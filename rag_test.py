from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import re
import json
load_dotenv()


def clean_text(text):
    """텍스트 공백 정리"""
    text = re.sub(r' {2,}', ' ', text)
    
    text = text.replace('\t', ' ')
    
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    lines = [line for line in text.split('\n') if line]
    text = '\n'.join(lines)
    
    return text.strip()


def load_documents():
    loader = PyPDFLoader("sample.pdf")
    documents = loader.load()
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
    return documents


def chunk_doc(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True 
    )
    splits = text_splitter.split_documents(documents)
    return splits


def save_documents(documents, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, page in enumerate(documents):
            data = {
                'page': i + 1,
                'text': page.page_content,
                'metadata': {
                    'source': 'sample.pdf',
                    'page': page.metadata.get('page', i)
                }
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


def main():
    text = input("무엇을 도와드릴까요? : ")
    documents = load_documents()
    chunked_documents = chunk_doc(documents)
    save_documents(chunked_documents, 'tmp.jsonl')
    embeddings = OllamaEmbeddings(
        model="llama3",
    )
    vectorstore = Chroma.from_documents(
        documents=chunked_documents,
        embedding=embeddings,
        persist_directory="./chroma_db"  # 저장 경로
    )
    # embeddings.embed_query(text)

    retriever = vectorstore.as_retriever(
        search_type="similarity",  # 또는 "mmr"
        search_kwargs={"k": 3}  # 상위 3개 문서 검색
    )

    retrieved_documents = retriever.invoke("Why do people have dark circles?")

    print(retrieved_documents[0].page_content)

if __name__ == "__main__":
    main()