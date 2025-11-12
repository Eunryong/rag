from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
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

    llm = OllamaLLM(model="llama3")

    # 프롬프트 템플릿 생성
    template = """다음 문맥을 사용하여 질문에 답하세요:

{context}

질문: {question}
답변:"""

    prompt = ChatPromptTemplate.from_template(template)

    # LCEL 방식으로 체인 구성
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    query = "다크서클이 생기는 이유는?"
    result = qa_chain.invoke(query)
    print("답변:", result)

    print("\n참고 문서:")
    retrieved_docs = retriever.invoke(query)
    for doc in retrieved_docs:
        print(doc.page_content)
    # results = vectorstore.similarity_search(query, k=3)

if __name__ == "__main__":
    main()