from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
import argparse

def process_rag_langchain(file, user_input):
    print(f"Processing file: {file}")
    print(f"User input: {user_input}")
    # Load PDF
    loader = PyPDFLoader(file)
    docs = loader.load()
    print("doc len: ",len(docs))
    print("docs[0].page_content[0:100]: ",docs[0].page_content[0:100])
    print("docs[0].metadata: ",docs[0].metadata)
    
    # assing open_ai_key
    os.environ["OPENAI_API_KEY"] = "open_ai_key_here"
    
    # set llm model
    llm = ChatOpenAI(model="gpt-4o")
    
    # below code is to store loaded doc into vector store for easy access
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    #vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    # you can also use FAISS instead of Chroma
    vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    
    system_prompt = (
      "You are an assistant for question-answering tasks. "
      "Use the following pieces of retrieved context to answer "
      "the question. If you don't know the answer, say that you "
      "don't know. Use three sentences maximum and keep the "
      "answer concise."
      "\n\n"
      "{context}"
    )
    
    # give system_prompt below with input from user
    prompt = ChatPromptTemplate.from_messages(
      [
        ("system", system_prompt),
        ("human", "{input}"),
      ]
    )
    
    # assign llm and promt to the doc chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    # creating retrievel chanin with retriever and question_answer_chain
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # invoke rag chain with input from user
    results = rag_chain.invoke({"input": user_input})
    
    return results

if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Process a file and user input text using RAG: LangChain")
    
    # Add arguments for 'file' and 'input'
    parser.add_argument('--file', type=str, required=True, help="The file to process (e.g., PDF file)")
    parser.add_argument('--input', type=str, required=True, help="The user input prompt text")
    
    # Parse the arguments from the command line
    args = parser.parse_args()
    
    # Call the processing function with parsed arguments
    results = process_rag_langchain(file=args.file, user_input=args.input)
    
    # print results
    # look for "input" for langchain input and "answer" for answer in results dict
    print(results)
