from langchain_community.llms import OpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import gradio as gr
import os
import time
os.environ["OPENAI_API_KEY"] = 'EMPTY'
os.environ["OPENAI_API_BASE"] = "http://server-llm-dev:8000/v1"


def load_llmmodel():
    model_name = "ChatGLM3-6B"
    llm = ChatOpenAI(model_name=model_name)
    return llm


def load_docs(directory):
    loader = DirectoryLoader(directory, glob='**/*.*',
                             show_progress=True)
    documents = loader.load()
    return documents


def split_docs(documents):
    text_splitter = CharacterTextSplitter(chunk_size=150,
                                          chunk_overlap=20)
    split_docs = text_splitter.split_documents(documents)
    return split_docs


def create_vectorstore(split_docs):
    embeddings = SentenceTransformerEmbeddings(
        model_name="./dataroot/models/shibing624/text2vec-base-chinese")
    vectorstore = Chroma.from_documents(split_docs, embeddings)
    return vectorstore


def search_docs(vectorstore, query):
    matching_docs = vectorstore.similarity_search(query)
    return matching_docs


def answer_fromchain(llm, matching_docs, query):
    chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
    answer = chain.run(input_documents=matching_docs, question=query)
    return answer


def chat_webui(llm):
    with gr.Blocks() as blocks:
        gr.HTML("""<h1 align="center">RAG演示</h1>""")
        chatbot = gr.Chatbot()
        msg = gr.Textbox(show_label=False,
                         placeholder="请输入问题...", container=False)
        clear = gr.Button("清除问题")

        def messages(user_message, history):
            return "", history + [[user_message, None]]

        def bot(history):
            query = history[-1][0]
            matching_docs = search_docs(vectorstore, query)
            history[-1][1] = ""
            bot_message = "知识库检索结果：\n" + \
                matching_docs[0].page_content + "[" + \
                matching_docs[0].metadata["source"] + "]"
            for character in bot_message:
                history[-1][1] += character
                time.sleep(0.05)
                yield history
            answer = answer_fromchain(llm, matching_docs, query)
            bot_message = "\nLLM生成结果：\n" + answer
            for character in bot_message:
                history[-1][1] += character
                time.sleep(0.05)
                yield history
        msg.submit(messages, [msg, chatbot],
                   [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        clear.click(lambda: None, None, chatbot, queue=False)
    blocks.queue()
    blocks.launch()


if __name__ == "__main__":
    llm = load_llmmodel()
    documents = load_docs("./documents")
    split_docs = split_docs(documents)
    vectorstore = create_vectorstore(split_docs)
    chat_webui(llm)
