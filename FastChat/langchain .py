from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.embeddings import OpenAIEmbeddings
import numpy as np
import openai
import os

template = """{history}
Human: {human_input}
Assistant:"""
openai.api_key = "EMPTY"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.base_url = "http://localhost:11001/v1/"


def test_embedding():
    embeddings = OpenAIEmbeddings()
    texts = ["Why does the chicken cross the road", "To be honest", "Long time ago"]
    query_result = embeddings.embed_query(texts[0])
    doc_result = embeddings.embed_documents(texts)
    assert np.allclose(query_result, doc_result[0], atol=1e-3)

def test_chain():

    prompt = PromptTemplate(
        input_variables=["history", "human_input"],
        template=template
    )
    chain = LLMChain(
        llm=OpenAI(model="text-embedding-ada-002", temperature=1), 
        prompt=prompt, 
        verbose=True, 
        memory=ConversationBufferWindowMemory(k=2),
    )
    output = chain.predict(human_input="你好")
    print(output)

if __name__ == "__main__":
    test_embedding()
    test_chain()