import openai

openai.api_key = "EMPTY"  # Not support yet
openai.base_url = "http://localhost:10101/v1"

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

chat = ChatOpenAI(model="vicuna-13b-v1.5",temperature=0)

template = "You are a helpful assistant that translates {input_language} to {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

chain = LLMChain(llm=chat, prompt=chat_prompt)
answer = chain.run(input_language="English", output_language="French", text="I love programming.")
print(answer)

# >> That's great! Programming can be a fun and rewarding hobby or career. What kind of programming do you enjoy the most? Do you have a favorite programming language or project you've worked on?
