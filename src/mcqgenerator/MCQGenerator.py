import os
import json
import pandas as pd
import traceback
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

# load environment varaibles from .env file
load_dotenv()

# Access open api key
openai_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(api_key = openai_key, model_name="gpt-3.5-turbo", temperature=0.7)



TEMPLATE = """

Material: {material}

You are a knowledgable professor, who is an expert at creating multiple choice questions. Given the above material, it is your job to \
create a multiple choice quiz of {number} multiple choice questions for students of {subject}. Set the difficulty of the exam to be {difficulty}.
Make sure the questions are not repeated and check to make sure that students can get the answer to each of your questions using the above text, and \
the above text alone.

Make sure to format your response like RESPONSE_JSON below, and use it as a guide for your output.

Ensure there are {number} Multiple Choice Questions.

### Response JSON below:
{response_json}

"""

quiz_gen_prompt = PromptTemplate(
    input_variables=["material", "number", "subject", "difficulty", "response_json"],
    template=TEMPLATE
)

quiz_chain = LLMChain(llm=llm, prompt=quiz_gen_prompt, output_key="quiz", verbose=True)

TEMPLATE_EVAL = """
You are an expert in English grammar and writing. You've been tasked with evaluating a Multiple Choice Quiz for {subject} students. \
Your analysis should include a concise assessment of its complexity, using a maximum of 50 words. \
If the quiz doesn't match the students' cognitive and analytical abilities, modify questions accordingly, adjusting the tone to suit their level.

Quiz_MCQs:
{quiz}

Seek evaluation from an expert English writer for the quiz above:
"""

quiz_eval_prompt = PromptTemplate(
    input_variables=["subject", "quiz"],
    template=TEMPLATE_EVAL
)

review_chain = LLMChain(llm=llm, prompt=quiz_eval_prompt, output_key="review", verbose=True)


generate_evaluation_chain = SequentialChain(chains=[quiz_chain, review_chain], input_variables=["material", "number", "subject", "difficulty", "response_json"], 
                                            output_variables = ["quiz", "review"], verbose=True)
