import os
import json
import pandas as pd
import traceback
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging

import streamlit as st
from langchain.callbacks import get_openai_callback # pip install -U langchain-community
from src.mcqgenerator.MCQGenerator import generate_evaluation_chain
from src.mcqgenerator.utils import read_file, get_table_data

with open(r'C:\Users\03123\OM\NLP\mcq_gen\Response.json', 'r') as file:
    RESPONSE_JSON = json.load(file)

st.title("Assessify")

with st.form("user_inputs"):
    uploaded_file = st.file_uploader("Upload a PDF or .txt file")
    mcq_count = st.number_input("No. of MCQs", min_value=3, max_value=10)
    subject = st.text_input("insert subject", max_chars=20)
    difficulty = st.text_input( "Complexity Level of questions", max_chars=20, placeholder="difficult, easy, etc...")
    button = st.form_submit_button("create MCQs")

    if button and uploaded_file is not None and mcq_count and subject and difficulty:
        with st.spinner("loading"):
            try:
                text = read_file(uploaded_file)
                with get_openai_callback() as cb:
                    response = generate_evaluation_chain(
                        {
                            "material": text,
                            "number": mcq_count,
                            "subject": subject,
                            "difficulty": difficulty,
                            "response_json": json.dumps(RESPONSE_JSON)
                        }
                    )
            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error("Error")
            else:
                print(f"Total Tokens:{cb.total_tokens}")
                print(f"Prompt Tokens:{cb.prompt_tokens}")
                print(f"Completion Tokens:{cb.completion_tokens}")
                print(f"Total Cost:{cb.total_cost}")

                if isinstance(response, dict):
                    quiz = response.get('quiz')
                    if quiz is not None:
                        table_data = get_table_data(quiz)
                        if table_data is not None:
                            df = pd.DataFrame(table_data)
                            df.index = df.index + 1
                            st.table(df)
                            st.text_area(label="Review", value=response["review"])
                        else:
                            st.error("Error in the table data")

                    else:
                        st.write(response)

