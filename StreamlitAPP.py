import os
import json
import pandas as pd
import traceback
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging

import streamlit as st
from langchain.callbacks import get_openai_callback
from src.mcqgenerator.utils import read_file, get_table_data

