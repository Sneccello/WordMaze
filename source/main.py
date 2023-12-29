import wget
import zipfile
import logging
import os.path

#ovverride default sqlite3 version for streamlit deployment
import sys
import pysqlite3
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import matplotlib
import streamlit as st

from consts import EMBEDDING_FILE
from db import WordDB
from page import build_page
from word_maze_state import WordMazeState

matplotlib.use('TkAgg')


@st.cache_resource
def get_db() -> WordDB:
    return WordDB()


@st.cache_resource
def get_state() -> WordMazeState:
    return WordMazeState()


def download_embeddings():
    with st.spinner(text="Downloading embeddings...(may take some minutes)", ):
        logging.info('Downloading embeddings...')
        wget.download(f"http://nlp.stanford.edu/data/{EMBEDDING_FILE.replace('.txt', '.zip')}")
    with st.spinner(text="Unzipping embeddings...", ):
        logging.info('Unzipping embeddings...')
        with zipfile.ZipFile(f"{EMBEDDING_FILE.replace('.txt', '.zip')}", 'r') as zip_ref:
            zip_ref.extractall()

def setup():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S'
                        )
    st.set_page_config(layout="wide")
    st.title('WordMaze Playground :dolphin:')
    if not os.path.exists(EMBEDDING_FILE):
        download_embeddings()


def main():
    setup()
    build_page(get_state(), get_db())


if __name__ == '__main__':
    main()
