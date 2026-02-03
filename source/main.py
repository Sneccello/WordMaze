import requests
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


def get_state() -> WordMazeState:
    if 'WordMazeState' not in st.session_state:
        st.session_state['WordMazeState'] = WordMazeState()

    return st.session_state['WordMazeState']

def extract_embeddings():
    zipfilename = EMBEDDING_FILE.replace('.txt', '.zip')
    with st.spinner(text="Unzipping embeddings...", ):
        logging.info('Unzipping embeddings...')
        with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
            internal_name = zip_ref.namelist()[0]
            with zip_ref.open(internal_name) as source, open(EMBEDDING_FILE, "wb") as target:
                target.write(source.read())


def download_embeddings():
    url = f"http://nlp.stanford.edu/data/{EMBEDDING_FILE.replace('.txt', '.zip')}"
    local_filename = EMBEDDING_FILE.replace('.txt', '.zip')

    with st.spinner("Downloading embeddings..."):

        progress_bar = st.progress(0)
        status_text = st.empty()

        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        downloaded_size = 0
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)

                    progress = downloaded_size / total_size
                    progress_bar.progress(progress)
                    status_text.text(f"Downloaded: {downloaded_size // 10 ** 6}MB / {total_size // 10 ** 6}MB")
        progress_bar.empty()


def setup():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S'
                        )
    st.set_page_config(layout="wide")
    st.title('WordMaze Playground :dolphin:')
    if not os.path.exists(EMBEDDING_FILE.replace('.txt', '.zip')):
        download_embeddings()
    if not os.path.exists(EMBEDDING_FILE):
        extract_embeddings()


def main():
    setup()
    build_page(get_state(), get_db())


if __name__ == '__main__':
    main()
