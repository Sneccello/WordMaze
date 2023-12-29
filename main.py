import logging
import matplotlib
import streamlit as st

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


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S'
                        )
    st.set_page_config(layout="wide")
    st.title('WordMaze Playground')

    build_page(get_state(), get_db())


if __name__ == '__main__':
    main()
