import numpy as np

from consts import NEGATING_PREFIX
from db import WordDB
from word_maze_state import WordMazeState


def is_negative_word(word: str) -> str:
    return word[0] == NEGATING_PREFIX


def get_positive_form(word: str) -> str:
    if is_negative_word(word):
        return word[1:]
    return word

def get_opposite_form(word: str) -> str:
    if is_negative_word(word):
        return get_positive_form(word)
    return NEGATING_PREFIX+word

def clean_input(word: str) -> str:
    word = word.lower()
    if word[0] == '+':
        return word[1:]
    return word

def get_guessed_vector_sum(state: WordMazeState, db: WordDB) -> np.ndarray:
    embeds = db.get_embeddings([get_positive_form(word) for word in state.words])
    for idx in range(len(state.words)):
        word = state.words[idx]
        if is_negative_word(word):
            embeds[idx] = -embeds[idx]

    curr_vec_sum = np.sum(embeds, axis=0)

    return curr_vec_sum

