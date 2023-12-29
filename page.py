import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from db import WordDB
from word_maze_state import WordMazeState
from word_utils import get_positive_form, clean_input, get_opposite_form, is_negative_word, get_guessed_vector_sum


def _query_top_hints(state: WordMazeState, db: WordDB, curr_vec_sum: np.array, top: int):
    goal_embed = db.get_embeddings([state.goal])[0]
    vector_diff_to_goal = goal_embed - np.array(curr_vec_sum)

    res = db.collection.query(
        query_embeddings=[vector_diff_to_goal.tolist()],
        n_results=top
    )

    return res['ids'][0]


def _build_header(state: WordMazeState, db: WordDB, notification_container):
    start_word_container, progress_bar_container, goal_word_container = st.columns([3, 5, 3])

    with start_word_container:
        st.markdown(f"<span style='color:lightblue; border: 2px solid lightblue; padding: 5px;border-radius: 10px;'>"
                    f"**Start: {state.start}**"
                    f"</span>", unsafe_allow_html=True)

        set_start_area_container, set_start_button_container = st.columns([2, 1])
        with set_start_area_container:
            set_start_text = st.text_input(label='Set', label_visibility='collapsed', key='text_start')
        set_start_error = False
        with set_start_button_container:
            set_button = st.button('Set', key='set_start')
            if set_button:
                set_start_text = set_start_text.lower()
                if set_start_text != '' and db.is_valid_word(set_start_text):
                    state.set_start(set_start_text)
                    st.rerun()
                else:
                    set_start_error = True
        with notification_container:
            if set_start_error:
                if set_start_text == '':
                    st.error('Cannot set empty text for start')
                else:
                    st.error(f'{set_start_text} is not in the database')

    with goal_word_container:
        st.markdown(f"<span style='color:lightblue; border: 2px solid lightblue; padding: 5px;border-radius: 10px;'>"
                    f"**Goal: {state.goal}**"
                    f"</span>", unsafe_allow_html=True)
        set_goal_area_container, set_goal_button_container = st.columns([2, 1])
        with set_goal_area_container:
            set_goal_text = st.text_input(label='Set', label_visibility='collapsed', key='text_goal')
        set_goal_error = False
        with set_goal_button_container:
            set_goal_text = set_goal_text.lower()
            set_button = st.button('Set', key='set_goal')
            if set_button:
                if set_goal_text != '' and db.is_valid_word(set_goal_text):
                    state.set_goal(set_goal_text)
                    st.rerun()
                else:
                    set_goal_error = True

        with notification_container:
            if set_goal_error:
                if set_goal_text == '':
                    st.error('Cannot set empty text for goal')
                else:
                    st.error(f'{set_goal_text} is not in the database')

    return progress_bar_container


def _handle_input(state: WordMazeState, db: WordDB, text_input: str):
    if text_input == '':
        return

    text_input = clean_input(text_input)
    positive_form = get_positive_form(text_input)
    opposite_form = get_opposite_form(text_input)
    if not db.is_valid_word(positive_form):
        st.error(f'{text_input} is not a valid word')
    elif text_input in state.words:
        st.error(f'{text_input} is already added')
    elif opposite_form in state.words and opposite_form != state.start:
        state.remove(opposite_form)
    elif opposite_form == state.start:
        st.error(f"You cannot remove starting word '{state.start}'")
    elif text_input == state.goal or opposite_form == state.goal:
        st.error(f"You cannot add or remove the goal word '{state.goal}'")
    else:
        state.add_word(text_input)


def _build_guess_container(state: WordMazeState, db: WordDB, guess_container):
    with guess_container:
        st.header('Guess Next')
        with st.form("input_word_form", clear_on_submit=True, border=True):
            text_input = st.text_input("Next word to add... (use '-' to negate. e.g.: '-man' for opposite vector of man)")

            submit_button_container, _, new_game_container = st.columns([2, 1, 2])

            with submit_button_container:
                submit_button = st.form_submit_button("Submit Word")

            with new_game_container:
                reset = st.form_submit_button('Reset')
                if reset:
                    state.clear()
            if submit_button:
                _handle_input(state, db, text_input)

            for word in state.words:
                color = 'lightcoral' if is_negative_word(word) else 'lightgreen'
                st.markdown(f"<span style='color:{color}'>**{word}**</span>", unsafe_allow_html=True)


def _build_rank_container(state: WordMazeState, db: WordDB, rank_container, curr_vec_sum: np.ndarray):
    with rank_container:
        st.header('Current neighbours')
        query_top_n = 5000
        closests = db.collection.query(query_embeddings=[curr_vec_sum.tolist()], n_results=query_top_n)
        try:

            for idx, close in enumerate(closests['ids'][0][:5]):
                st.write(f'{idx + 1}: {close}')

            goal_rank = closests['ids'][0].index(state.goal)
            last_digit = int(str(goal_rank)[-1])
            suffix = ['st', 'nd', 'rd'][last_digit] if 0 <= last_digit < 3 else 'th'
            st.markdown(
                f"<span style='color:rgb(220,182,210)'>"
                f"**'{state.goal}' is currently the <u>{goal_rank + 1}{suffix}</u> "
                f"closest embedding to your combined guess**"
                f"</span>",
                unsafe_allow_html=True)

        except ValueError:
            st.write(f'{state.goal} is not among the closest {query_top_n} embeddings')
            goal_rank = db.get_size()

    return goal_rank


def _fill_progress_bar(state: WordMazeState, db: WordDB, progress_bar_container, goal_rank) -> int:
    with progress_bar_container:
        progress = 1 - (goal_rank / db.get_size()) ** (1 / 3)
        progress_rounded = int(round(progress, 4) * 100)
        st.progress(progress_rounded, text=f'\t~{round(progress * 100, 1)}%')

        equation = f"{state.goal} = {state.start}"
        template = ":{color}[{sign}{word}]"

        for word in state.get_guessed_words():
            if is_negative_word(word):
                equation += template.format(color='red', sign='-', word=get_positive_form(word))
            else:
                equation += template.format(color='green', sign='+', word=word)
        st.markdown(equation)

    return progress


def _plot_embeddings(state: WordMazeState, db: WordDB, curr_vec_sum: np.ndarray):
    with st.container(border=True):
        reducer = PCA()

        embeds = db.collection.get(
            ids=state.words + [state.goal],
            include=['embeddings']
        )
        scaler = StandardScaler()

        scaled_embeddings = scaler.fit_transform(embeds['embeddings'] + [curr_vec_sum])
        reduced_embeds = reducer.fit_transform(scaled_embeddings)

        df = pd.DataFrame({
            'Reduced Dimension 1': reduced_embeds[:, 0],
            'Reduced Dimension 2': reduced_embeds[:, 1],
            'Labels': embeds['ids'] + ['Current Position']
        })

        fig = px.scatter(
            df,
            'Reduced Dimension 1',
            'Reduced Dimension 2',
            color='Labels',
            hover_data={'Labels': True},
            title='Visualization of Embeddings Projected to 2D (Interactive)'
        )

        fig.update_layout(xaxis_title="Reduced Dimension 1", yaxis_title="Reduced Dimension 2")

        st.plotly_chart(fig)


def _build_info_container(info_container):
    with info_container:
        st.subheader('About')
        text = \
            "Large Language Models are trained on huge chunks text and learn useful vector representations, embeddings, for words and" \
            " expressions. " \
            "These vector representations often have remarkable linear properties, making it possible to add and substract ~meanings. " \
            "One classic example is\n\n$$\small Queen = King - Man + Woman$$\n\nThat is, if you take the learnt King vector, subtract Man and add Woman," \
            " the resulting vector will be close to Queen. This is a small tool for exploration of similar examples. Have fun!"
        st.markdown(text)


def build_page(state: WordMazeState, db: WordDB):
    notification_container = st.container()

    progress_bar_container = _build_header(state, db, notification_container)
    st.markdown("""---""")

    info_container, guess_container, rank_container = st.columns([2, 5, 2])
    _build_info_container(info_container)
    _build_guess_container(state, db, guess_container)

    curr_vec_sum = get_guessed_vector_sum(state, db)

    _plot_embeddings(state, db, curr_vec_sum)

    goal_rank = _build_rank_container(state, db, rank_container, curr_vec_sum)

    progress = _fill_progress_bar(state, db, progress_bar_container, goal_rank)

    if progress == 1:
        with notification_container:
            st.success('Success!')
