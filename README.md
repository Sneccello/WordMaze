# WordMaze: A Vector Embeddings Playground
## About
A simple app to visualize and enable playing around with word embeddings.
Word embeddings are vectors learned by language models to represent words and expressions.
These vectors often have linear properties, so we can add and subtract different meanings.
For example, if you take France's representation vector, subtract Paris, and add Tokyo, you get
a vector close to Japan (France-Paris+Tokyo≈Japan).
More details can be read here in these papers: [Word2Vec](https://arxiv.org/abs/1301.3781) , [GloVe](https://nlp.stanford.edu/pubs/glove.pdf)

This app makes it easy to play around with embeddings and see what other words you can build from word vectors.

<img width="1319" alt="Screenshot" src="https://github.com/Sneccello/WordMaze/assets/78796219/73defe10-8308-4c27-a64e-17c289e0bd7e">

You can also see relationships between vectors projected down to 2D:
<img width="829" alt="Screenshot 2023-12-29 at 23 29 38" src="https://github.com/Sneccello/WordMaze/assets/78796219/6a04071c-4231-4acc-9267-aa32e164fd3c">
France-Paris and Japan-Tokyo have very similar relationships in the vector space. 
## Running it locally

1. clone the repo

    ```git clone https://github.com/Sneccello/WordMaze.git```
2. setup environment

    ```conda create --name word_maze python=3.11```

    ```conda activate word_maze```

    ```pip install -r requirements.txt```

3. Run the streamlit app (it may take some time to install the [GloVe](https://github.com/stanfordnlp/GloVe) 
vector embeddings for the first time)

    ```streamlit run source/main.py```

## Troubleshoot
- pip install may fail for M1(+) Macs on pysqlite3-binary. This requirement is only meant for streamlit deployment 
where the default sqlite3 library version does not match the required chromadb version. You can remove this requirement
with the right sqlite3 library version. More info on this issue [here](https://discuss.streamlit.io/t/issues-with-chroma-and-sqlite/47950/3)
