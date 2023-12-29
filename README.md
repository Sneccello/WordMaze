# WordMaze
## About
A simple app to visualize and enable playing around with word embeddings.
Word embeddings are vectors learnt by language models to represent words, expressions.
These vectors often have linear properties, so we can add and subtract different meanings.
For example, with if you take the France's representation vector, subtract Paris and add Tokyo, you get
a vector close to Japan (France-Paris+Tokyo≈Japan).

This app makes it easy to play around with embeddings, and see what other words you can build out of word vectors.


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
with the right sqlite3 library version. More info of this issue [here](https://discuss.streamlit.io/t/issues-with-chroma-and-sqlite/47950/3)