import itertools
import logging
import os
from typing import List

import chromadb
import numpy as np

from consts import NEGATING_PREFIX, EMBEDDING_FILE


class WordDB:
    WORD_COLLECTION = 'Words'
    DB_PATH = './db'
    BATCH_SIZE = 8192
    LOAD_N_BATCHES = 2

    def __init__(self):

        self.logger = logging.getLogger(self.__class__.__name__)
        os.makedirs(self.DB_PATH, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.DB_PATH)

        self.collection = self.client.get_or_create_collection(self.WORD_COLLECTION, metadata={"hnsw:space": "cosine"})
        self.EMBEDDING_FILE = EMBEDDING_FILE
        self._setup_db()

    def _get_line_count(self):
        with open(self.EMBEDDING_FILE, 'r') as file:
            line_count = sum(1 for _ in file)
        return line_count

    def _setup_db(self):
        goal_word_count = self._read_ideal_size()
        if self.collection.count() == goal_word_count:
            self.logger.info(
                f'DB word count checks out with {self.WORD_COLLECTION} collection of size {goal_word_count}.')
            return

        self.logger.info(
            f'Resetting inconsistent DB: there is {self.collection.count()} words in DB but there should be {goal_word_count}.')
        self.client.delete_collection(self.WORD_COLLECTION)
        self.collection = self.client.create_collection(name=self.WORD_COLLECTION)

        self._fill_db()

    def is_valid_word(self, word: str):
        if self.collection.get(ids=[word])['ids']:
            return True
        return False

    def _read_ideal_size(self):
        ideal_size = 0
        with open(self.EMBEDDING_FILE, "r") as f:
            for _ in range(self.BATCH_SIZE * self.LOAD_N_BATCHES):
                word = f.readline().split()[0]
                if word.isalnum():
                    ideal_size += 2  # itself plus negation
        return ideal_size

    def get_size(self):
        return self.collection.count()

    def get_embeddings(self, words: List[str]) -> List[np.ndarray]:

        res = self.collection.get(
            ids=words,
            include=['embeddings']
        )

        mapping = dict()
        for idx in range(len(res['ids'])):
            embed = res['embeddings'][idx]
            word = res['ids'][idx]
            mapping[word] = np.array(embed)

        return [mapping[word] for word in words]

    def _read_lines_in_batches(self):
        with open(self.EMBEDDING_FILE, 'r') as file:
            batch = []
            for line in file:
                batch.append(line.strip())
                if len(batch) == self.BATCH_SIZE:
                    yield batch
                    batch = []

            if batch:
                yield batch

    def _extract_word_embeddings(self, line: str):
        split_line = line.split()
        word = split_line[0]
        embedding = np.array(split_line[1:], dtype=np.float64)

        if word.isalnum():
            return [word, NEGATING_PREFIX + word], [embedding.tolist(), (-embedding).tolist()]

        return [], []

    def _fill_db(self):

        total_words = self._read_ideal_size()
        embeddings = []
        words = []
        inserted_words = 0
        for line_batch in itertools.islice(self._read_lines_in_batches(), self.LOAD_N_BATCHES):
            for line in line_batch:
                words_from_line, embeddings_from_line = self._extract_word_embeddings(line)
                words += words_from_line
                embeddings += embeddings_from_line

            self.collection.add(
                embeddings=embeddings,
                ids=words
            )
            self.logger.info(
                f'Inserted {len(embeddings)} words into DB. Currently at: {inserted_words} / {total_words}')
            embeddings = []
            words = []

        logging.info(f'Loading Done! Total inserted: {self.get_size()} words')
