import logging


class WordMazeState:

    def __init__(self, start: str = 'france', goal: str = 'japan'):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.start = start
        self.goal = goal
        self.words = [start]

    def add_word(self, word: str):
        self.words.append(word)


    def clear(self):
        self.words = [self.start]

    def undo(self):
        if len(self.words) == 1:
            return

        self.words.pop()

    def set_start(self, start: str):
        self.start = start
        self.clear()

    def set_goal(self, goal: str):
        self.goal = goal
        self.clear()

    def remove(self, word: str):
        self.words.remove(word)

    def get_guessed_words(self):
        return self.words[1:]
