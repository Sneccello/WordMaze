class WordMazeState:

    def __init__(self):
        self.start = 'dog'
        self.goal = 'whale'
        self.words = [self.start, '-cat', 'huge', 'dolphin']

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
