import itertools

class Word2IdTransformer:
    """字转id"""

    def __init__(self):
        self.word2id = {}
        self.UNKNOW = 1 # pad is 0

    def fit(self, X):
        vocab = set(itertools.chain(*X))
        for i, w in enumerate(vocab, start=2):
            self.word2id[w] = i

    def transform(self, X):
        r = []
        for sample in X:
            s = []
            for w in sample:
                s.append(self.word2id.get(w, self.UNKNOW))
            r.append(s)
        return r

    def __len__(self):
        return len(self.word2id) + 1
