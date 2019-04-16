import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle

BATCH_SIZE = 32
INPUT_TENSOR_NAME = 'inputs_input'

EPOCHS = 5
MAX_LEN = 75
EMBEDDING = 20

max_len = 75
max_len_char = 10


class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


if __name__ == '__main__':
    data = pd.read_csv("ner_dataset.csv", encoding="latin1")

    data = data.fillna(method="ffill")
    words = list(set(data["Word"].values))
    n_words = len(words)

    tags = list(set(data["Tag"].values))
    n_tags = len(tags)

    word2idx = {w: i + 2 for i, w in enumerate(words)}
    word2idx["UNK"] = 1
    word2idx["PAD"] = 0
    idx2word = {i: w for w, i in word2idx.items()}
    tag2idx = {t: i + 1 for i, t in enumerate(tags)}
    tag2idx["PAD"] = 0
    idx2tag = {i: w for w, i in tag2idx.items()}

    getter = SentenceGetter(data)

    sentences = getter.sentences

    print(sentences[0])
    print([[w[1] for w in s] for s in sentences][0])

    X_word = [[word2idx[w[0]] for w in s] for s in sentences]
    X_word = pad_sequences(maxlen=max_len, sequences=X_word, value=word2idx["PAD"], padding='post', truncating='post')
    chars = set([w_i for w in words for w_i in w])
    print(X_word[0:5])
    print(chars)
    n_chars = len(chars)
    print(n_chars)
    char2idx = {c: i + 2 for i, c in enumerate(chars)}
    char2idx["UNK"] = 1
    char2idx["PAD"] = 0
    X_char = []
    print(sentences[0], 'sents')
    for sentence in sentences:
        sent_seq = []
        for i in range(max_len):
            word_seq = []
            for j in range(max_len_char):
                try:
                    word_seq.append(char2idx.get(sentence[i][0][j]))
                except:
                    word_seq.append(char2idx.get("PAD"))
            sent_seq.append(word_seq)
        X_char.append(np.array(sent_seq))


    # with open('idx2tag.k', 'wb') as f:
    #     pickle.dump(idx2tag, f)
    # with open('word2idx.k', 'wb') as i:
    #     pickle.dump(word2idx, i)
    # with open('tag2idx.k', 'wb') as n:
    #     pickle.dump(tag2idx, n)
    # with open('char2idx.k', 'wb') as n:
    #     pickle.dump(char2idx, n)



    y = [[tag2idx[w[2]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=max_len, sequences=y, value=tag2idx["PAD"], padding='post', truncating='post')

    X_word_tr, X_word_te, y_tr, y_te = train_test_split(X_word, y, test_size=0.1, random_state=2018)
    X_char_tr, X_char_te, _, _ = train_test_split(X_char, y, test_size=0.1, random_state=2018)
    # X_tr, X_te, y_tr, y_te = np.array(X_char_tr), np.array(X_char_te), np.array(y_tr), np.array(y_te)

    np.save('x_char_train.npy', X_char_tr)
    np.save('x_word_train.npy', X_word_tr)

    np.save('x_char_test.npy', X_char_te)
    np.save('x_word_test.npy', X_word_te)

    np.save('y_train.npy', y_tr)
    np.save('y_test.npy', y_te)
