from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

class CodeTokenizer:
    def __init__(self, num_words=10000, oov_token="<OOV>"):
        self.tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
        self.num_words = num_words

    def fit(self, token_sequences):
        self.tokenizer.fit_on_texts([" ".join(seq) for seq in token_sequences])

    def transform(self, token_sequences, max_len):
        sequences = self.tokenizer.texts_to_sequences([" ".join(seq) for seq in token_sequences])
        padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
        return padded

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.tokenizer, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.tokenizer = pickle.load(f)