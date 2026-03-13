import re
import pandas as pd

class MathTokenizer:
    def __init__(self):
        self.PAD = "<PAD>"
        self.SOS = "<SOS>"
        self.EOS = "<EOS>"
        self.UNK = "<UNK>"

        self.token_to_id = {self.PAD: 0, self.SOS: 1, self.EOS: 2, self.UNK: 3}
        self.id_to_token = {0: self.PAD, 1: self.SOS, 2: self.EOS, 3: self.UNK}
        self.vocab_size = 4

    def _split_latex(self, text):
         return re.findall(r"\\[a-zA-z]+|[^s]", str(text))
    
    def build_vocab(self, csv_path):
        df = pd.read_csv(csv_path)
        for latex_str in df['label'].dropna():
            tokens = self._split_latex(latex_str)
            for token in tokens:
                if token not in self.token_to_id:
                    self.token_to_id[token] = self.vocab_size
                    self.id_to_token[self.vocab_size] = token
                    self.vocab_size += 1

    def encode(self, text):
        tokens = [self.token_to_id[self.SOS]]
        parsed_text = self._split_latex(text)

        for token in parsed_text:
            tokens.append(self.token_to_id.get(token, self.token_to_id[self.UNK]))
        tokens.append(self.token_to_id[self.EOS])
        return tokens
    
    def decode(self, token_ids):
        ignore_ids = [self.token_to_id[self.PAD], self.token_to_id[self.SOS], self.token_to_id[self.EOS]]
        return " ".join([self.id_to_token.get(idx, self.UNK) for idx in token_ids if idx not in ignore_ids])
    