import re
import json

class MathTokenizer:
    def __init__(self, vocab_path="vocab.json"):
        self.PAD = "<PAD>"
        self.SOS = "<SOS>"
        self.EOS = "<EOS>"
        self.UNK = "<UNK>"

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
            
        self.token_to_id = vocab_data["token_to_id"]
        self.id_to_token = {int(k): v for k, v in vocab_data["id_to_token"].items()}
        self.vocab_size = len(self.token_to_id)

    def _split_latex(self, text):
        return re.findall(r"\\[a-zA-Z]+|[^\s]", str(text))

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