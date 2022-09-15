import torch
from typing import Dict, Any

LETTERS = '- абвгдежзийклмнопрстуфхцчшщъыьэюя'

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels: Dict, blank: int = 0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor):
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1) 
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = "".join([self.labels[i.item()] for i in indices])
        return joined.replace("|", " ").strip().split()

class TextEncDec:
    def __init__(self, 
        letters: str = LETTERS
        ):
        
        self.token_to_id = {
            letters[i] : i for i, _ in enumerate(letters)
        }
        self.id_to_token = {
            i : letters[i] for i, _ in enumerate(letters)
        }

    def encode(self, text):
        return list(map(self.token_to_id.get, text.lower()))

    def decode(self, idx):
        return list(map(self.id_to_token.get, idx))
