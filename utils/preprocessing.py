from enum import Enum 
from typing import List 
from string import punctuation

TRANSLATION_TABLE = "".maketrans(punctuation, " " * len(punctuation))

class SpecialTokens(Enum):
    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    MASK = "<?>"

def clean_text(text:str) -> str:
    return text.lower().translate(TRANSLATION_TABLE)

def tokenise(sentence:str) -> str:
    return clean_text(sentence).split()

def pad(tokens:List[str], pad_length:int) -> List[str]:
    padding = [SpecialTokens.PAD.value]*pad_length
    return tokens + padding

def add_special_tokens(tokens:List[str], include_eos:bool) -> List[str]:
    tokens = [SpecialTokens.BOS.value] + tokens
    if include_eos:
        tokens += [SpecialTokens.EOS.value]
    return tokens