from enum import Enum 
from typing import List 

class SpecialTokens(Enum):
    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    MASK = "<?>"

def tokenise(sentence:str) -> str:
    return sentence.split()

def pad(tokens:List[str], pad_length:int) -> List[str]:
    return [SpecialTokens.PAD.value]*pad_length + tokens

def add_special_tokens(tokens:List[str], include_eos:bool) -> List[str]:
    tokens = [SpecialTokens.BOS.value] + tokens
    if include_eos:
        tokens += [SpecialTokens.EOS.value]
    return tokens