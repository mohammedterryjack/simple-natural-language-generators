from typing import List 

from numpy import ndarray, concatenate, zeros, ones, full
import en_core_web_md

from utils.preprocessing import SpecialTokens

spacy_encoder = en_core_web_md.load()
spacy_embedding_dimension = 300

def encode_token(token:str) -> ndarray:
    if token == SpecialTokens.MASK.value:
        return zeros(spacy_embedding_dimension)
    if token in (
        SpecialTokens.PAD.value,
        SpecialTokens.EOS.value,
        SpecialTokens.BOS.value,
    ):
        return full(spacy_embedding_dimension, -1.)
    spacy_token = spacy_encoder(token)
    if spacy_token.has_vector:
        return spacy_token.vector
    return ones(spacy_embedding_dimension)

def encode_tokens(tokens:List[str]) -> ndarray:
    return concatenate(list(map(encode_token,tokens)))
