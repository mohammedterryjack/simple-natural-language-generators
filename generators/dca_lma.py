from typing import Generator, Tuple, List, Dict, Optional

from numpy import ndarray

from utils.preprocessing import tokenise, pad, add_special_tokens, SpecialTokens
from utils.decoding import argmax, top_k_sampling
from utils.retrieval import measure_similarity, save_embeddings, save_keys, load_keys, load_embeddings
from utils.encoding import encode_tokens

class DelayCoordinateEmbeddingNLG:
    """
    Natural Language Generation using:
    Delay coordinate embedding with
    Lorenz Method of Analogues (LMA) 
    for model-free forecasting
    """    

    def __init__(
        self, 
        embedding_dimension:int=3, 
        embedding_delay:int=2, 
        max_generation_length:int=100,
        top_k:int=0,
        save_path:Optional[str]=None
    ) -> None:
        self.m = embedding_dimension
        self.τ = embedding_delay
        self.k = top_k
        self.forecast_length = max_generation_length
        self.slide_rate = 1
        self.save_path = save_path 
        self.delay_coordinate_embeddings:Dict[str,ndarray] = dict() if self.save_path is None else self.load_delay_coordinate_embeddings()

    def __len__(self) -> int:
        return self.τ * self.m

    def save_delay_coordinate_embeddings(self) -> None:
        if self.save_path is not None:
            save_keys(path_to_file=self.save_path, keys=list(self.delay_coordinate_embeddings))
            save_embeddings(path_to_file=self.save_path, embeddings=list(self.delay_coordinate_embeddings.values()))
    
    def load_delay_coordinate_embeddings(self) -> Dict[str,ndarray]:
        return dict(zip(load_keys(path_to_file=self.save_path), load_embeddings(path_to_file=self.save_path)))

    def train(self,sequences:List[str]) -> None:
        for sequence in sequences:
            tokens = self.format_sequence(
                sequence=sequence, 
                sequence_length=len(self), 
                is_training_example=True
            )
            for key,delay_coordinate_embedding in self.embed_sequence(tokens=tokens):
                self.delay_coordinate_embeddings[key] = delay_coordinate_embedding
        self.save_delay_coordinate_embeddings()

    def embed_sequence(self, tokens:List[str]) -> Generator[Tuple[str,ndarray],None,None]:
        end_of_sequence_index = len(tokens)-len(self)+self.τ
        for start_span_index in range(end_of_sequence_index):
            end_span_index = start_span_index + len(self) - 1
            yield self.delay_coordinate_embedding(
                tokens=tokens[start_span_index:end_span_index], 
                delay=self.τ
            )

    def infer(self, sequence:str) -> str:
        tokens = self.format_sequence(
            sequence=sequence, 
            sequence_length=len(self), 
            is_training_example=False
        )
        end_span_index = len(tokens)
        start_span_index = end_span_index - len(self) + 2
        selected_tokens = tokens[start_span_index:end_span_index]
        return ' '.join(self._infer(tokens=selected_tokens))

    def _infer(self, tokens:List[str],generated:Optional[List[str]]=None) -> List[str]:
        if generated is None:
            generated = []
        if len(generated) == self.forecast_length:
           return generated
        next_token = self._forecast_next_token(tokens=tokens)
        if next_token == SpecialTokens.EOS.value:
            return generated
        tokens = tokens[self.slide_rate:-1] + [next_token]
        generated.append(next_token)
        return self._infer(tokens=tokens, generated=generated)

    def _forecast_next_token(self, tokens:List[str]) -> str:
        tokens += [SpecialTokens.MASK.value]
        _,embedding = self.delay_coordinate_embedding(tokens=tokens, delay=self.τ)
        return self.lorenz_method_of_analogies(embedding=embedding)

    def lorenz_method_of_analogies(self, embedding:ndarray) -> str:
        keys = list(self.delay_coordinate_embeddings)
        similarities = list(map(
            lambda other_embedding: measure_similarity(embedding, other_embedding), 
            self.delay_coordinate_embeddings.values()
        ))
        nearest_key = self.retrieve_nearest_key(keys, similarities, self.k)
        return nearest_key.split()[-1] 

    @staticmethod
    def format_sequence(sequence:str, sequence_length:int, is_training_example:bool) -> List[str]:
        tokens = tokenise(sequence)
        padding_length = sequence_length - len(tokens)
        if padding_length > 0 and is_training_example:
            tokens = pad(tokens=tokens,pad_length=padding_length)
        return add_special_tokens(tokens=tokens, include_eos=is_training_example)

    @staticmethod
    def retrieve_nearest_key(keys:List[str], similarities:List[float], sample_size:int) -> str:
        return top_k_sampling(keys,similarities,sample_size) if sample_size > 0 else argmax(keys,similarities)

    @staticmethod
    def delay_coordinate_embedding(tokens:List[str], delay:int) -> Tuple[str,ndarray]:
        selected_tokens = list(map(
            lambda index:tokens[index],
            range(0,len(tokens),delay)
        ))
        return (
            ' '.join(selected_tokens), 
            encode_tokens(selected_tokens[:-1]) 
        )
