from typing import List, Generator
from scipy.spatial.distance import cosine
from os.path import exists 
from enum import Enum

from numpy import ndarray, savetxt, loadtxt

class FileNames(Enum):
    KEYS = "keys.txt"
    EMBEDDINGS = "embeddings.txt"

def measure_similarity(embedding_a:ndarray, embedding_b:ndarray) -> float:
    return 1.0-cosine(embedding_a, embedding_b)

def save_keys(path_to_file:str, keys:List[str]) -> None:
    with open(f"{path_to_file}/{FileNames.KEYS.value}",'w') as key_file:
        for key in keys:
            key_file.write(f"{key}\n")

def save_embeddings(path_to_file:str, embeddings:List[ndarray]) -> None:
    savetxt(f"{path_to_file}/{FileNames.EMBEDDINGS.value}", embeddings, fmt='%.2f')

def load_keys(path_to_file:str) -> Generator[str,None,None]:
    filename = f"{path_to_file}/{FileNames.KEYS.value}"
    if exists(filename):
        with open(filename) as key_file:
            for line in key_file.readlines():
                yield line.strip()

def load_embeddings(path_to_file:str) -> List[ndarray]:
    filename = f"{path_to_file}/{FileNames.EMBEDDINGS.value}"
    if exists(filename):
        return loadtxt(filename, dtype=float)
    return []