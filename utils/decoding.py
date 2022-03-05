from typing import List
from random import choices 

from utils.preprocessing import SpecialTokens

def argmax(candidates:List[str], similarities:List[float], min_similarity:float) -> str:
    best_similarity = max(similarities)
    if best_similarity < min_similarity:
        return f"{SpecialTokens.NONE_FOUND.value} {SpecialTokens.EOS.value}"
    index_best = similarities.index(best_similarity)
    return candidates[index_best]

def top_k_sampling(candidates:List[str], similarities:List[float], k:int) -> str:
    sample_size = min(len(candidates),k)
    data = list(zip(similarities,candidates))
    data.sort(reverse=True)
    sample = data[:sample_size]
    chosen_similarities,chosen_candidates = zip(*sample)
    return choices(chosen_candidates, weights=chosen_similarities, k=1)[0]
