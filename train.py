from nltk.corpus import gutenberg

from generators.dca_lma import DelayCoordinateEmbeddingNLG

sentences = list(
    map(
        lambda sentence:' '.join(sentence),
        gutenberg.sents('austen-persuasion.txt')
    )
)[:500]

x = DelayCoordinateEmbeddingNLG(save_path="data")
x.train(sequences=sentences)