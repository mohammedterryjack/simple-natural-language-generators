from nltk.corpus import gutenberg

from generators.dca_lma import DelayCoordinateEmbeddingNLG

sentences = list(
    map(
        lambda sentence:' '.join(sentence),
        gutenberg.sents('austen-persuasion.txt')
    )
)[:100]

x = DelayCoordinateEmbeddingNLG(top_k=3)
x.train(sequences=sentences)

y = x.infer(sequence="then followed the history")
print(y)