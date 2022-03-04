from generators.dca_lma import DelayCoordinateEmbeddingNLG

examples = [
    "hi fahtima how are you ? ? ? ",
    "hi whats up this is your dad ? ? ? ",
    "hello family how are you ? ? ? ",
    "hey what are you doing today ? ? ? "
]
x = DelayCoordinateEmbeddingNLG()
x.train(sequences=examples)

y = x.infer(sequence="hi fahtima how is")
print(y)
