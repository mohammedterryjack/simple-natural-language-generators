from generators.dca_lma import DelayCoordinateEmbeddingNLG

x = DelayCoordinateEmbeddingNLG(save_path="data")
y = x.infer(sequence="i have never seen that")
print(y)