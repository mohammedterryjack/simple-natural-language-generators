class FastFourierTransformNLG:
    """
    Natural Language Generation using:
    Fast Fourier Transform (FFT)
    to model and forecast sequences
    """
    def __init__(self) -> None:
        pass
        #embed all vocab in data
        #dimensionality reduction on embeddings to 1D
        #store 1d values for vocab

        #encode data using 1D values
        #FFT on encoded data (all of them or just one? how to combine the data)
        #filter noise by zeroing bottom n% of coefficients
        #predict future values using IFFT

        #find nearest word for each forecasted value
        #return words as generated text