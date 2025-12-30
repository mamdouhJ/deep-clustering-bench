from clustpy.deep import IDEC, DDC, DeepECT, DipEncoder, VaDE
from clustpy.deep.neural_networks import FeedforwardAutoencoder, VariationalAutoencoder 


ALGORITHMS = {"IDEC": IDEC,
              "DDC": DDC,
              "DeepECT": DeepECT,
              "DipEncoder": DipEncoder,
              "VaDE": VaDE}

)

class DeepClusteringAlgorithm:
        """
        Wrapper class for deep clustering algorithms from ClustPy.

        Parameters:
        config (dict): Configuration dictionary for the deep clustering algorithm.
        Configuration should include:
                "algo": (str): Name of the deep clustering algorithm to use. Must be one of the keys in ALGORITHMS.
                "data": (dict): Data configuration dictionary containing:
                        {"input_dim": (int): Dimensionality of the input data.,
                        "n_clusters": (int): Number of clusters.,
                        "did": (int): Dataset ID.}
                "pretrain": (dict): Pretraining configuration dictionary containing:
                        {lr: (float): Learning rate for pretraining.,
                        epochs: (int): Number of epochs for pretraining.}
                "clustering": (dict): Clustering configuration dictionary containing:
                        {lr: (float): Learning rate for clustering.,
                        epochs: (int): Number of epochs for clustering.}
                "batch_size": (int): Batch size for training.
                "early_stopping": (bool): Whether to use early stopping during training.
                "arch": (list): List defining the architecture of the neural network.
                "nn_type": (str): Type of neural network to use. Must be either "ae" or "vae".

                        
        """

        def __init__(self, config, random_state=None):
                self.config = config    
                
