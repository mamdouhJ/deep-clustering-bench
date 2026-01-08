import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
from clustpy.deep import IDEC, DDC, DeepECT, DipEncoder, VaDE
from clustpy.deep.neural_networks import FeedforwardAutoencoder, VariationalAutoencoder 
from clustpy.metrics import unsupervised_clustering_accuracy as uca


ALGORITHMS = {"IDEC": IDEC,
              "DDC": DDC,
              "DeepECT": DeepECT,
              "DipEncoder": DipEncoder,
              "VaDE": VaDE}

def _build_autoencoder(nn_type: str,input_dim: int,arch: list,bottleneck_dim: int,random_state: int):
        """
        Helper function to build the appropriate autoencoder based on the specified type.
        Parameters:
        nn_type (str): Type of neural network ("ae" for Autoencoder, "vae" for Variational Autoencoder).
        arch (list): List defining the architecture of the neural network.
        input_dim (int): Dimensionality of the input data.
        bottleneck_dim (int): Dimensionality of the bottleneck layer.
        random_state (int): Random seed for reproducibility.
        Returns:
        Autoencoder instance.
        """
        layres = [input_dim] + arch + [bottleneck_dim]
        if nn_type == "ae":
                autoencoder = FeedforwardAutoencoder(layres,random_state=random_state)
                print(f"The seed for Autoencoder is: {random_state}")
        elif nn_type == "vae":
                autoencoder = VariationalAutoencoder(layres,random_state=random_state)
        else:
                raise ValueError(f"Unsupported nn_type: {nn_type}. Choose either 'ae' or 'vae'.")
        return autoencoder

def _build_idec(n_clusters: int, input_dim: int, arch: list, bottleneck_dim: int,
                nn_type: str, pretrain_lr: float, pretrain_epochs: int, clustering_lr: float,
                clustering_epochs: int, batch_size: int,seed: int):
        """
        Helper function to build an IDEC instance with the specified configuration.
        """
        if nn_type != "ae":
                raise ValueError("IDEC requires an Autoencoder (nn_type='ae').")
        
        autoencoder = _build_autoencoder(nn_type, input_dim, arch, bottleneck_dim,seed)
        idec = IDEC(n_clusters=n_clusters,batch_size=batch_size,pretrain_epochs=pretrain_epochs,
                    clustering_epochs=clustering_epochs,pretrain_optimizer_params={"lr": pretrain_lr},
                        clustering_optimizer_params={"lr": clustering_lr},neural_network=autoencoder,random_state=seed)
        print(f"The seed for IDEC is: {seed}")
        return idec

def _build_vade(n_clusters: int, input_dim: int, arch: list, bottleneck_dim: int,
                nn_type: str, pretrain_lr: float, pretrain_epochs: int, clustering_lr: float,
                clustering_epochs: int, batch_size: int,seed: int):
        """
        Helper function to build a VaDE instance with the specified configuration.
        """
        if nn_type != "vae":
                raise ValueError("VaDE requires a Variational Autoencoder (nn_type='vae').")
        autoencoder = _build_autoencoder(nn_type, input_dim, arch, bottleneck_dim,seed)
        vade = VaDE(n_clusters=n_clusters,batch_size=batch_size,pretrain_epochs=pretrain_epochs,
                    clustering_epochs=clustering_epochs,pretrain_optimizer_params={"lr": pretrain_lr},
                        clustering_optimizer_params={"lr": clustering_lr},neural_network=autoencoder,
                        embedding_size=bottleneck_dim,random_state=seed)
        return vade

def _build_ddc(n_clusters: int, input_dim: int, arch: list, bottleneck_dim: int,
                nn_type: str, pretrain_lr: float, pretrain_epochs: int, clustering_lr: float,
                clustering_epochs: int, batch_size: int,seed: int):
        """
        Helper function to build a DDC instance with the specified configuration.
        """
        if nn_type != "ae":
                raise ValueError("DDC requires an Autoencoder (nn_type='ae').")
        autoencoder = _build_autoencoder(nn_type, input_dim, arch, bottleneck_dim,seed)
        ddc = DDC(n_clusters=n_clusters,batch_size=batch_size,pretrain_epochs=pretrain_epochs,
                pretrain_optimizer_params={"lr": pretrain_lr},neural_network=autoencoder,random_state=seed)
        return ddc

def _build_ect(n_clusters: int, input_dim: int, arch: list, bottleneck_dim: int,
                nn_type: str, pretrain_lr: float, pretrain_epochs: int, clustering_lr: float,
                clustering_epochs: int, batch_size: int,seed: int):
        """
        Helper function to build a DeepECT instance with the specified configuration.
        """
        if nn_type != "ae":
                raise ValueError("DeepECT requires an Autoencoder (nn_type='ae').")
        autoencoder = _build_autoencoder(nn_type, input_dim, arch, bottleneck_dim,seed)
        deepect = DeepECT(max_n_leaf_nodes=n_clusters,batch_size=batch_size,pretrain_epochs=pretrain_epochs,
                    clustering_epochs=clustering_epochs,pretrain_optimizer_params={"lr": pretrain_lr},
                        clustering_optimizer_params={"lr": clustering_lr},neural_network=autoencoder,random_state=seed)
        return deepect

def _build_dipencoder(n_clusters: int, input_dim: int, arch: list, bottleneck_dim: int,
                nn_type: str, pretrain_lr: float, pretrain_epochs: int, clustering_lr: float,
                clustering_epochs: int, batch_size: int,seed: int):
        """
        Helper function to build a DipEncoder instance with the specified configuration.
        """
        if nn_type != "ae":
                raise ValueError("DipEncoder requires an Autoencoder (nn_type='ae').")
        autoencoder = _build_autoencoder(nn_type, input_dim, arch, bottleneck_dim,seed)
        dipencoder = DipEncoder(n_clusters=n_clusters,batch_size=batch_size,pretrain_epochs=pretrain_epochs,
                    clustering_epochs=clustering_epochs,pretrain_optimizer_params={"lr": pretrain_lr},
                        clustering_optimizer_params={"lr": clustering_lr},neural_network=autoencoder
                        ,embedding_size=bottleneck_dim,random_state=seed)
        return dipencoder

BUILDERS = {"IDEC": _build_idec,
                "DDC": _build_ddc,
                "DeepECT": _build_ect,
                "DipEncoder": _build_dipencoder,
                "VaDE": _build_vade}

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

        def __init__(self, config):
                self.config = config   
                self.transformed_data = None
                self.init_algorithm()
                self.fitted = False
        

        def init_algorithm(self):
                self.algo_name = self.config["algo"]
                if self.algo_name not in ALGORITHMS:
                        raise ValueError(f"Unsupported algorithm: {self.algo_name}. Choose from {list(ALGORITHMS.keys())}.")
                self.model = BUILDERS[self.algo_name](
                        n_clusters=self.config["data"]["n_clusters"],
                        input_dim=self.config["data"]["input_dim"],
                        arch=self.config["arch"],
                        bottleneck_dim=self.config.get("bottleneck_dim", 10),
                        nn_type=self.config["nn_type"],
                        pretrain_lr=self.config["pretrain"]["lr"],
                        pretrain_epochs=int(self.config["pretrain"]["epochs"]),
                        clustering_lr=self.config["clustering"]["lr"],
                        clustering_epochs=int(self.config["clustering"]["epochs"]),
                        batch_size=self.config["batch_size"],
                        seed = self.config["seed"]
                )
                
        def fit(self, data: np.ndarray, val_set: np.ndarray=None):
                """
                Fit the deep clustering model to the data.
                Parameters:
                data (np.ndarray): Input data for clustering.
                val_set (np.ndarray, optional): Validation set for early stopping. Defaults to None.
                """
                self.model.fit(data, val_set=val_set)
                self.fitted = True
                return self.model
        
        def predict(self, data: np.ndarray):
                """
                Predict cluster labels for the input data.
                Parameters:
                data (np.ndarray): Input data for clustering.
                """
                return self.model.predict(data)
        
        def transform(self, data: np.ndarray):
                """
                Transform the input data to the latent space.
                Parameters:
                data (np.ndarray): Input data to transform.
                """
                return self.model.transform(data)
        
        def get_metrics(self,groundtruth: np.ndarray, data: np.ndarray) -> dict:
                """
                Get the clustering metrics based on the ground truth labels.
                Parameters:
                groundtruth (np.ndarray): Ground truth labels for the data.
                Returns:
                dict: Dictionary containing clustering metrics. ("ARI", "NMI", "UCA", "Silhouette", "Davies-Bouldin")

                """
                if not self.fitted:
                        raise RuntimeError("Model must be fitted before calculating metrics.")
                embs = self.transform(data)
                predicted_labels = self.predict(data)
                if len(self.model.labels_) == len(predicted_labels):
                        predicted_labels = self.model.labels_
                metrics = {
                        "ARI": adjusted_rand_score(groundtruth, predicted_labels),
                        "NMI": normalized_mutual_info_score(groundtruth, predicted_labels),
                        "UCA": uca(groundtruth, predicted_labels),
                        "Silhouette": silhouette_score(embs, predicted_labels),
                        "Davies-Bouldin": davies_bouldin_score(embs, predicted_labels)
                }
                return metrics
        


        
