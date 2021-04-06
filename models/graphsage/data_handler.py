### LIBRARIES ###
# Global libraries
from tqdm import tqdm
from math import isnan

import numpy as np

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from scipy import spatial
import scipy.sparse as sp

import torch
from torch.utils.data import Dataset

# Custom libraries
from utils.file_handler import load_node_info

### CLASS DEFINITIONS ###
class GraphDataset(Dataset):
    """Dataset for link prediction."""

    def __init__(
        self,
        list_data,
        cfg,
        use_labels,
        mode="train",
        language="english",
        n_layers=2,
        normalize_adj=False,
    ):
        """Initiates a dataset for link prediction.

        Args:
            list_data: List[Edge]
                list of elements from the considered dataset
            cfg: Dict
                configuration to use
            use_labels: bool
                True if the labels are known (training, validation)
                False otherwise (test)
            mode: str
                either "train", "val" or "test"
            language: str
                language to use
            n_layers: int
                number of layers in the computation graph
            normalize_adj: Boolean
                whether to use symmetric normalization on the adjacency matrix
        """
        super(GraphDataset, self).__init__()

        self.mode = mode
        self.n_layers = n_layers
        self.normalize_adj = normalize_adj

        # Load node information
        node_info, node_dict = load_node_info(cfg["paths"]["node_infos"])

        # Compute the TF-IDF vector of each paper
        abstract_features_tfidf, title_features_tfidf = self.tf_idf(
            node_info, cfg["language"]
        )

        # Construct a one-hot embedding for each journal
        journal_names = {}
        journal_idx = 0
        for element in node_info:
            if element.journal not in journal_names:
                journal_names[element.journal] = journal_idx
                journal_idx += 1

        self.process_features(
            list_data,
            node_dict,
            abstract_features_tfidf,
            title_features_tfidf,
            journal_names,
            use_labels,
            language,
            mode,
        )

        ## Setting up the graph ##
        # Dictionary to "translate" node IDs in IDs used for
        # the adjacency matrix
        dict_node_ids = {}
        dict_node_idx = 0
        for edge in list_data:
            if edge.origin not in dict_node_ids:
                dict_node_ids[edge.origin] = dict_node_idx
                dict_node_idx += 1
            if edge.target not in dict_node_ids:
                dict_node_ids[edge.target] = dict_node_idx
                dict_node_idx += 1

        edges_t = np.zeros((len(list_data), 3), dtype=int)
        for i, edge in enumerate(list_data):
            edges_t[i, 0] = dict_node_ids[edge.origin]
            edges_t[i, 1] = dict_node_ids[edge.target]
            edges_t[i, 2] = edge.exists
        edges_s = np.unique(edges_t[:, :2], axis=0)
        self.n = len(dict_node_ids)
        self.m_s, self.m_t = edges_s.shape[0], edges_t.shape[0]

        adj = sp.coo_matrix(
            (np.ones(self.m_s), (edges_s[:, 0], edges_s[:, 1])),
            shape=(self.n, self.n),
            dtype=np.float32,
        )

        adj += adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        if normalize_adj:
            degrees = np.power(np.array(np.sum(adj, axis=1)), -0.5).flatten()
            degrees = sp.diags(degrees)
            adj = degrees.dot(adj.dot(degrees))

        self.adj = adj.tolil()
        self.edges_s = edges_s
        self.nbrs_s = self.adj.rows

        nbrs_t = [[] for _ in range(self.n)]
        for (u, v, t) in edges_t:
            nbrs_t[u].append((v, t))
            nbrs_t[v].append((u, t))
        self.nbrs_t = np.array(nbrs_t)

        last_time = np.max(edges_t[:, -1]) + 1
        timestamps = dict()
        for (u, v, t) in edges_t:
            if u not in timestamps.keys():
                timestamps[u] = dict()
            if v not in timestamps[u].keys():
                timestamps[u][v] = []
            timestamps[u][v].append(last_time - t)

            if v not in timestamps.keys():
                timestamps[v] = dict()
            if u not in timestamps[v].keys():
                timestamps[v][u] = []
            timestamps[v][u].append(last_time - t)

        for u in range(self.n):
            if u not in timestamps.keys():
                timestamps[u] = dict()
            timestamps[u][u] = [1]
        self.timestamps = timestamps
        ## Finished setting up graph

    def __getitem__(self, idx):
        """Loads the item of the given index.

        Args:
            index: int
                index of the item to extract
        Returns:
            features: torch.Tensor

            targets: torch.Tensor
                ground-truth labels
        """
        # Get the information from the init function

        return self.edges_s[idx], self.labels[idx]

    def __len__(self):
        """Computes the length of the dataset.

        Returns:
            length: int
                length of the dataset
        """
        return len(self.features)

    def get_dims(self):
        return self.features.shape[1], 1

    def collate_wrapper(self, batch):
        """Function used by the DataLoader.

        Args:
            batch: List
                list of `(edge, label)` from the dataset
        Returns:
            edges: np.array
                edges in the batch
            features: torch.FloatTensor (n' x input_dim)
                input node features
            node_layers:
                `node_layers[i]` corresponds to the nodes in
                the i-th layer of the computation graph
            mappings: List[Dict]
                maps node `v` (labelled 0 to |V|-1) in
                `node_layers[i]` to its position in
                `node_layers[i]`
            rows: np.array
                list of neighbors of nodes in `node_layers[0]`
            labels: torch.LongTensor
                labels for the edges in the batch in {0, 1}
        """
        idx = list(set([v.item() for sample in batch for v in sample[0][:2]]))

        node_layers, mappings = self._form_computation_graph(idx)

        rows = self.nbrs_s[node_layers[0]]
        features = self.features[node_layers[0], :]
        labels = torch.FloatTensor([sample[1] for sample in batch])
        edges = np.array([sample[0] for sample in batch])
        edges = np.array([mappings[-1][v] for v in edges.flatten()]).reshape(
            edges.shape
        )
        return edges, features, node_layers, mappings, rows, labels

    def _form_computation_graph(self, idx):
        """Creates a computation graph.

        Args:
            idx: int or List[int]
                indices of the nodes for which the forward pass
                needs to be computed
        Returns:
            node_layers: List[np.array]
                `node_layers[i]` corresponds to the nodes in the
                i-th layer of the computation graph
            mappings: List[Dict]
                maps node `v` (labelled 0 to |V|-1) in
                `node_layers[i]` to its position in `node_layers[i]`
        """
        _list, _set = list, set
        if type(idx) is int:
            node_layers = [np.array([idx], dtype=np.int64)]
        elif type(idx) is list:
            node_layers = [np.array(idx, dtype=np.int64)]

        for _ in range(self.n_layers):
            prev = node_layers[-1]
            arr = [node for node in prev]
            arr.extend([e[0] for node in arr for e in self.nbrs_t[node]])
            arr = np.array(_list(_set(arr)), dtype=np.int64)
            node_layers.append(arr)
        node_layers.reverse()

        mappings = [{j: i for (i, j) in enumerate(arr)} for arr in node_layers]

        return node_layers, mappings

    def process_features(
        self,
        dataset,
        node_dict,
        abstract_features_tfidf,
        title_features_tfidf,
        journal_names,
        use_labels=True,
        language="english",
        mode="training",
    ):
        """Processes the features.

        Args:
            dataset: List[Edge]
                dataset to process
            node_dict: Dict {int: Node}
                dictionary to quickly find the nodes by their ID
            abstract_features_tfidf: Dict{int: np.array}
                dictionary linking a node ID to its corresponding abstract TF-IDF features
            title_features_tfidf: Dict{str: np.array}
                dictionary linking a Node ID to its corresponding title TF-IDF features
            journal_names: Dict{str: int}
                associates a 1-hot encoding index for each journal name
            training: bool
                True if the labels are known (training, validation)
                False otherwise (test)
            language: str
                language to use
            mode: str
                status used to dispay the current processing mode
        """
        stpwds = set(nltk.corpus.stopwords.words(language))
        stemmer = nltk.stem.PorterStemmer()

        overlap_title = [0] * len(dataset)
        title_diff = [0] * len(dataset)
        temp_diff = [0] * len(dataset)
        impossible = [0] * len(dataset)
        comm_auth = [0] * len(dataset)
        abstract_diff = [0] * len(dataset)
        journals = np.zeros((len(dataset), len(journal_names)), dtype=int)

        for i in tqdm(range(len(dataset)), desc=f"Processing examples ({mode})"):
            source_info = node_dict[dataset[i].origin]
            target_info = node_dict[dataset[i].target]

            # Process the article titles
            # Compute the word overlap
            source_title = self.process_title(source_info.title, stpwds, stemmer)
            target_title = self.process_title(target_info.title, stpwds, stemmer)
            # Compute cosine distance between their TF-IDF features
            source_title_features = title_features_tfidf[dataset[i].origin]
            target_title_features = title_features_tfidf[dataset[i].target]

            # Extract the list of authors
            source_auth = source_info.authors.split(",")
            target_auth = target_info.authors.split(",")

            # Compare the journals
            idx_journal_source = journal_names[source_info.journal]
            idx_journal_target = journal_names[target_info.journal]
            journals[i, idx_journal_source] = 1
            journals[i, idx_journal_target] = 1

            # Compare the abstracts
            source_abstract = abstract_features_tfidf[dataset[i].origin]
            target_abstract = abstract_features_tfidf[dataset[i].target]

            overlap_title[i] = len(set(source_title).intersection(set(target_title)))
            title_diff[i] = spatial.distance.cosine(
                source_title_features, target_title_features
            )
            if isnan(title_diff[i]):
                title_diff[i] = 0

            temp_diff[i] = source_info.year - target_info.year
            impossible[i] = int(source_info.year >= target_info.year)
            comm_auth[i] = len(set(source_auth).intersection(set(target_auth)))
            abstract_diff[i] = spatial.distance.cosine(source_abstract, target_abstract)

        features = np.array(
            [overlap_title, title_diff, temp_diff, impossible, comm_auth, abstract_diff]
        ).T
        features = preprocessing.scale(features)
        features = np.hstack((features, journals))
        self.features = torch.from_numpy(features)

        if use_labels:
            labels = [element.exists for element in dataset]
            self.labels = torch.Tensor(list(labels))

    def process_title(self, title, stpwds, stemmer):
        """Processes the article title.

        Args:
            title: str
                title to process
            stpwds: set
                set of stop words
            stemmer: nltk.stem.PorterStemmer
                stemmer
        Returns:
            title: str
                processed title
        """
        # Convert article title to lowercase and tokenize it
        title = title.lower().split(" ")
        # Remove the stopwords
        title = [token for token in title if token not in stpwds]
        return [stemmer.stem(token) for token in title]

    def tf_idf(self, node_info, language):
        """Computes the TF-IDF vector for each paper

        Args:
            node_info: ListNode
                list with information about each node
            language: str
                language to use
        Returns:
            abstract_features_tfidf: Dic{int: np.array}
                TF-IDF features (abstract) of the corpus fo reach node ID
            title_features_tfidf: Dict{int: np.array}
                TF-IDF features (title) of the corpus for each node ID
        """
        corpus = [element.abstract for element in node_info]
        for element in node_info:
            corpus.append(element.title)
        vectorizer = TfidfVectorizer(stop_words=language)
        features = vectorizer.fit_transform(corpus).todense()

        abstract_features_tfidf = {}
        title_features_tfidf = {}
        for i, element in enumerate(node_info):
            abstract_features_tfidf[element.id] = features[i]
            title_features_tfidf[element.id] = features[i + len(node_info)]

        return abstract_features_tfidf, title_features_tfidf
