### LIBRARIES ###
# Global libraries
from tqdm import tqdm
from math import isnan, sqrt

import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing

from scipy import spatial

import spacy

### FUNCTION DEFINITIONS ###
def tf_idf(node_info, language):
    """Computes the TF-IDF vector of each paper

    Args:
        node_info: ListNode
            list with information about each node
        language: str
            language to use
    Returns:
        abstract_features_tfidf: Dict{int: np.array}
            TF-IDF features (abstract) of the corpus for each node ID
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


def process_title(title, stpwds, stemmer):
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


def process_features(
    dataset,
    node_dict,
    abstract_features_tfidf,
    title_features_tfidf,
    journal_names,
    dict_neighbors,
    training=True,
    language="english",
    step="training",
):
    """Processes the features.

    Args:
        dataset: List
            dataset to process
        node_dict: Dict
            dictionary to quickly find the nodes by their ID
        abstract_features_tfidf: Dict{int: np.array}
            dictionary linking a Node ID to its corresponding abstract TF-IDF features
        title_features_tfidf: Dict{int: np.array}
            dictionary linking a Node ID to its corresponding title TF-IDF features
        journal_names: Dict{str: int}
            associates a 1-hot encoding index for each journal name
        dict_neighbors: Dict{int: set(int)}
            neighbors for each node ID
        training: bool
            True if the labels are known (training, validation)
            False otherwise (test)
        language: str
            language to use
        step: str
            status used to display the current processing step
    Returns:
        features: np.array
            processed features
        labels: np.array
            ground-truth values
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

    pa_index = np.zeros(len(dataset))
    salton_index = np.zeros(len(dataset))
    local_aff_struct_index = np.zeros(len(dataset))

    # nlp = spacy.load("en_core_web_md")
    # nlp = spacy.load("en_core_web_lg")

    for i in tqdm(range(len(dataset)), desc=f"Processing examples ({step})"):
        source_info = node_dict[dataset[i].origin]
        target_info = node_dict[dataset[i].target]

        # Process the article titles
        # Compute the word overlap
        source_title = process_title(source_info.title, stpwds, stemmer)
        target_title = process_title(target_info.title, stpwds, stemmer)
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
        # source_abstract = nlp(source_info.abstract)
        # target_abstract = nlp(target_info.abstract)

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
        # abstract_diff[i] = source_abstract.similarity(target_abstract)

        pa_index[i] = len(dict_neighbors[dataset[i].origin]) * len(
            dict_neighbors[dataset[i].target]
        )
        intersect = len(
            dict_neighbors[dataset[i].origin].intersection(
                dict_neighbors[dataset[i].target]
            )
        )
        salton_index[i] = intersect / sqrt(
            len(dict_neighbors[dataset[i].origin])
            * len(dict_neighbors[dataset[i].target])
        )

        local_aff_struct_index[i] = intersect / len(
            dict_neighbors[dataset[i].origin]
        ) + intersect / len(dict_neighbors[dataset[i].target])

    features = np.array(
        # [overlap_title, temp_diff, comm_auth]
        # [overlap_title, temp_diff, comm_auth, local_aff_struct_index]
        # [overlap_title, temp_diff, impossible, comm_auth]
        # [overlap_title, temp_diff, comm_auth, abstract_diff]
        # [overlap_title, temp_diff, comm_auth, title_diff]
        [
            overlap_title,
            title_diff,
            temp_diff,
            impossible,
            comm_auth,
            abstract_diff,
            pa_index,
            salton_index,
            local_aff_struct_index,
        ]
    ).T
    features = preprocessing.scale(features)
    features = np.hstack((features, journals))

    if training:
        labels = [element.exists for element in dataset]
        labels_array = np.array(list(labels))

        return features, labels
    else:
        return features