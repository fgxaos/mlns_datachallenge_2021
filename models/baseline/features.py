### LIBRARIES ###
# Global libraries
from tqdm import tqdm

import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing

### FUNCTION DEFINITIONS ###
def tf_idf(node_info, language):
    """Computes the TF-IDF vector of each paper

    Args:
        node_info: ListNode
            list with information about each node
        language: str
            language to use
    Returns:
        features_tfidf: scipy.sparse.csr.csr_matrix
            TF-IDF features of the corpus
    """
    corpus = [element.abstract for element in node_info]
    vectorizer = TfidfVectorizer(stop_words=language)
    features_tfidf = vectorizer.fit_transform(corpus)

    return features_tfidf


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
    dataset, node_dict, training=True, language="english", step="training"
):
    """Processes the features.

    Args:
        dataset: List
            dataset to process
        node_dict: Dict
            dictionary to quickly find the nodes by their ID
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
    temp_diff = [0] * len(dataset)
    comm_auth = [0] * len(dataset)

    for i in tqdm(range(len(dataset)), desc=f"Processing examples ({step})"):
        source_info = node_dict[dataset[i].origin]
        target_info = node_dict[dataset[i].target]

        # Process the article titles
        source_title = process_title(source_info.title, stpwds, stemmer)
        target_title = process_title(target_info.title, stpwds, stemmer)

        # Extract the list of authors
        source_auth = source_info.authors.split(",")
        target_auth = target_info.authors.split(",")

        overlap_title[i] = len(set(source_title).intersection(set(target_title)))
        temp_diff[i] = source_info.year - target_info.year
        comm_auth[i] = len(set(source_auth).intersection(set(target_auth)))

    features = np.array([overlap_title, temp_diff, comm_auth]).T
    features = preprocessing.scale(features)

    if training:
        labels = [element.exists for element in dataset]
        labels_array = np.array(list(labels))

        return features, labels
    else:
        return features
