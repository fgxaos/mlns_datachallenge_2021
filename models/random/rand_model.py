### LIBRARIES ###
# Global libraries
import numpy as np

### FUNCTION DEFINITION ###
def random_predict(len_set):
    """Predicts randomly a connection.

    Args:
        len_set: int
            length of the dataset to use to predict
    Returns:
        predictions: np.array (values in {0, 1})
            random predictions for each element of the given dataset
    """
    return np.random.choice([0, 1], size=len_set)
    
