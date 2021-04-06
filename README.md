# Link prediction in information networks
> [Kaggle challenge](https://www.kaggle.com/t/28307687dc7742c6ad081ba0edefcd89) for the MLNS course

## Project description
Edges have been deleted at random from a citation network. The goal is to accurately reconstruct the initial network using graph-theoretical, textual, and other information.

In this competition, we define a citation network as a graph where nodes are research papers and there is an edge between two nodes if one of the two papers cite the other.

In addition to the graph structure, each node (i.e., article) is also associated with information such as the title of the paper, publication year, author names and a short abstract.

Your goal is to utilize information from both the underlying citation network and the textual information in order to accurately predict missing edges.


## How to run the code
For this datachallenge, I implemented several methods to predict the links. The implementations of each method can be found in the `models` folder.

To run the program, do the following:
- install the required dependencies with:

 ```pip install -r requirements.txt```

- fill the `data` folder with the files from the Kaggle datachallenge
- specify the configuration of your experiments in the `cfg.yml` file
- run the program with:

 ```python main.py```

To have access to visualizations with wandb, do not forget to activate it in the `cfg.yml` file.

## Implemented methods
- `random`: random prediction
- `baseline`: baseline given with the data
- `svm`: like the baseline, uses the SVM, but with additional features
- `log_reg`: uses a Logistic Regression classifier
- `rand_forest`: uses a Random Forest classifier
- `boosting`: uses a Gradient Boosting classifier
- `graphsage`: uses a GraphSAGE network (implementation inspired by [this repository](https://github.com/raunakkmr/GraphSAGE-and-GAT-for-link-prediction/tree/master/src))
- `gat`: uses a Graph Attention Network (implementation inspired by [this repository](https://github.com/raunakkmr/GraphSAGE-and-GAT-for-link-prediction/tree/master/src))
