### LIBRARIES ###
# Global libraries
from dataclasses import dataclass

### CLASS DEFINITIONS ###
@dataclass
class Edge:
    """Class to represent a reference/edge in the graph.

    Args:
        origin: int
            ID of the origin node
        target: int
            ID of the target node
        exists: int {0, 1}
            1 if the link exists, 0 otherwise
    """

    origin: int
    target: int
    exists: int = -1

    def __post_init__(self):
        if not isinstance(self.origin, int):
            self.origin = int(self.origin)
        if not isinstance(self.target, int):
            self.target = int(self.target)
        if not isinstance(self.exists, int):
            self.exists = int(self.exists)


@dataclass
class Node:
    """Class to represent a paper/node in the graph.

    Attributes:
        id: int
            ID of the paper
        year: int
            publication year of the paper
        title: str
            title of the paper
        authors: str
            name(s) of the author(s)
        journal: str
            name of the journal
        abstract: str
            abstract of the paper
    """

    id: int
    year: int
    title: str
    authors: str
    journal: str
    abstract: str

    def __post_init__(self):
        if not isinstance(self.id, int):
            self.id = int(self.id)
        if not isinstance(self.year, int):
            self.year = int(self.year)
        if not isinstance(self.title, str):
            self.title = str(self.title)
        if not isinstance(self.authors, str):
            self.authors = str(self.authors)
        if not isinstance(self.journal, str):
            self.journal = str(self.journal)
        if not isinstance(self.abstract, str):
            self.abstract = str(self.abstract)
