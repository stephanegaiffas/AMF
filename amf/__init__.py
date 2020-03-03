# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: GPL 3.0


from .dummy import OnlineDummyClassifier
from .forest import AMFClassifier

__all__ = [
    "OnlineDummyClassifier",
    "AMFClassifier",
]
