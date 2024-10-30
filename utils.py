import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation
    text = example["text"]  # Assuming each example has a "text" field
    typo_probability = 0.1  # 10% probability to introduce a typo

    # Simulate typos by replacing characters with adjacent keys on the QWERTY keyboard
    qwerty_neighbors = {
        'a': ['s', 'q', 'z'], 'b': ['v', 'n', 'g'], 'c': ['x', 'v'], 'd': ['s', 'f', 'e'],
        'e': ['w', 'r', 'd'], 'f': ['d', 'g', 'r'], 'g': ['f', 'h', 't'], 'h': ['g', 'j', 'y'],
        'i': ['u', 'o', 'k'], 'j': ['h', 'k', 'u'], 'k': ['j', 'l', 'i'], 'l': ['k', 'o'],
        'm': ['n', 'j'], 'n': ['b', 'm', 'h'], 'o': ['i', 'p', 'l'], 'p': ['o', 'l'],
        'q': ['w', 'a'], 'r': ['e', 't', 'f'], 's': ['a', 'd', 'w'], 't': ['r', 'y', 'g'],
        'u': ['y', 'i', 'j'], 'v': ['c', 'b'], 'w': ['q', 'e', 's'], 'x': ['z', 'c'],
        'y': ['t', 'u', 'h'], 'z': ['x', 'a']
    }

    new_text = []
    for char in text:
        if char in qwerty_neighbors and random.random() < typo_probability:
            new_text.append(random.choice(qwerty_neighbors[char]))
        else:
            new_text.append(char)

    example["text"] = ''.join(new_text)
    
    ##### YOUR CODE ENDS HERE ######

    return example
