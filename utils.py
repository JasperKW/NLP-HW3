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

    # Example transformation: introduce typos
    text = example['text']
    words = text.split()

    # Synonym Replacement (50% chance for each word)
    for i in range(len(words)):
        if random.random() < 0.5:  # 50% chance to replace the word
            synonyms = wordnet.synsets(words[i])
            if synonyms:
                lemmas = synonyms[0].lemmas()
                if lemmas:
                    words[i] = lemmas[0].name().replace('_', ' ')

    # Introduce Typos (5% chance for each character)
    def introduce_typo(word):
        typo_idx = random.randint(0, len(word) - 1)
        if word[typo_idx] in string.ascii_letters:
            # Replace with a random nearby key (e.g., QWERTY typo)
            nearby_chars = {
                'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'ersfx', 'e': 'wsdr', 'f': 'rtgv', 'g': 'tyhb',
                'h': 'yujnb', 'i': 'ujko', 'j': 'uikmnh', 'k': 'ijol', 'l': 'kop', 'm': 'njk',
                'n': 'bhjm', 'o': 'iklp', 'p': 'ol', 'q': 'wa', 'r': 'edf', 's': 'wedxa', 't': 'rfgy',
                'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu', 'z': 'asx'
            }
            if word[typo_idx].lower() in nearby_chars:
                possible_replacements = nearby_chars[word[typo_idx].lower()]
                typo_char = random.choice(possible_replacements)
                word = word[:typo_idx] + typo_char + word[typo_idx + 1:]
        return word

    transformed_words = [introduce_typo(word) if random.random() < 0.05 else word for word in words]
    transformed_text = " ".join(transformed_words)

    # Update example with transformed text
    example['text'] = transformed_text

    ##### YOUR CODE ENDS HERE ######

    return example
