import torch
import numpy as np
import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.wordpunct_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in sentence_words:
            bag[idx] = 1
    return bag


def bow_vector(sentence, all_words):
    # Convert a sentence to a bag-of-words vector using the vocabulary
    # defined by `all_words`
    # for discord NLP
    if isinstance(sentence, str):
        sentence = sentence.split()
    vector = torch.zeros(len(all_words))
    for word in sentence:
        if word in all_words:
            vector[all_words.index(word)] += 1
    return vector.view(1, -1)
