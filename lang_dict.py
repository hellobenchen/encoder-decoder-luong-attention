# -*- coding: utf-8 -*-
"""

read a language to language mapping file into dictionary
copied and modified from https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation

"""


import unicodedata
import re
import torch

SOS_token = 0 #start of sentence, index=0
EOS_token = 1 #end of sentence, index=1

class Lang:
    def __init__(self, name):
        #dictionary of a language
        #index number to word
        #word to index number
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2 # Count SOS and EOS
      
    #add words from a sentence to the index
    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    #add a word to the index
    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            
            
#Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)  # \1 means the first matched group, add space (to itself) to before [.!?] here so they are considered seprated words
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s) #remove other characters
    return s


def read_langs(file_path, lang1, lang2):
    print("Reading lines... language 1: {0}, language 2: {1}".format(lang1, lang2))

    # Read the file and split into lines
    lines = open(file_path).read().strip().split('\n')
    
    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')[:2]] for l in lines]
    
    # Reverse pairs, make Lang instances
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
        
    return input_lang, output_lang, pairs

def prepare_data(file_path, input_lang_name, output_lang_name):
    input_lang, output_lang, pairs = read_langs(file_path, input_lang_name, output_lang_name)
    print("Read %s sentence pairs" % len(pairs))    
    
    print("Indexing words...")
    for pair in pairs: 
        input_lang.index_words(pair[0])
        output_lang.index_words(pair[1])

    return input_lang, output_lang, pairs


# Return a list of indexes, one for each word in the sentence
def word_indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def word_encodings_from_sentence(lang, sentence):
    indexes = word_indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.LongTensor(indexes)


def word_encodings_from_pair(input_lang, output_lang, pair):
    input_word_encodings = word_encodings_from_sentence(input_lang, pair[0])
    target_word_encodings = word_encodings_from_sentence(output_lang, pair[1])
    return input_word_encodings, target_word_encodings

def word_encodings_from_pairs(input_lang, output_lang, pairs):
    inputs = []
    input_lengths = []
    targets = []
    target_lengths = []
    for pair in pairs:
        i, t = word_encodings_from_pair(input_lang, output_lang, pair)
        inputs.append(i)
        input_lengths.append(len(i))
        targets.append(t)
        target_lengths.append(len(t))
    return inputs, input_lengths, targets, target_lengths