'''
all module needed for text preprocessing
'''

# import string
import csv
import re
import itertools
import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer

GOLVE_VECTOR = pd.read_table('glove.twitter.27B.25d.txt',
                           sep=" ",
                           quoting=csv.QUOTE_NONE,
                           encoding='utf-8',
                           header=None)
GLOVE_LST = list(GOLVE_VECTOR.loc[:, 0])
GLOVE_DICT = {GLOVE_LST[num - 1]: num for num in range(1, len(GLOVE_LST) + 1)}


# it seems useless to use __init__ in class.
class TextPreprocessing():
    '''
    preprocessing data
    '''
    def __init__(self, text, max_length_tweets=20, max_length_dictionary=None):
        self.text = text
        self.max_length_tweets = max_length_tweets
        self.max_length_dictionary = max_length_dictionary

    def clean(self):
        '''
         The clean function  inputs a tweets text and output a plain english text including emoji
        '''
        text = self.text
        text = " ".join(text.split())
        text = text.lower()
        # punctuation = string.punctuation + '…' + '’' + '⚠'
        p_1 = r'https?:\/\/[\S\/]*'
        p_2 = r'#[\S]*'
        p_3 = r'@[\S]*'
        p_4 = r'\s?rt'
        p_5 = r'\n+'
        p_6 = r'\s?[0-9]\s?'
        text = re.sub(f'({p_1})|({p_2})|({p_3})|({p_4})|({p_5})|({p_6})',
                      '',
                      text,
                      flags=re.MULTILINE)
        # following code filter punctuation, but punctuation exists in glove dictionary
        # text = text.translate(str.maketrans('', '', punctuation))
        return text

    def tokenize_text(self):
        '''
        tokenized clean_text
        '''
        clean_text = self.clean()
        tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
        return tknzr.tokenize(clean_text)

    def replace_token_with_index(self):
        '''
            Replace token with index
        '''
        text_token = self.tokenize_text()
        tokens = []
        if self.max_length_dictionary:
            glove_dict = dict(itertools.islice(GLOVE_DICT.items(), self.max_length_dictionary))
        glove_dict = GLOVE_DICT
        for token in text_token:
            if glove_dict.get(token):
                tokens.append(glove_dict[token])
            else:
                tokens.append(0)
        return tokens

    def pad_sequence(self):
        '''
        Returns a padding sequence with same length
        '''

        tweetsword_token = self.replace_token_with_index()
        length = self.max_length_tweets - len(tweetsword_token)
        if length > 0:
            tweetsword_token.extend(np.repeat(0, length))
        else:
            tweetsword_token = tweetsword_token[:length]
        return tweetsword_token
