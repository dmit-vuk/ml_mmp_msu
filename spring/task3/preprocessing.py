from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import xml
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    file = open(filename, 'r', encoding='utf-8')
    s = file.read()
    s = s.replace('&', '&amp;')
    file.close()
    
    file = open('new', 'w', encoding='utf-8')
    file.write(s)
    file.close()
    
    sentence_pairs, alignments = [], []
    tree = ET.parse('new')
    root = tree.getroot()
    for token in root:
        sentence = SentencePair(token[0].text.split(), token[1].text.split())
        sentence_pairs.append(sentence)
        sure, possible = [], []
        if(token[2].text is not None):
            sure = [(int(s.split('-')[0]), int(s.split('-')[1])) for s in token[2].text.split()]
        if(token[3].text is not None):
            possible = [(int(s.split('-')[0]), int(s.split('-')[1])) for s in token[3].text.split()]
        alignment = LabeledAlignment(sure, possible)
        alignments.append(alignment)
    return sentence_pairs, alignments


def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff -- natural number -- most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language
        
    Tip: 
        Use cutting by freq_cutoff independently in src and target. Moreover in both cases of freq_cutoff (None or not None) - you may get a different size of the dictionary

    """
    source_vocab, target_vocab = {}, {}
    for sentense in sentence_pairs:
        for token in sentense.source:
            if token in source_vocab.keys():
                source_vocab[token] += 1
            else:
                source_vocab[token] = 1
        
        for token in sentense.target:
            if token in target_vocab.keys():
                target_vocab[token] += 1
            else:
                target_vocab[token] = 1
    source_vocab = dict(sorted(source_vocab.items(), key=lambda item: item[1], reverse=True))
    target_vocab = dict(sorted(target_vocab.items(), key=lambda item: item[1], reverse=True))
    
    if freq_cutoff is not None:
        source_vocab = {token: it for it, token in enumerate(list(source_vocab.keys())[:freq_cutoff])}
        target_vocab = {token: it for it, token in enumerate(list(target_vocab.keys())[:freq_cutoff])}
    else:
        source_vocab = {token: it for it, token in enumerate(source_vocab.keys())}
        target_vocab = {token: it for it, token in enumerate(target_vocab.keys())}
    return source_vocab, target_vocab


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    res = []
    for sentense in sentence_pairs:
        source_ints, target_ints = [], []
        
        for token in sentense.source:
            if token in source_dict.keys():
                source_ints.append(source_dict[token])
            else:
                source_ints = []
        
        for token in sentense.target:
            if token in target_dict.keys():
                target_ints.append(target_dict[token])
            else:
                target_ints = []
        if len(target_ints) and len(source_ints):
            token_sentence = TokenizedSentencePair(np.array(source_ints), np.array(target_ints))
            res.append(token_sentence)
    return res
