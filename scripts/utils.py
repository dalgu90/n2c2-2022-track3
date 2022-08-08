# -*- coding: utf-8 -*-
# Author: Juyong Kim (dalgu90@gmail.com)
""" Util functions used in analysis """

from pprint import pprint
import re
from string import punctuation

from fastDamerauLevenshtein import damerauLevenshtein
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from rouge import Rouge
from rouge_score import rouge_scorer


# Markup text
def textbf(text):
    return '\033[1m' + text + '\033[0m'

def textcolor(text, color):
    if color == 'gray':
        return '\u001b[38;5;244m' + text + '\033[0m'
    code = {'red': '31', 'green': '32', 'yellow': '33', 'blue': '34', 'magenta': '35', 'cyan': '36', 'white': '37'}
    return '\033[' + code[color] + 'm' + text + '\033[0m'

def mark_keyword(text, keyword, color='red'):
    ret = text
    matches_rev = list(re.finditer(keyword, text, re.IGNORECASE))[::-1]
    for m in matches_rev:
        start, end = m.span()
        ret = ret[:start] + textcolor(ret[start:end], color) + ret[end:]
    return ret

def pad_left(s, l):
    return ' ' * max(0, l-len(s)) + s[:l]

def pad_right(s, l):
    return s[:l]+ ' ' * max(0, l-len(s))

def format_score_str(score):
    assert 0.0 <= score <= 1.0
    score_str = f'{score:.3f}'
    if score >= 0.95:
        return textcolor(textbf(score_str), 'green')
    elif 0.6001 <= score < 0.95:
        return textcolor(score_str, 'green')
    elif 0.3999 <= score < 0.6001:
        return textcolor(score_str, 'gray')
#         return score_str
    elif 0.05 <= score < 0.3999:
        return textcolor(score_str, 'red')
    else: # score < 0.05
        return textcolor(textbf(score_str), 'red')


# Rouge scores
customStopWords = set(stopwords.words('english')+list(punctuation))
porter = PorterStemmer()
rouge = Rouge()
google_rouge_dict = {}

def get_rouge_score(ref, hyp, stopword_removal=True, stemming=True, version='google', metric='rouge1', verbose=False):
    # version='python' -> metric in ['rouge-1', 'rouge-2', 'rouge-l']
    # version='google' -> metric in ['rouge1', ..., 'rouge9', 'rougeL', 'rougeLsum']
    ref_orig, hyp_orig = ref, hyp

    # Tokenize (list of words)
    ref = word_tokenize(ref)
    hyp = word_tokenize(hyp)

    # Stopword removal (optional)
    if stopword_removal:
        ref = [word for word in ref if word not in customStopWords]
        hyp = [word for word in hyp if word not in customStopWords]

    # Stemming (optional)
    if stemming:
        ref = map(porter.stem, ref)
        hyp = map(porter.stem, hyp)

    # Joining
    ref = ' '.join(ref)
    hyp = ' '.join(hyp)
    if verbose:
        print(f'Ref: {ref_orig} => {ref}')
        print(f'Hyp: {hyp_orig} => {hyp}')

    # When both are empty, mark as match. When only one of them is empty, mark as mismatch
    ref_empty = not ref.strip()
    ref_orig_empty = not ref_orig.strip()
    hyp_empty = not hyp.strip()
    hyp_orig_empty = not hyp_orig.strip()
    if ref_empty and ref_orig_empty and hyp_empty and hyp_orig_empty: return 1.0
    if ref_empty or hyp_empty: return 0.0

    if version == 'python':
        scores_all = rouge.get_scores(hyp, ref)[0]
        if verbose:
            for name, score_dict in scores_all.items():
                print(f'{name}: r={score_dict["r"]:.4f}  p={score_dict["p"]:.4f}  f={score_dict["f"]:.4f}')
        return scores_all[metric]['f']
    elif version == 'google':
        global google_rouge_dict
        if metric not in google_rouge_dict:
            scorer = rouge_scorer.RougeScorer(rouge_types=[metric], use_stemmer=False)
            google_rouge_dict[metric] = scorer
        else:
            scorer = google_rouge_dict[metric]
        scores_all = scorer.score(ref, hyp)
        if verbose:
            pprint(scores_all)
        return scores_all[metric].fmeasure
    else:
        raise ValueError(f'Wrong version: {version}. Should be either python or datasets')


# Levenshtein score
def get_levenshtein_score(arr1, arr2, similarity=False):
    return damerauLevenshtein(arr1, arr2, similarity=similarity)
