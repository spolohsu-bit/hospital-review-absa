#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LDA K=7 topic model training for Taiwanese hospital reviews.

Reproduces the LDA analysis reported in the manuscript:
- Corpus: 5,467 reviews (full analytic sample)
- Dictionary filtering: no_below=5, no_above=0.5
- Parameters: alpha='symmetric', eta='auto', passes=10,
              iterations=100, random_state=42
- Coherence metric: C_v

Requirements:
    pip install gensim pandas jieba scikit-learn

Usage:
    python run_lda_k7.py --data path/to/combined_hospital_data.csv
    python run_lda_k7.py --data path/to/combined_hospital_data.csv \
                         --stopwords stopwords.txt --medical-dict medical_dict.txt
"""

import argparse
from collections import Counter
from pathlib import Path

import jieba
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel


def tokenize_reviews(df, stopwords=None, medical_dict_path=None):
    if medical_dict_path and Path(medical_dict_path).exists():
        with open(medical_dict_path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    jieba.add_word(word)

    if stopwords is None:
        stopwords = set()

    texts = []
    for idx, review_text in enumerate(df["review_text"]):
        if pd.isna(review_text):
            continue
        tokens = list(jieba.cut(str(review_text)))
        filtered = [
            tok for tok in tokens
            if tok not in stopwords
            and len(tok) > 1
            and not tok.isdigit()
            and tok.strip()
        ]
        if filtered:
            texts.append(filtered)

        if (idx + 1) % 1000 == 0:
            print(f"  Tokenized {idx + 1}/{len(df)} reviews")

    print(f"Valid reviews after tokenization: {len(texts)}")
    return texts


def train_lda(texts, num_topics=7):
    dictionary = corpora.Dictionary(texts)
    orig_size = len(dictionary)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    dictionary.compactify()
    print(f"Dictionary: {orig_size} -> {len(dictionary)} tokens")

    corpus = [dictionary.doc2bow(text) for text in texts]

    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        alpha="symmetric",
        eta="auto",
        passes=10,
        iterations=100,
        random_state=42,
    )

    coherence_model = CoherenceModel(
        model=model, texts=texts, dictionary=dictionary, coherence="c_v"
    )
    coherence = coherence_model.get_coherence()
    print(f"Coherence (C_v): {coherence:.4f}")

    return model, dictionary, corpus, coherence


def print_topics(model, num_topics=7):
    print(f"\n{'Topic':<8} {'Top-10 Keywords'}")
    print("-" * 70)
    for tid in range(num_topics):
        words = model.show_topic(tid, topn=10)
        kw = ", ".join(w for w, _ in words)
        print(f"T{tid+1:<7} {kw}")


def print_distribution(model, corpus, texts):
    dominant = []
    for doc_bow in corpus:
        dist = model.get_document_topics(doc_bow)
        if dist:
            dominant.append(max(dist, key=lambda x: x[1])[0])
        else:
            dominant.append(-1)

    counts = Counter(dominant)
    total = len(texts)
    print(f"\n{'Topic':<8} {'Count':>8} {'%':>8}")
    print("-" * 28)
    for tid in range(7):
        c = counts.get(tid, 0)
        print(f"T{tid+1:<7} {c:>8} {c/total*100:>7.1f}%")


def main():
    parser = argparse.ArgumentParser(description="LDA K=7 for hospital reviews")
    parser.add_argument("--data", type=Path, required=True,
                        help="Path to combined_hospital_data.csv")
    parser.add_argument("--stopwords", type=Path, default=None,
                        help="Path to stopwords file (one word per line)")
    parser.add_argument("--medical-dict", type=Path, default=None,
                        help="Path to medical dictionary for jieba")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} reviews from {args.data.name}")

    stopwords = set()
    if args.stopwords and args.stopwords.exists():
        with open(args.stopwords, "r", encoding="utf-8") as f:
            stopwords = {line.strip() for line in f if line.strip()}
        print(f"Loaded {len(stopwords)} stopwords")

    texts = tokenize_reviews(df, stopwords, args.medical_dict)
    model, dictionary, corpus, coherence = train_lda(texts)

    print_topics(model)
    print_distribution(model, corpus, texts)


if __name__ == "__main__":
    main()
