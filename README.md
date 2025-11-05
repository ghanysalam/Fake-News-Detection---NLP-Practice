# Fake News Detection — NLP Practical

## Background (Business Context)
Social platforms face a constant stream of user-generated text, some of which can be misleading or outright fake news. To support trust & safety workflows, this project builds an end-to-end NLP pipeline that explores language patterns (POS/NER), normalizes text, analyzes sentiment and topics, and then classifies whether an article is **Fake** or **Factual**. The working dataset contains **198** English-language articles with a balanced `fake_or_factual` label.

## Problem Statement
How can we reliably classify long-form articles as *fake* vs *factual* under noisy, varied language? The system should:
- handle longer text (title + body),
- remain robust after normalization (lowercasing, punctuation removal, stopword removal, tokenization, lemmatization, and stripping leading location tags),
- surface early signals for moderation (prioritize catching *fake*),
- be transparent to stakeholders via interpretable analyses (POS/NER, sentiment, topic patterns).

## Goals
- **Data & quick EDA.** Verify structure (198 rows; balanced labels), inspect text length distributions, and compare POS/NER patterns across classes.  
- **Consistent preprocessing.** Remove leading location tags with ``^[^-]*-\s``, lowercase, strip punctuation/stopwords, tokenize, and lemmatize into a `text_clean` field.  
- **Sentiment & topics.** Compute VADER sentiment (`compound` score + label) and compare by class; run topic modeling on fake news (LDA with **7 topics** and LSA/TF-IDF with **8 topics**) to reveal dominant themes.  
- **Features & models.** Vectorize with Bag-of-Words (CountVectorizer), perform a 70/30 split, and train baseline classifiers: **Logistic Regression** (test accuracy ≈ **82%**) and a linear **SVM** comparator (≈ **80%**).  
- **Clear evaluation.** Report accuracy and a full classification report; emphasize the recall of the *fake* class for triage while keeping precision reasonable.  
- **Integration-ready.** Package a reproducible pipeline (preprocessing → vectorization → inference) and model artifacts for easy handoff to moderation workflows.
