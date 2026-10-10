# Sentiment Classification: TF-IDF vs Dense Vectors

A text/web content-mining assignment exploring two ways to turn text into features for
sentiment classification: sparse **TF-IDF** vectors versus **dense (embedding-style)** vectors,
both fed into a logistic-regression classifier built step by step — manual dot product with
pretrained weights, sigmoid, cross-entropy loss — so the mechanics of each approach are explicit
rather than hidden behind a library call.

Run on a small labeled set of movie/book reviews, with a side-by-side comparison of predicted
probabilities, accuracy and cross-entropy loss between the two representations, plus a written
analysis of when each approach is preferable.

See [`Projetwebcontentmining.ipynb`](./Projetwebcontentmining.ipynb) for the full code and
[`projetwencontent.pdf`](./projetwencontent.pdf) for the written report.

## Task 1 — TF-IDF

- Builds a TF-IDF matrix by hand over a fixed 12-word vocabulary extracted from 4 reviews.
- Scores each review against a set of pretrained logistic-regression weights (dot product).
- Converts scores to probabilities via sigmoid, predicts positive/negative, and computes the
  cross-entropy loss per review against the true labels.
- Predicts the sentiment of a new, unseen review using the same pipeline.

## Task 2 — Dense vectors

- Repeats the exact same logistic-regression pipeline (score → sigmoid → predict → loss), this
  time using dense embedding-style vectors instead of TF-IDF, for a direct comparison on the
  same 4 reviews.
- Predicts the sentiment of a new dense vector using the trained weights.

## Task 3 — Comparison

- Recomputes both approaches side by side (this time using `sklearn.feature_extraction.text.TfidfVectorizer`
  for the TF-IDF side) and builds a comparison table of probabilities, predictions and loss per
  review for both representations.
- Compares overall accuracy and mean cross-entropy loss between TF-IDF and dense vectors.
- Visualizes the comparison (loss, accuracy, per-review loss, and a probability scatter plot).
- Written analysis of the two representations:
  - **Nature:** TF-IDF is a sparse representation based on term frequency/importance in the
    corpus; dense vectors are a compact, fixed-dimension representation capturing semantic
    concepts.
  - **Semantic capture:** TF-IDF only captures word presence/absence with no semantic
    relationships; dense vectors capture relationships between words and concepts.
  - **Best-fit applications:** TF-IDF suits keyword search and classification with a
    well-defined vocabulary; dense vectors suit applications that need contextual/semantic
    understanding.

## Stack

Python, NumPy, pandas, scikit-learn (`TfidfVectorizer`, `accuracy_score`, `log_loss`),
Matplotlib.

## Running it

```bash
pip install numpy pandas scikit-learn matplotlib scipy
jupyter notebook Projetwebcontentmining.ipynb
```
