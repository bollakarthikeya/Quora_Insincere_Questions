# Quora Insincere Questions Classification

Kaggle Challenge: https://www.kaggle.com/c/quora-insincere-questions-classification

## Problem

Quora needs a scalable way to identify **insincere questions**—questions based on false premises or intended to make statements rather than seek useful answers (https://www.kaggle.com/c/quora-insincere-questions-classification). This project builds a binary text classifier to distinguish sincere (`0`) from insincere (`1`) questions.

## Approach

The solution uses Facebook's **fastText** open-source, lightweight machine learning library to perform supervised text classification by implementing text preprocessing, model experimentation, hyperparameter selection, validation, and test-set prediction.

## Workflow

1. **Load training data** from `train.csv`.
2. **Preprocess question text**:
   - Convert text to lowercase.
   - Remove punctuation.
   - Remove apostrophes.
   - Replace digits `0–9` with their word equivalents.
3. **Prepare fastText labels**:
   - `0 → __label__0`
   - `1 → __label__1`
4. **Shuffle the dataset** using a fixed random seed (`9999`).
5. **Split the data 80/20** into training and validation sets.
6. Export data in fastText supervised-learning format.
7. Train and compare multiple **fastText** configurations.
8. Tune embedding dimension, word n-grams, and character n-grams.
9. Train the selected combined model.
10. Preprocess `test.csv` using the same text pipeline.
11. Predict each test question and generate `sample_submission.csv`.

## Model Experiments

| Model / Experiment | Validation P@1 / R@1 |
|---|---:|
| Default unigram | 0.95301 |
| Unigram + character n-grams (`minn=2`, `maxn=6`) | 0.95267 |
| Bigram | 0.95391 |
| Trigram | 0.95465 |
| **Final combined model** | **0.95558** |

Additional searches were performed over:

- **Embedding dimensions:** `10, 25, 50, 75, 100, 150, 200, 250, 300`
- **Word n-grams:** `1–5`
- **Character n-grams:** combinations of `minn` and `maxn` from `2–6`

## Final Model

The parameters used in the final combined model are:

- `dim = 300`
- `wordNgrams = 3`
- `minn = 5`
- `maxn = 5`

```python
model = fasttext.train_supervised(
    input='fasttext_train.txt',
    wordNgrams=3,
    minn=5,
    maxn=5,
    dim=300
)
```

The remaining fastText parameters use the defaults documented in the notebook, including a learning rate of `0.1` and `5` epochs.

### Validation Result

```text
Validation examples: 261,225
P@1:                0.955582
R@1:                0.955582
```

The trained model is saved as:

```text
finalModel.bin
```

## Algorithms and Techniques

- Natural Language Processing (NLP)
- Binary text classification
- fastText supervised learning
- Text normalization
- Word n-grams
- Character subword n-grams
- Learned text-vector/embedding dimensions
- Hyperparameter experimentation
- Train/validation split
- Precision@1 and Recall@1 evaluation

## Output

The final model predicts labels for the processed Quora test questions. fastText labels are converted back to binary values:

```text
__label__0 → 0
__label__1 → 1
```

Predictions are exported to:

```text
sample_submission.csv
```

containing:

```text
qid, prediction
```

## Libraries Used

- Python
- pandas
- Facebook fastText
- PrettyTable
- CSV
