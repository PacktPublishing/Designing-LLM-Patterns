# Chapter 6. Dataset Annotation and Labeling for LLMs

## 6.1 The Importance of Quality Annotations in LLM Development

```python
import spacy
from spacy.tokens import DocBin
from spacy.training import Example

def create_training_data(texts, annotations):
    nlp = spacy.blank("en")
    db = DocBin()
    
    for text, annot in zip(texts, annotations):
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annot:
            span = doc.char_span(start, end, label=label)
            if span:
                ents.append(span)
        doc.ents = ents
        db.add(doc)
    
    return db

texts = [
    "Apple Inc. is planning to open a new store in New York.",
    "Microsoft CEO Satya Nadella announced new AI features."
]
annotations = [
    [(0, 9, "ORG"), (41, 49, "GPE")],
    [(0, 9, "ORG"), (14, 27, "PERSON")]
]

training_data = create_training_data(texts, annotations)
training_data.to_disk("./train.spacy")
```

## 6.2 Annotation Strategies for Different LLM Tasks

### 6.2.1 Text Classification

```python
from datasets import Dataset

texts = [
    "This movie was fantastic!",
    "The service was terrible.",
    "The weather is nice today."
]
labels = [1, 0, 2]  # 1: positive, 0: negative, 2: neutral

dataset = Dataset.from_dict({"text": texts, "label": labels})
print(dataset[0])
```

### 6.2.2 Named Entity Recognition (NER)

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Apple Inc. was founded by Steve Jobs"
labels = ["B-ORG", "I-ORG", "O", "O", "O", "O", "B-PER", "I-PER"]

tokens = tokenizer.tokenize(text)
inputs = tokenizer(text, return_tensors="pt")
print(list(zip(tokens, labels)))
```

### 6.2.3 Question Answering

```python
context = "The capital of France is Paris. It is known for its iconic Eiffel Tower."
question = "What is the capital of France?"
answer = "Paris"

start_idx = context.index(answer)
end_idx = start_idx + len(answer)

print(f"Answer: {context[start_idx:end_idx]}")
print(f"Start index: {start_idx}, End index: {end_idx}")
```

## 6.3 Tools and Platforms for Large-Scale Text Annotation

```python
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification

def load_doccano_ner(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

doccano_data = load_doccano_ner('doccano_export.jsonl')

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased")

for item in doccano_data:
    text = item['text']
    labels = item['labels']
    
    # Process annotations and prepare for model input
    tokens = tokenizer.tokenize(text)
    ner_tags = ['O'] * len(tokens)
    for start, end, label in labels:
        start_token = len(tokenizer.tokenize(text[:start]))
        end_token = len(tokenizer.tokenize(text[:end]))
        ner_tags[start_token] = f'B-{label}'
        for i in range(start_token + 1, end_token):
            ner_tags[i] = f'I-{label}'
    
    # Now you can use tokens and ner_tags for model training or inference
```

## 6.4 Managing Annotation Quality for LLM Datasets

```python
from sklearn.metrics import cohen_kappa_score

annotator1 = [0, 1, 2, 0, 1]
annotator2 = [0, 1, 1, 0, 1]

kappa = cohen_kappa_score(annotator1, annotator2)
print(f"Cohen's Kappa: {kappa}")

def calculate_accuracy(gold_standard, annotations):
    return sum(g == a for g, a in zip(gold_standard, annotations)) / len(gold_standard)

gold_standard = [0, 1, 2, 0, 1]
annotator_result = [0, 1, 1, 0, 1]

accuracy = calculate_accuracy(gold_standard, annotator_result)
print(f"Accuracy: {accuracy}")
```

## 6.5 Crowd-sourcing Annotations: Benefits and Challenges

```python
from collections import Counter

def aggregate_annotations(annotations):
    return Counter(annotations).most_common(1)[0][0]

crowd_annotations = [
    ['PERSON', 'PERSON', 'ORG', 'PERSON'],
    ['PERSON', 'ORG', 'ORG', 'PERSON'],
    ['PERSON', 'PERSON', 'ORG', 'LOC']
]

aggregated = [aggregate_annotations(annot) for annot in zip(*crowd_annotations)]
print(f"Aggregated annotations: {aggregated}")
```

## 6.6 Semi-automated Annotation Techniques

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def semi_automated_ner(text):
    doc = nlp(text)
    return [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

text = "Apple Inc. was founded by Steve Jobs in Cupertino."
auto_annotations = semi_automated_ner(text)
print(f"Auto-generated annotations: {auto_annotations}")
```

## 6.7 Scaling Annotation Processes for Massive Language Datasets

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner

# Simulated unlabeled dataset
X_pool = np.random.rand(1000, 10)

# Initialize active learner
learner = ActiveLearner(
    estimator=RandomForestClassifier(),
    X_training=X_pool[:10],
    y_training=np.random.randint(0, 2, 10)
)

# Active learning loop
n_queries = 100
for _ in range(n_queries):
    query_idx, query_inst = learner.query(X_pool)
    # In real scenario, get human annotation here
    y_new = np.random.randint(0, 2, 1)
    learner.teach(X_pool[query_idx], y_new)
    X_pool = np.delete(X_pool, query_idx, axis=0)

print(f"Model accuracy after active learning: {learner.score(X_pool, np.random.randint(0, 2, len(X_pool)))}")
```
