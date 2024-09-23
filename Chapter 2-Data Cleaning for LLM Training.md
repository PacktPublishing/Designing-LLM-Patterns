Certainly! I'll extract all the Python code from the document and organize it by section number. Here's the compiled code:

2.1 Understanding the importance of clean data:

```python
import torch
from transformers import GPT4LMHeadModel, GPT4Tokenizer

def calculate_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()

model = GPT4LMHeadModel.from_pretrained("GPT4")
tokenizer = GPT4Tokenizer.from_pretrained("GPT4")

# Example texts
clean_text = "The quick brown fox jumps over the lazy dog."
noisy_text = "Th3 qu1ck br0wn f0x jumps 0ver th3 l@zy d0g."

# Calculate perplexity
clean_perplexity = calculate_perplexity(model, tokenizer, clean_text)
noisy_perplexity = calculate_perplexity(model, tokenizer, noisy_text)

print(f"Clean text perplexity: {clean_perplexity:.2f}")
print(f"Noisy text perplexity: {noisy_perplexity:.2f}")
```

2.2 Common data quality issues in language datasets:

```python
def analyze_text_quality(text):
    doc = nlp(text)
    
    # Check for spelling errors (using spaCy's built-in spell checker)
    misspelled = [token.text for token in doc if token._.is_misspelled]
    
    # Check for grammatical issues (simplistic approach using POS tags)
    pos_counts = Counter(token.pos_ for token in doc)
    grammar_score = pos_counts['NOUN'] + pos_counts['VERB'] + pos_counts['ADJ'] + pos_counts['ADV']
    
    # Check for sentence completeness
    incomplete_sentences = [sent.text for sent in doc.sents if len(sent) < 3]
    
    return {
        "misspelled_words": misspelled,
        "grammar_score": grammar_score,
        "incomplete_sentences": incomplete_sentences
    }

# Example usage
text = "This iz a smple txt with sum issues. Incomplet"
quality_report = analyze_text_quality(text)
print(quality_report)
```

2.3 Text preprocessing techniques for LLMs:

```python
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Join tokens back into text
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Example usage
raw_text = "This is an EXAMPLE of text preprocessing... It's quite useful!"
cleaned_text = preprocess_text(raw_text)
print(f"Original: {raw_text}")
print(f"Preprocessed: {cleaned_text}")
```

2.4 Handling multilingual and code-mixed data:

```python
def handle_multilingual_text(text):
    # Detect language
    try:
        lang = detect(text)
    except:
        lang = 'unknown'
    
    # Transliterate non-ASCII characters
    transliterated_text = unidecode(text)
    
    # Tokenize (using NLTK for simplicity, but consider language-specific tokenizers)
    tokens = word_tokenize(transliterated_text)
    
    return {
        'original': text,
        'language': lang,
        'transliterated': transliterated_text,
        'tokens': tokens
    }

# Example usage
texts = [
    "This is English text.",
    "Dies ist deutscher Text.",
    "これは日本語のテキストです。",
    "This is mixed language text avec un peu de français."
]

for text in texts:
    result = handle_multilingual_text(text)
    print(f"Original: {result['original']}")
    print(f"Detected Language: {result['language']}")
    print(f"Transliterated: {result['transliterated']}")
    print(f"Tokens: {result['tokens']}\n")
```

2.5 Deduplication strategies for large text corpora:

```python
def deduplicate_corpus(corpus, similarity_threshold=0.9):
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Compute pairwise similarities
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Find duplicates
    duplicates = set()
    for i in range(len(corpus)):
        for j in range(i + 1, len(corpus)):
            if similarity_matrix[i, j] > similarity_threshold:
                duplicates.add(j)
    
    # Create deduplicated corpus
    deduplicated_corpus = [doc for i, doc in enumerate(corpus) if i not in duplicates]
    
    return deduplicated_corpus

# Example usage
corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast auburn fox leaps above the sleepy canine.",
    "The quick brown fox jumps over the lazy dog.",
    "An entirely different sentence about cats.",
]

deduplicated = deduplicate_corpus(corpus)
print(f"Original corpus size: {len(corpus)}")
print(f"Deduplicated corpus size: {len(deduplicated)}")
print("Deduplicated corpus:")
for doc in deduplicated:
    print(f"- {doc}")
```

2.6 Automated data cleaning pipelines:

```python
class DataCleaningPipeline:
    def __init__(self, similarity_threshold=0.9, min_length=10, max_length=1000):
        self.similarity_threshold = similarity_threshold
        self.min_length = min_length
        self.max_length = max_length
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def preprocess(self, text):
        # Basic preprocessing
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = [word for word in text.split() if word not in stop_words]
        return ' '.join(tokens)
    
    def filter_by_length(self, df):
        return df[(df['text'].str.len() >= self.min_length) & (df['text'].str.len() <= self.max_length)]
    
    def deduplicate(self, df):
        tfidf_matrix = self.vectorizer.fit_transform(df['text'])
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        duplicates = set()
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                if similarity_matrix[i, j] > self.similarity_threshold:
                    duplicates.add(j)
        
        return df.drop(df.index[list(duplicates)])
    
    def clean(self, input_file, output_file):
        # Read data
        df = pd.read_csv(input_file)
        
        # Preprocess
        df['text'] = df['text'].apply(self.preprocess)
        
        # Filter by length
        df = self.filter_by_length(df)
        
        # Deduplicate
        df = self.deduplicate(df)
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        
        print(f"Cleaned data saved to {output_file}")

# Example usage
pipeline = DataCleaningPipeline()
pipeline.clean('input_data.csv', 'cleaned_data.csv')
```

2.7 Data validation and quality assurance:

```python
def validate_cleaned_data(file_path, sample_size=100):
    df = pd.read_csv(file_path)
    
    # Basic statistics
    print(f"Total samples: {len(df)}")
    print(f"Average text length: {df['text'].str.len().mean():.2f}")
    print(f"Unique samples: {df['text'].nunique()}")
    
    # Check for empty or very short texts
    short_texts = df[df['text'].str.len() < 10]
    print(f"Texts shorter than 10 characters: {len(short_texts)}")
    
    # Sample for manual review
    sample = df.sample(n=min(sample_size, len(df)))
    print("\nSample for manual review:")
    print(sample['text'].head())

    # Check for common issues
    common_issues = {
        'special_chars': df['text'].str.contains(r'[^a-zA-Z0-9\s]'),
        'numbers': df['text'].str.contains(r'\d'),
        'all_caps': df['text'].str.isupper()
    }
    
    for issue, mask in common_issues.items():
        print(f"Samples with {issue}: {mask.sum()}")
    
    # Evaluate impact on model perplexity
    model = GPT4LMHeadModel.from_pretrained('GPT4')
    tokenizer = GPT4Tokenizer.from_pretrained('GPT4')
    
    def calculate_perplexity(text):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = model(inputs, labels=inputs['input_ids'])
        return torch.exp(outputs.loss).item()
    
    sample_perplexities = sample['text'].apply(calculate_perplexity)
    print(f"\nAverage perplexity on sample: {sample_perplexities.mean():.2f}")

# Example usage
validate_cleaned_data('cleaned_data.csv')
```

These code snippets cover all the Python code examples provided in the document, organized by their respective section numbers.