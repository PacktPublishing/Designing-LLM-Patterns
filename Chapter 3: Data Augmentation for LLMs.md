
# Chapter 3 Data Augmentation for LLMs

## Text data augmentation techniques

### Synonym replacement

```python
def synonym_replacement(text, n=1):
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word.isalnum()]))
    random.shuffle(random_word_list)
    num_replaced = 0
    
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    return ' '.join(new_words)
```

### Back-translation

```python
def back_translation(text, target_lang='fr'):
    translator = Translator()
    translated = translator.translate(text, dest=target_lang)
    back_translated = translator.translate(translated.text, dest='en')
    return back_translated.text
```

### Text generation with T5

```python
def t5_augmentation(text, model, tokenizer, num_return_sequences=1):
    input_ids = tokenizer.encode(f"paraphrase: {text}", return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        input_ids=input_ids,
        max_length=150,
        num_return_sequences=num_return_sequences,
        num_beams=5,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
    )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
```

## Leveraging existing LLMs for data generation

```python
def gpt4o_data_generation(prompt, num_samples=5):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        n=num_samples,
        temperature=0.7,
    )
    return [choice.message.content.strip() for choice in response.choices]
```

## Multilingual data augmentation strategies

### Cross-lingual back-translation

```python
def cross_lingual_back_translation(text, target_langs=['fr', 'de', 'es']):
    translator = Translator()
    augmented_texts = []
    for lang in target_langs:
        translated = translator.translate(text, dest=lang)
        back_translated = translator.translate(translated.text, dest='en')
        augmented_texts.append(back_translated.text)
    return augmented_texts
```

### Multilingual T5 augmentation

```python
def multilingual_t5_augmentation(text, model, tokenizer, target_langs=['fr', 'de', 'es']):
    augmented_texts = []
    for lang in target_langs:
        input_ids = tokenizer.encode(f"translate English to {lang}: {text}", return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(input_ids=input_ids, max_length=150)
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        augmented_texts.append(translated)
    return augmented_texts
```

## Semantic preservation in text augmentation

### Use of sentence embeddings

```python
def semantic_similarity(original, augmented, model):
    original_embedding = model.encode(original)
    augmented_embedding = model.encode(augmented)
    similarity = cosine_similarity([original_embedding], [augmented_embedding])[0][0]
    return similarity

def filter_by_semantic_similarity(original, augmented_list, model, threshold=0.8):
    return [aug for aug in augmented_list if semantic_similarity(original, aug, model) >= threshold]
```

### Contextual word embeddings for synonym replacement

```python
def contextual_synonym_replacement(text, model, tokenizer, n=1):
    words = text.split()
    new_words = words.copy()
    
    for i in range(n):
        word_index = random.randint(0, len(words) - 1)
        original_word = words[word_index]
        
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(inputs)
        
        word_embedding = outputs.last_hidden_state[0, word_index]
        similar_words = find_similar_words(word_embedding, model, tokenizer)
        
        if similar_words:
            new_words[word_index] = random.choice(similar_words)
    
    return ' '.join(new_words)
```

## Balancing augmentation and data quality

### Quality filtering

```python
def quality_filter(augmented_texts, original_text, similarity_threshold=0.8, perplexity_threshold=100):
    filtered_texts = []
    for aug_text in augmented_texts:
        if (semantic_similarity(original_text, aug_text, similarity_model) >= similarity_threshold and
            calculate_perplexity(aug_text, perplexity_model) <= perplexity_threshold):
            filtered_texts.append(aug_text)
    return filtered_texts
```

### Human-in-the-loop validation

```python
def human_validation(augmented_texts):
    validated_texts = []
    for text in augmented_texts:
        if input(f"Is this text valid? (y/n)\n{text}\n").lower() == 'y':
            validated_texts.append(text)
    return validated_texts
```

## Evaluating the impact of data augmentation

### Perplexity

```python
def evaluate_perplexity(model, tokenizer, test_data):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in test_data:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            outputs = model(inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)
    
    perplexity = math.exp(total_loss / total_tokens)
    return perplexity
```

### Task-specific metrics

```python
def evaluate_classification(model, tokenizer, test_data, test_labels):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for text in test_data:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            outputs = model(inputs)
            predictions.append(torch.argmax(outputs.logits).item())
    
    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, average='weighted')
    return accuracy, f1
```

### Diversity metrics

```python
def calculate_diversity_metrics(texts):
    all_words = [word for text in texts for word in text.split()]
    vocab_size = len(set(all_words))
    
    all_trigrams = [text[i:i+3] for text in texts for i in range(len(text)-2)]
    unique_trigrams = len(set(all_trigrams))
    
    return {
        "vocabulary_size": vocab_size,
        "unique_trigrams": unique_trigrams
    }
```
