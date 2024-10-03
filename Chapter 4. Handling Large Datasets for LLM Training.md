
# Chapter 4. Handling Large Datasets for LLM Training

## 4.1 Challenges of large datasets

```python
from datasets import load_dataset, Dataset
import psutil

def load_and_process_large_dataset(dataset_name, num_proc):
    # Load the dataset
    dataset = load_dataset(dataset_name, streaming=True)
    
    # Define a preprocessing function
    def preprocess_function(examples):
        # Implement your preprocessing logic here
        return examples
    
    # Apply preprocessing in parallel
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset["train"].column_names
    )
    
    return processed_dataset

# Determine the number of CPU cores for parallel processing
num_cores = psutil.cpu_count(logical=False)

# Load and process a large dataset (e.g., C4 dataset)
large_dataset = load_and_process_large_dataset("c4", num_proc=num_cores)

# Print the first few examples
for example in large_dataset["train"].take(5):
    print(example)
```

### 4.1.1 GPU-accelerated processing

```python
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

def load_and_process_dataset(dataset_name, batch_size):
    dataset = load_dataset(dataset_name, streaming=True)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def preprocess(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors="pt")
    
    def process_batch(batch):
        return {k: v.to(device) for k, v in preprocess(batch).items()}
    
    return DataLoader(dataset["train"].map(process_batch), batch_size=batch_size, num_workers=2, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataloader = load_and_process_dataset("c4", batch_size=32)

for i, batch in enumerate(dataloader):
    if i >= 5: break
    print(f"Batch {i}:", {k: v.shape for k, v in batch.items()})
```

## 4.2 Data sampling techniques

```python
import numpy as np
from datasets import Dataset

def stratified_length_sampling(dataset, num_samples, num_strata=10):
    # Calculate text lengths
    lengths = [len(example['text']) for example in dataset]
    
    # Create strata based on text length
    strata_bounds = np.percentile(lengths, np.linspace(0, 100, num_strata + 1))
    
    sampled_data = []
    for i in range(num_strata):
        stratum = [
            example for example in dataset 
            if strata_bounds[i] <= len(example['text']) < strata_bounds[i+1]
        ]
        stratum_samples = np.random.choice(
            stratum, 
            size=num_samples // num_strata, 
            replace=False
        )
        sampled_data.extend(stratum_samples)
    
    return Dataset.from_dict({
        key: [example[key] for example in sampled_data]
        for key in dataset[0].keys()
    })

# Usage
sampled_dataset = stratified_length_sampling(large_dataset, num_samples=100000)
```

## 4.3 Distributed data processing

```python
import dask.dataframe as dd
from dask.distributed import Client

def distributed_preprocessing(data_path, num_partitions):
    # Initialize Dask client
    client = Client()
    
    # Read the dataset into a Dask DataFrame
    df = dd.read_csv(data_path, blocksize="64MB")
    
    # Repartition the data for better distribution
    df = df.repartition(npartitions=num_partitions)
    
    # Define preprocessing function
    def preprocess(text):
        # Implement your preprocessing logic here
        return processed_text
    
    # Apply preprocessing in parallel
    df['processed_text'] = df['text'].map(preprocess)
    
    # Trigger computation and return results
    result = df.compute()
    
    client.close()
    return result

# Usage
processed_data = distributed_preprocessing("path/to/large/dataset.csv", num_partitions=100)
```

## 4.4 Efficient data storage formats

```python
import pyarrow as pa
import pyarrow.parquet as pq

def convert_to_parquet(dataset, output_path):
    # Convert dataset to Arrow Table
    table = pa.Table.from_pydict(dataset[0])
    
    # Write to Parquet file
    pq.write_table(table, output_path)

def read_from_parquet(file_path):
    # Read Parquet file
    table = pq.read_table(file_path)
    
    # Convert back to dictionary
    return table.to_pydict()

# Usage
convert_to_parquet(large_dataset, "large_dataset.parquet")
loaded_dataset = read_from_parquet("large_dataset.parquet")
```

## 4.5 Streaming data processing for continuous LLM training

```python
import faust

class Text(faust.Record):
    content: str

app = faust.App('llm-training', broker='kafka://localhost:9092')
topic = app.topic('raw-text', value_type=Text)

@app.agent(topic)
async def process(stream):
    async for text in stream:
        processed_text = preprocess(text.content)
        # Here you would typically send the processed text to your LLM training pipeline
        print(f"Processed: {processed_text}")

if __name__ == '__main__':
    app.main()
```

## 4.6 Data sharding and parallelization strategies

```python
import hashlib

def shard_data(dataset, num_shards):
    shards = [[] for _ in range(num_shards)]
    
    for item in dataset:
        # Use a hash function to determine the shard
        shard_index = int(hashlib.md5(item['id'].encode()).hexdigest(), 16) % num_shards
        shards[shard_index].append(item)
    
    return shards

# Usage
sharded_data = shard_data(large_dataset, num_shards=10)
```

## 4.7 Memory-efficient data loading techniques

```python
import numpy as np

def create_memmap_dataset(dataset, output_file):
    # Determine the shape of the dataset
    num_samples = len(dataset)
    sample_shape = dataset[0]['input'].shape
    
    # Create a memory-mapped array
    mmap = np.memmap(output_file, dtype='float32', mode='w+', shape=(num_samples, *sample_shape))
    
    # Write data to the memory-mapped array
    for i, sample in enumerate(dataset):
        mmap[i] = sample['input']
    
    # Flush to disk
    mmap.flush()

def load_memmap_dataset(file_path, shape):
    # Load the memory-mapped array
    return np.memmap(file_path, dtype='float32', mode='r', shape=shape)

# Usage
create_memmap_dataset(large_dataset, "large_dataset.mmap")
mmap_dataset = load_memmap_dataset("large_dataset.mmap", shape=(len(large_dataset), *large_dataset[0]['input'].shape))
```
