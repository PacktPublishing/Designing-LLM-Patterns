# Chapter 5. Data Versioning for LLM Development

## 5.1 Understanding the Need for Data Versioning in LLM Projects

```python
from datetime import datetime
import hashlib
import json

class DatasetVersion:
    def __init__(self, data, metadata=None):
        self.data = data
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()
        self.version_hash = self._generate_hash()

    def _generate_hash(self):
        data_str = json.dumps(self.data, sort_keys=True).encode()
        return hashlib.sha256(data_str).hexdigest()

    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump({
                'data': self.data,
                'metadata': self.metadata,
                'timestamp': self.timestamp,
                'version_hash': self.version_hash
            }, f, indent=2)

    @classmethod
    def load(cls, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        instance = cls(data['data'], data['metadata'])
        instance.timestamp = data['timestamp']
        instance.version_hash = data['version_hash']
        return instance
```

## 5.2 Data Versioning Strategies for Large Language Datasets

```python
import difflib

class DeltaDatasetVersion(DatasetVersion):
    def __init__(self, data, base_version=None, metadata=None):
        super().__init__(data, metadata)
        self.base_version = base_version
        self.delta = self._compute_delta() if base_version else None

    def _compute_delta(self):
        base_data = json.dumps(self.base_version.data, sort_keys=True).splitlines()
        current_data = json.dumps(self.data, sort_keys=True).splitlines()
        diff = list(difflib.unified_diff(base_data, current_data, lineterm=''))
        return '\n'.join(diff)

    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump({
                'metadata': self.metadata,
                'timestamp': self.timestamp,
                'version_hash': self.version_hash,
                'base_version_hash': self.base_version.version_hash if self.base_version else None,
                'delta': self.delta
            }, f, indent=2)

    @classmethod
    def load(cls, filename, base_version):
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Apply delta to base version
        base_data = json.dumps(base_version.data, sort_keys=True).splitlines()
        patched_data = difflib.restore(base_data, data['delta'].splitlines(), 1)
        current_data = json.loads('\n'.join(patched_data))

        instance = cls(current_data, base_version, data['metadata'])
        instance.timestamp = data['timestamp']
        instance.version_hash = data['version_hash']
        instance.delta = data['delta']
        return instance
```

## 5.3 Tools for Data Versioning in LLM Development

```python
import subprocess

def initialize_dvc():
    subprocess.run(["dvc", "init"])
    print("DVC initialized in the current directory.")

def add_dataset_to_dvc(dataset_path):
    subprocess.run(["dvc", "add", dataset_path])
    print(f"Dataset {dataset_path} added to DVC.")

def commit_dataset_version(message):
    subprocess.run(["git", "add", ".dvc"])
    subprocess.run(["git", "commit", "-m", message])
    print(f"Dataset version committed with message: {message}")

def push_dataset_to_remote():
    subprocess.run(["dvc", "push"])
    subprocess.run(["git", "push"])
    print("Dataset pushed to remote storage.")

# Usage example
if __name__ == "__main__":
    initialize_dvc()
    add_dataset_to_dvc("path/to/your/large_language_dataset.txt")
    commit_dataset_version("Add initial version of language dataset")
    push_dataset_to_remote()
```

## 5.4 Integrating Data Versioning in LLM Training Workflows

```python
import json
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DatasetInfo:
    version_hash: str
    metadata: Dict[str, Any]

def load_dataset_info(filename: str) -> DatasetInfo:
    with open(filename, 'r') as f:
        data = json.load(f)
    return DatasetInfo(data['version_hash'], data['metadata'])

def train_llm(model, dataset, dataset_info: DatasetInfo):
    # Log dataset version information
    print(f"Training model with dataset version: {dataset_info.version_hash}")
    print(f"Dataset metadata: {dataset_info.metadata}")

    # Actual training code would go here
    # ...

    # Save model with dataset version information
    model.save(f"model_trained_on_{dataset_info.version_hash[:8]}.pt")

# Usage in training script
dataset_info = load_dataset_info("dataset_info.json")
dataset = load_dataset()  # Your dataset loading function
model = initialize_model()  # Your model initialization function

train_llm(model, dataset, dataset_info)
```

## 5.5 Version Control for Text Corpora

```python
import os
import hashlib
from typing import Dict, List

def hash_file(filepath: str) -> str:
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def generate_corpus_manifest(corpus_dir: str) -> Dict[str, str]:
    manifest = {}
    for root, _, files in os.walk(corpus_dir):
        for file in files:
            filepath = os.path.join(root, file)
            manifest[os.path.relpath(filepath, corpus_dir)] = hash_file(filepath)
    return manifest

def compare_manifests(old_manifest: Dict[str, str], new_manifest: Dict[str, str]) -> Dict[str, List[str]]:
    changes = {
        "added": [],
        "removed": [],
        "modified": []
    }
    
    for file, hash in new_manifest.items():
        if file not in old_manifest:
            changes["added"].append(file)
        elif old_manifest[file] != hash:
            changes["modified"].append(file)
    
    for file in old_manifest:
        if file not in new_manifest:
            changes["removed"].append(file)
    
    return changes

# Usage example
old_manifest = generate_corpus_manifest("path/to/old_corpus")
new_manifest = generate_corpus_manifest("path/to/new_corpus")
changes = compare_manifests(old_manifest, new_manifest)

print("Corpus changes:")
for change_type, files in changes.items():
    print(f"{change_type.capitalize()}:")
    for file in files:
        print(f"  - {file}")
```

## 5.6 Managing Dataset Variants and Experiments

```python
from typing import Dict, Any
import json
import os

class DatasetVariantManager:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.variants: Dict[str, Dict[str, Any]] = {}
        self._load_variants()

    def _load_variants(self):
        if os.path.exists(os.path.join(self.base_path, "variants.json")):
            with open(os.path.join(self.base_path, "variants.json"), 'r') as f:
                self.variants = json.load(f)

    def save_variants(self):
        with open(os.path.join(self.base_path, "variants.json"), 'w') as f:
            json.dump(self.variants, f, indent=2)

    def create_variant(self, name: str, base_variant: str, changes: Dict[str, Any]):
        if name in self.variants:
            raise ValueError(f"Variant {name} already exists")
        
        self.variants[name] = {
            "base": base_variant,
            "changes": changes
        }
        self.save_variants()

    def get_variant(self, name: str) -> Dict[str, Any]:
        if name not in self.variants:
            raise ValueError(f"Variant {name} does not exist")
        
        variant = self.variants[name]
        base_data = self.get_variant(variant["base"]) if variant["base"] else {}
        return {**base_data, **variant["changes"]}

# Usage example
manager = DatasetVariantManager("path/to/dataset/variants")

manager.create_variant("base", None, {"size": 1000000, "language": "en"})
manager.create_variant("large", "base", {"size": 5000000})
manager.create_variant("multilingual", "large", {"language": ["en", "es", "fr"]})

print(manager.get_variant("multilingual"))
```
