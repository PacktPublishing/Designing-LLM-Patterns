
## 8 Hyperparameter Tuning for LLMs

## 8.1 Understanding Hyperparameters in LLMs

```python
from transformers import GPT2Config, GPT2LMHeadModel

def create_llm(num_layers, hidden_size, num_heads, ff_dim, vocab_size):
    config = GPT2Config(
        n_layer=num_layers,
        n_embd=hidden_size,
        n_head=num_heads,
        n_inner=ff_dim,
        vocab_size=vocab_size
    )
    model = GPT2LMHeadModel(config)
    return model

# Example usage
model = create_llm(num_layers=12, hidden_size=768, num_heads=12, ff_dim=3072, vocab_size=50257)
print(f"Model parameters: {model.num_parameters():,}")
```

## 8.2 Manual vs. Automated Tuning for Large Language Models

### Manual Tuning

```python
import numpy as np
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Load a sample dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# Manual Tuning
manual_hyperparameters = [
    {"num_layers": 6, "hidden_size": 512, "num_heads": 8, "ff_dim": 2048},
    {"num_layers": 12, "hidden_size": 768, "num_heads": 12, "ff_dim": 3072},
    {"num_layers": 24, "hidden_size": 1024, "num_heads": 16, "ff_dim": 4096}
]

for hp in manual_hyperparameters:
    model = create_llm(hp["num_layers"], hp["hidden_size"], hp["num_heads"], hp["ff_dim"], vocab_size=50257)
    
    training_args = TrainingArguments(
        output_dir=f"./results_{hp['num_layers']}_{hp['hidden_size']}",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        logging_dir=f"./logs_{hp['num_layers']}_{hp['hidden_size']}",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    trainer.train()
    
    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Hyperparameters: {hp}")
    print(f"Evaluation results: {eval_results}")
```

### Random Search for Automated Tuning

```python
import random

def random_hp_search(num_trials=10):
    best_eval_loss = float('inf')
    best_hp = None
    
    for _ in range(num_trials):
        hp = {
            "num_layers": random.choice([6, 12, 24]),
            "hidden_size": random.choice([512, 768, 1024]),
            "num_heads": random.choice([8, 12, 16]),
            "ff_dim": random.choice([2048, 3072, 4096])
        }
        
        model = create_llm(hp["num_layers"], hp["hidden_size"], hp["num_heads"], hp["ff_dim"], vocab_size=50257)
        
        training_args = TrainingArguments(
            output_dir=f"./results_random_{_}",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            logging_dir=f"./logs_random_{_}",
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )
        
        trainer.train()
        
        eval_results = trainer.evaluate()
        eval_loss = eval_results['eval_loss']
        
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_hp = hp
        
        print(f"Trial {_+1}: Hyperparameters: {hp}, Eval Loss: {eval_loss}")
    
    print(f"Best Hyperparameters: {best_hp}, Best Eval Loss: {best_eval_loss}")

random_hp_search()
```

## 8.3 Grid Search and Random Search for LLM Hyperparameters

### Grid Search

```python
import itertools

def grid_search():
    hp_grid = {
        "num_layers": [6, 12, 24],
        "hidden_size": [512, 768, 1024],
        "num_heads": [8, 12, 16],
        "ff_dim": [2048, 3072, 4096]
    }
    
    best_eval_loss = float('inf')
    best_hp = None
    
    for hp in itertools.product(*hp_grid.values()):
        hp_dict = dict(zip(hp_grid.keys(), hp))
        
        model = create_llm(hp_dict["num_layers"], hp_dict["hidden_size"], hp_dict["num_heads"], hp_dict["ff_dim"], vocab_size=50257)
        
        training_args = TrainingArguments(
            output_dir=f"./results_grid_{hp_dict['num_layers']}_{hp_dict['hidden_size']}",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            logging_dir=f"./logs_grid_{hp_dict['num_layers']}_{hp_dict['hidden_size']}",
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )
        
        trainer.train()
        
        eval_results = trainer.evaluate()
        eval_loss = eval_results['eval_loss']
        
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_hp = hp_dict
        
        print(f"Hyperparameters: {hp_dict}, Eval Loss: {eval_loss}")
    
    print(f"Best Hyperparameters: {best_hp}, Best Eval Loss: {best_eval_loss}")

grid_search()
```

## 8.4 Bayesian Optimization for LLM Tuning

```python
import optuna

def objective(trial):
    # Define the hyperparameters to optimize
    hp = {
        "num_layers": trial.suggest_int("num_layers", 6, 24),
        "hidden_size": trial.suggest_categorical("hidden_size", [512, 768, 1024]),
        "num_heads": trial.suggest_categorical("num_heads", [8, 12, 16]),
        "ff_dim": trial.suggest_categorical("ff_dim", [2048, 3072, 4096]),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "num_epochs": trial.suggest_int("num_epochs", 2, 5),
        "warmup_steps": trial.suggest_int("warmup_steps", 100, 1000),
        "weight_decay": trial.suggest_uniform("weight_decay", 0, 0.2)
    }
    
    model = create_llm(hp["num_layers"], hp["hidden_size"], hp["num_heads"], hp["ff_dim"], vocab_size=50257)
    
    training_args = TrainingArguments(
        output_dir=f"./results_bayesian_{trial.number}",
        num_train_epochs=hp["num_epochs"],
        per_device_train_batch_size=hp["batch_size"],
        learning_rate=hp["learning_rate"],
        warmup_steps=hp["warmup_steps"],
        weight_decay=hp["weight_decay"],
        logging_dir=f"./logs_bayesian_{trial.number}",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    trainer.train()
    
    eval_results = trainer.evaluate()
    eval_loss = eval_results['eval_loss']
    return eval_loss

# Run the optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

print("Best trial:")
trial = study.best_trial
print(f"Value: {trial.value}")
print("Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
```

## 8.5 Population-Based Methods for LLM Optimization

```python
import random
import copy

class SimplePBT:
    def __init__(self, population_size=4, num_generations=5):
        self.population_size = population_size
        self.num_generations = num_generations
        self.population = []

    def initialize_population(self):
        for _ in range(self.population_size):
            hp = {
                "num_layers": random.choice([6, 12, 24]),
                "hidden_size": random.choice([512, 768, 1024]),
                "num_heads": random.choice([8, 12, 16]),
                "ff_dim": random.choice([2048, 3072, 4096]),
                "learning_rate": 10**random.uniform(-5, -3),
                "batch_size": random.choice([8, 16, 32]),
                "weight_decay": random.uniform(0, 0.2)
            }
            self.population.append({"hp": hp, "score": None})

    def train_and_evaluate(self, hp):
        model = create_llm(num_layers=hp["num_layers"], hidden_size=hp["hidden_size"], 
                           num_heads=hp["num_heads"], ff_dim=hp["ff_dim"], vocab_size=50257)
        
        training_args = TrainingArguments(
            output_dir=f"./results_pbt_{random.randint(0, 1000)}",
            ```python
            num_train_epochs=3,
            per_device_train_batch_size=hp["batch_size"],
            learning_rate=hp["learning_rate"],
            weight_decay=hp["weight_decay"],
            logging_dir=f"./logs_pbt_{random.randint(0, 1000)}",
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )
        
        trainer.train()
        
        eval_results = trainer.evaluate()
        return eval_results["eval_loss"]

    def exploit_and_explore(self):
        # Sort population by score
        self.population.sort(key=lambda x: x['score'])
        
        # Replace bottom half with mutated versions of top half
        for i in range(self.population_size // 2):
            self.population[i + self.population_size // 2]["hp"] = self.mutate(copy.deepcopy(self.population[i]["hp"]))

    def mutate(self, hp):
        # Randomly mutate one hyperparameter
        param_to_mutate = random.choice(list(hp.keys()))
        if param_to_mutate in ["num_layers", "hidden_size", "num_heads", "ff_dim", "batch_size"]:
            hp[param_to_mutate] = random.choice([6, 12, 24] if param_to_mutate == "num_layers" else 
                                                [512, 768, 1024] if param_to_mutate == "hidden_size" else
                                                [8, 12, 16] if param_to_mutate == "num_heads" else
                                                [2048, 3072, 4096] if param_to_mutate == "ff_dim" else
                                                [8, 16, 32])
        elif param_to_mutate == "learning_rate":
            hp[param_to_mutate] *= random.uniform(0.8, 1.2)
        elif param_to_mutate == "weight_decay":
            hp[param_to_mutate] = min(max(hp[param_to_mutate] + random.uniform(-0.05, 0.05), 0), 0.2)
        return hp

    def run(self):
        self.initialize_population()
        
        for generation in range(self.num_generations):
            print(f"Generation {generation + 1}")
            
            for i, individual in enumerate(self.population):
                individual["score"] = self.train_and_evaluate(individual["hp"])
                print(f"Individual {i + 1}: Score = {individual['score']}")
            
            self.exploit_and_explore()
        
        best_individual = min(self.population, key=lambda x: x["score"])
        print("\nBest Hyperparameters:")
        print(best_individual["hp"])
        print(f"Best Score: {best_individual['score']}")

# Run PBT
pbt = SimplePBT()
pbt.run()
```

## 8.6 Multi-Objective Hyperparameter Optimization for LLMs

```python
import optuna

def objective(trial):
    hp = {
        "num_layers": trial.suggest_int("num_layers", 6, 24),
        "hidden_size": trial.suggest_categorical("hidden_size", [512, 768, 1024]),
        "num_heads": trial.suggest_categorical("num_heads", [8, 12, 16]),
        "ff_dim": trial.suggest_categorical("ff_dim", [2048, 3072, 4096]),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "weight_decay": trial.suggest_uniform("weight_decay", 0, 0.2)
    }
    
    model = create_llm(num_layers=hp["num_layers"], hidden_size=hp["hidden_size"], 
                       num_heads=hp["num_heads"], ff_dim=hp["ff_dim"], vocab_size=50257)
    
    training_args = TrainingArguments(
        output_dir=f"./results_multi_objective_{trial.number}",
        num_train_epochs=3,
        per_device_train_batch_size=hp["batch_size"],
        learning_rate=hp["learning_rate"],
        weight_decay=hp["weight_decay"],
        logging_dir=f"./logs_multi_objective_{trial.number}",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    trainer.train()
    
    eval_results = trainer.evaluate()
    eval_loss = eval_results["eval_loss"]
    
    # Calculate model size in MB
    model_size = sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024  # assuming float32
    
    # Simulate inference time (this would be more accurate if actually measured)
    inference_time = 0.001 * hp["num_layers"] * (hp["hidden_size"] / 512) ** 2
    
    return eval_loss, model_size, inference_time

# Run the multi-objective optimization
study = optuna.create_study(directions=["minimize", "minimize", "minimize"])
study.optimize(objective, n_trials=50)

print("Pareto front:")
for trial in study.best_trials:
    print(f"Trial {trial.number}")
    print(f"  Value: Loss={trial.values[0]:.4f}, Size={trial.values[1]:.2f}MB, Inference Time={trial.values[2]:.4f}s")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
```

## 8.7 Hyperparameter Tuning at Scale: Challenges and Solutions

### Multi-Fidelity Optimization Example

```python
import optuna

def objective(trial):
    hp = {
        "num_layers": trial.suggest_int("num_layers", 6, 24),
        "hidden_size": trial.suggest_categorical("hidden_size", [512, 768, 1024]),
        "num_heads": trial.suggest_categorical("num_heads", [8, 12, 16]),
        "ff_dim": trial.suggest_categorical("ff_dim", [2048, 3072, 4096]),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "weight_decay": trial.suggest_uniform("weight_decay", 0, 0.2)
    }
    
    model = create_llm(num_layers=hp["num_layers"], hidden_size=hp["hidden_size"], 
                       num_heads=hp["num_heads"], ff_dim=hp["ff_dim"], vocab_size=50257)
    
    # Multi-fidelity: start with a small number of steps
    for steps in [100, 500, 2000]:
        training_args = TrainingArguments(
            output_dir=f"./results_multi_fidelity_{trial.number}_{steps}",
            max_steps=steps,
            per_device_train_batch_size=hp["batch_size"],
            learning_rate=hp["learning_rate"],
            weight_decay=hp["weight_decay"],
            logging_dir=f"./logs_multi_fidelity_{trial.number}_{steps}",
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )
        
        trainer.train()
        
        eval_results = trainer.evaluate()
        eval_loss = eval_results['eval_loss']
        
        trial.report(eval_loss, step=steps)
        
        # Prune unpromising trials
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return eval_loss

# Run the multi-fidelity optimization
study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=30)

print("Best trial:")
trial = study.best_trial
print(f"Value: {trial.value}")
print("Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
```
