
## 7 LLM Training Pipeline Design

## 7.1 Components of an LLM Training Pipeline

```python
# Data Ingestion and Preprocessing
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
import torch
from torch.nn import functional as F
import wandb

# Data Ingestion and Preprocessing
dataset = load_dataset("wikipedia", "20220301.en", split="train")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
```

```python
# Dataset Creation
train_dataloader = DataLoader(tokenized_dataset, shuffle=True, batch_size=8)

# Model Architecture
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Optimization
optimizer = AdamW(model.parameters(), lr=5e-5)
```

## 7.2 Training Loop

```python
# Training Loop
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_training_steps)

# Initialize wandb for logging
wandb.init(project="llm_training", name="gpt2_finetune")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch)
        loss = outputs.loss
        loss.backward()
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        wandb.log({"loss": loss.item()})

    # Evaluation
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch in train_dataloader:  # Using training data for simplicity
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch)
            eval_loss += outputs.loss.item()
    eval_loss /= len(train_dataloader)
    wandb.log({"eval_loss": eval_loss})

    # Checkpointing
    torch.save(model.state_dict(), f"model_checkpoint_epoch_{epoch}.pt")

wandb.finish()
```

## 7.3 Data Input and Preprocessing for LLM Training

```python
# Data Input and Preprocessing for LLM Training
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import numpy as np

# Load multiple datasets
wiki_dataset = load_dataset("wikipedia", "20220301.en", split="train")
books_dataset = load_dataset("bookcorpus", split="train")

# Combine datasets
combined_dataset = concatenate_datasets([wiki_dataset, books_dataset])

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def preprocess_function(examples):
    tokenized = tokenizer(examples["text"], truncation=True, max_length=1024)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    labels = [ids[1:] + [tokenizer.eos_token_id] for ids in input_ids]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# Apply preprocessing
tokenized_dataset = combined_dataset.map(preprocess_function, batched=True, remove_columns=combined_dataset.column_names, num_proc=4)

# Create DataLoader
train_dataloader = DataLoader(tokenized_dataset, shuffle=True, batch_size=16, collate_fn=lambda x: {k: np.stack([xi[k] for xi in x]) for k in x[0]})
```

## 7.4 LLM Architecture Design Considerations

```python
# LLM Architecture Design Considerations
from transformers import GPT2Config, GPT2LMHeadModel

# Define custom model configuration
config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_ctx=1024,
    n_embd=768,
    n_layer=12,
    n_head=12
)

# Initialize the model with custom configuration
model = GPT2LMHeadModel(config)
print(f"Model parameters: {model.num_parameters():,}")
```

## 7.5 Loss Functions and Optimization Strategies for LLMs

```python
# Loss Functions and Optimization Strategies for LLMs
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Loss function is typically handled by the model itself in Transformers

# Optimization
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

# Learning rate scheduler with warmup
num_epochs = 3
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        batch = {k: torch.tensor(v).to(device) for k, v in batch.items()}
        outputs = model(batch)
        loss = outputs.loss
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

## 7.6 Monitoring and Logging During LLM Training

```python
# Monitoring and Logging During LLM Training
from torch.utils.tensorboard import SummaryWriter
import time

# Initialize TensorBoard writer
writer = SummaryWriter()

# Training loop with monitoring
model.train()
total_loss = 0
log_interval = 100
start_time = time.time()

for i, batch in enumerate(train_dataloader):
    batch = {k: torch.tensor(v).to(device) for k, v in batch.items()}
    outputs = model(batch)
    loss = outputs.loss
    total_loss += loss.item()
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    
    if (i + 1) % log_interval == 0:
        cur_loss = total_loss / log_interval
        elapsed = time.time() - start_time
        writer.add_scalar('training_loss', cur_loss, global_step=i)
        writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], global_step=i)
        print(f'| epoch {epoch:3d} | {i:5d}/{len(train_dataloader):5d} batches | '
              f'lr {scheduler.get_last_lr()[0]:02.2f} | ms/batch {elapsed * 1000 / log_interval:5.2f} | '
              f'loss {cur_loss:5.2f}')
        total_loss = 0
        start_time = time.time()

writer.close()
```

## 7.7 Pipeline Modularity and Reusability

```python
# Pipeline Modularity and Reusability
class LLMTrainer:
    def __init__(self, model, train_dataloader, optimizer, scheduler, device):
        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.writer = SummaryWriter()
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        log_interval = 100
        start_time = time.time()
        
        for i, batch in enumerate(self.train_dataloader):
            batch = {k: torch.tensor(v).to(self.device) for k, v in batch.items()}
            outputs = self.model(batch)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            if (i + 1) % log_interval == 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                self.writer.add_scalar('training_loss', cur_loss, global_step=i)
                self.writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], global_step=i)
                print(f'| {i:5d}/{len(self.train_dataloader):5d} batches | '
                      f'lr {self.scheduler.get_last_lr()[0]:02.2f} | ms/batch {elapsed * 1000 / log_interval:5.2f} | '
                      f'loss {cur_loss:5.2f}')
                total_loss = 0
                start_time = time.time()
    
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            print(f'Starting epoch {epoch+1}')
            self.train_epoch()
        
        self.writer.close()

# Usage
```python
trainer = LLMTrainer(model, train_dataloader, optimizer, scheduler, device)
trainer.train(num_epochs=3)
```

## 7.8 Scaling Your Training Pipeline for Larger Models

```python
# Scaling Your Training Pipeline for Larger Models
import torch.cuda.amp as amp

class LargeScaleLLMTrainer(LLMTrainer):
    def __init__(self, model, train_dataloader, optimizer, scheduler, device, accumulation_steps=4):
        super().__init__(model, train_dataloader, optimizer, scheduler, device)
        self.accumulation_steps = accumulation_steps
        self.scaler = amp.GradScaler()
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        log_interval = 100
        start_time = time.time()
        
        for i, batch in enumerate(self.train_dataloader):
            batch = {k: torch.tensor(v).to(self.device) for k, v in batch.items()}
            
            with amp.autocast():
                outputs = self.model(batch)
                loss = outputs.loss / self.accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            if (i + 1) % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.accumulation_steps
            
            if (i + 1) % log_interval == 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                self.writer.add_scalar('training_loss', cur_loss, global_step=i)
                self.writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], global_step=i)
                print(f'| {i:5d}/{len(self.train_dataloader):5d} batches | '
                      f'lr {self.scheduler.get_last_lr()[0]:02.2f} | ms/batch {elapsed * 1000 / log_interval:5.2f} | '
                      f'loss {cur_loss:5.2f}')
                total_loss = 0
                start_time = time.time()

# Usage
large_trainer = LargeScaleLLMTrainer(model, train_dataloader, optimizer, scheduler, device)
large_trainer.train(num_epochs=3)
```
