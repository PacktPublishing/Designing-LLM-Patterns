## 10 Checkpointing and Recovery for LLM Training

## 10.1 Importance of Checkpointing in LLM Training

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Config
import os

class LLMTrainer:
    def __init__(self, model, optimizer, checkpoint_dir='checkpoints'):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, epoch, step, loss):
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}_step_{step}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['step'], checkpoint['loss']

# Simulating training loop
for epoch in range(10):
    for step in range(1000):
        # ... training code ...
        if step % 100 == 0:
            trainer.save_checkpoint(epoch, step, loss.item())

# Loading a checkpoint
epoch, step, loss = trainer.load_checkpoint('checkpoints/checkpoint_epoch_5_step_500.pt')
print(f"Resumed training from epoch {epoch}, step {step}, with loss {loss}")
```

## 10.2 Checkpoint Frequency and Storage Strategies for LLMs

```python
import time
import shutil

class AdvancedLLMTrainer(LLMTrainer):
    def __init__(self, model, optimizer, checkpoint_dir='checkpoints', max_checkpoints=5):
        super().__init__(model, optimizer, checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []

    def save_checkpoint(self, epoch, step, loss):
        checkpoint_path = super().save_checkpoint(epoch, step, loss)
        self.checkpoints.append(checkpoint_path)
        
        if len(self.checkpoints) > self.max_checkpoints:
            oldest_checkpoint = self.checkpoints.pop(0)
            os.remove(oldest_checkpoint)
            print(f"Removed old checkpoint: {oldest_checkpoint}")

    def save_checkpoint_by_time(self, epoch, step, loss, interval_minutes=60):
        current_time = time.time()
        if not hasattr(self, 'last_checkpoint_time') or current_time - self.last_checkpoint_time >= interval_minutes * 60:
            self.save_checkpoint(epoch, step, loss)
            self.last_checkpoint_time = current_time

    def save_best_checkpoint(self, epoch, step, loss):
        if not hasattr(self, 'best_loss') or loss < self.best_loss:
            self.best_loss = loss
            checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'step': step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss
            }, checkpoint_path)
            print(f"Best model saved: {checkpoint_path}")

# Usage example
trainer = AdvancedLLMTrainer(model, optimizer)

for epoch in range(10):
    for step in range(1000):
        # ... training code ...
        trainer.save_checkpoint_by_time(epoch, step, loss.item(), interval_minutes=30)
        trainer.save_best_checkpoint(epoch, step, loss.item())
```

## 10.3 Efficient Checkpoint Formats for Large Language Models

```python
import torch
import io
import zipfile

class EfficientLLMTrainer(AdvancedLLMTrainer):
    def save_checkpoint_efficient(self, epoch, step, loss):
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}_step_{step}.zip')
        
        with zipfile.ZipFile(checkpoint_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for key, value in checkpoint.items():
                if isinstance(value, dict):  # For model and optimizer state_dicts
                    buffer = io.BytesIO()
                    torch.save(value, buffer)
                    zipf.writestr(f'{key}.pt', buffer.getvalue())
                else:
                    zipf.writestr(f'{key}.txt', str(value))
        
        print(f"Efficient checkpoint saved: {checkpoint_path}")

    def load_checkpoint_efficient(self, checkpoint_path):
        checkpoint = {}
        with zipfile.ZipFile(checkpoint_path, 'r') as zipf:
            for filename in zipf.namelist():
                if filename.endswith('.pt'):
                    with zipf.open(filename) as f:
                        key = filename[:-3]  # Remove .pt extension
                        checkpoint[key] = torch.load(io.BytesIO(f.read()))
                else:
                    with zipf.open(filename) as f:
                        key = filename[:-4]  # Remove .txt extension
                        value = f.read().decode('utf-8')
                        checkpoint[key] = int(value) if key in ['epoch', 'step'] else float(value)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['step'], checkpoint['loss']

# Usage
trainer = EfficientLLMTrainer(model, optimizer)
trainer.save_checkpoint_efficient(epoch, step, loss.item())
epoch, step, loss = trainer.load_checkpoint_efficient('checkpoints/checkpoint_epoch_5_step_500.zip')
```

## 10.4 Recovering from Failures in LLM Training

```python
import signal
import sys

class RobustLLMTrainer(EfficientLLMTrainer):
    def __init__(self, model, optimizer, checkpoint_dir='checkpoints', autosave_interval=15):
        super().__init__(model, optimizer, checkpoint_dir)
        self.autosave_interval = autosave_interval
        self.setup_signal_handlers()

    def setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self.handle_interrupt)
        signal.signal(signal.SIGTERM, self.handle_interrupt)

    def handle_interrupt(self, signum, frame):
        print("Interrupted! Saving checkpoint before exiting...")
        self.save_checkpoint_efficient(self.current_epoch, self.current_step, self.current_loss)
        sys.exit(0)

    def train(self, epochs, steps_per_epoch, train_fn):
        try:
            start_epoch, start_step = 0, 0
            latest_checkpoint = self.get_latest_checkpoint()
            if latest_checkpoint:
                start_epoch, start_step, _ = self.load_checkpoint_efficient(latest_checkpoint)
                print(f"Resuming from epoch {start_epoch}, step {start_step}")

            for epoch in range(start_epoch, epochs):
                self.current_epoch = epoch
                for step in range(start_step, steps_per_epoch):
                    self.current_step = step
                    self.current_loss = train_fn(self.model, epoch, step)
                    
                    if step % self.autosave_interval == 0:
                        self.save_checkpoint_efficient(epoch, step, self.current_loss)

                start_step = 0  # Reset step counter at the start of each epoch

        except Exception as e:
            print(f"Error occurred: {e}")
            print("Saving checkpoint before exiting...")
            self.save_checkpoint_efficient(self.current_epoch, self.current_step, self.current_loss)
            raise

    def get_latest_checkpoint(self):
        checkpoints = sorted(os.listdir(self.checkpoint_dir))
        return os.path.join(self.checkpoint_dir, checkpoints[-1]) if checkpoints else None

# Usage
def train_step(model, epoch, step):
    # Simulated training step
    loss = 1 / (epoch + 1 + step + 1)  # Dummy loss that decreases over time
    return loss

trainer = RobustLLMTrainer(model, optimizer)
trainer.train(epochs=10, steps_per_epoch=1000, train_fn=train_step)
```

## 10.5 Checkpointing in Distributed LLM Training

```python
import torch.distributed as dist

class DistributedLLMTrainer(RobustLLMTrainer):
    def __init__(self, model, optimizer, checkpoint_dir='checkpoints', autosave_interval=15):
        super().__init__(model, optimizer, checkpoint_dir, autosave_interval)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def save_checkpoint_distributed(self, epoch, step, loss):
        if self.rank == 0:  # Only the main process saves checkpoints
            self.save_checkpoint_efficient(epoch, step, loss)
        dist.barrier()  # Synchronize all processes

    def load_checkpoint_distributed(self, checkpoint_path):
        if self.rank == 0:
            epoch, step, loss = self.load_checkpoint_efficient(checkpoint_path)
        else:
            epoch, step, loss = 0, 0, 0.0

        # Broadcast the loaded data to all processes
        epoch = torch.tensor(epoch).to(self.rank)
        step = torch.tensor(step).to(self.rank)
        loss = torch.tensor(loss).to(self.rank)

        dist.broadcast(epoch, 0)
        dist.broadcast(step, 0)
        dist.broadcast(loss, 0)

        dist.barrier()  # Ensure all processes have loaded the checkpoint
```python
        return epoch.item(), step.item(), loss.item()

    def train_distributed(self, epochs, steps_per_epoch, train_fn):
        try:
            start_epoch, start_step = 0, 0
            if self.rank == 0:
                latest_checkpoint = self.get_latest_checkpoint()
                if latest_checkpoint:
                    start_epoch, start_step, _ = self.load_checkpoint_efficient(latest_checkpoint)
            
            # Broadcast the starting epoch and step to all processes
            start_epoch = torch.tensor(start_epoch).to(self.rank)
            start_step = torch.tensor(start_step).to(self.rank)
            dist.broadcast(start_epoch, 0)
            dist.broadcast(start_step, 0)
            start_epoch = start_epoch.item()
            start_step = start_step.item()

            if self.rank == 0:
                print(f"Resuming from epoch {start_epoch}, step {start_step}")

            for epoch in range(start_epoch, epochs):
                self.current_epoch = epoch
                for step in range(start_step, steps_per_epoch):
                    self.current_step = step
                    self.current_loss = train_fn(self.model, epoch, step)
                    
                    if step % self.autosave_interval == 0:
                        self.save_checkpoint_distributed(epoch, step, self.current_loss)

                start_step = 0  # Reset step counter at the start of each epoch

        except Exception as e:
            print(f"Error occurred on rank {self.rank}: {e}")
            self.save_checkpoint_distributed(self.current_epoch, self.current_step, self.current_loss)
            dist.destroy_process_group()
            raise

def init_distributed():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank

def distributed_train_step(model, epoch, step):
    # Simulated distributed training step
    loss = 1 / (epoch + 1 + step + 1)  # Dummy loss that decreases over time
    return loss

def main():
    rank = init_distributed()
    model = GPT2LMHeadModel(GPT2Config()).to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    trainer = DistributedLLMTrainer(model, optimizer)
    trainer.train_distributed(epochs=10, steps_per_epoch=1000, train_fn=distributed_train_step)

if __name__ == "__main__":
    main()
```

## 10.6 Version Control for LLM Checkpoints

```python
import os
import json
import shutil

class VersionControlledLLMTrainer(DistributedLLMTrainer):
    def __init__(self, model, optimizer, checkpoint_dir='checkpoints', version_file='versions.json'):
        super().__init__(model, optimizer, checkpoint_dir)
        self.version_file = version_file
        self.versions = self.load_versions()

    def load_versions(self):
        if os.path.exists(self.version_file):
            with open(self.version_file, 'r') as f:
                return json.load(f)
        return {}

    def save_versions(self):
        with open(self.version_file, 'w') as f:
            json.dump(self.versions, f, indent=2)

    def save_checkpoint_versioned(self, epoch, step, loss, version_name):
        checkpoint_path = self.save_checkpoint_efficient(epoch, step, loss)
        self.versions[version_name] = {
            'path': checkpoint_path,
            'epoch': epoch,
            'step': step,
            'loss': loss
        }
        self.save_versions()
        print(f"Saved version '{version_name}': {checkpoint_path}")

    def load_checkpoint_versioned(self, version_name):
        if version_name not in self.versions:
            raise ValueError(f"Version '{version_name}' not found")
        version_info = self.versions[version_name]
        return self.load_checkpoint_efficient(version_info['path'])

    def create_branch(self, base_version, new_version):
        if base_version not in self.versions:
            raise ValueError(f"Base version '{base_version}' not found")
        base_info = self.versions[base_version]
        new_path = f"{self.checkpoint_dir}/branch_{new_version}.pt"
        shutil.copy(base_info['path'], new_path)
        self.versions[new_version] = {
            'path': new_path,
            'epoch': base_info['epoch'],
            'step': base_info['step'],
            'loss': base_info['loss'],
            'branched_from': base_version
        }
        self.save_versions()
        print(f"Created branch '{new_version}' from '{base_version}'")

# Usage
trainer = VersionControlledLLMTrainer(model, optimizer)
trainer.save_checkpoint_versioned(epoch=10, step=500, loss=0.1, version_name="v1.0")
trainer.create_branch("v1.0", "experimental_branch")
epoch, step, loss = trainer.load_checkpoint_versioned("experimental_branch")
```

## 10.7 Automated Checkpointing and Recovery Systems

```python
import threading
import time

class AutomatedLLMTrainer(VersionControlledLLMTrainer):
    def __init__(self, model, optimizer, checkpoint_dir='checkpoints', autosave_interval=15, 
                 version_file='versions.json', health_check_interval=60):
        super().__init__(model, optimizer, checkpoint_dir, version_file)
        self.autosave_interval = autosave_interval
        self.health_check_interval = health_check_interval
        self.training_active = False

    def start_autosave_thread(self):
        def autosave_loop():
            while self.training_active:
                time.sleep(self.autosave_interval)
                if self.training_active:
                    self.save_checkpoint_versioned(self.current_epoch, self.current_step, 
                                                   self.current_loss, f"autosave_{time.time()}")
        self.autosave_thread = threading.Thread(target=autosave_loop)
        self.autosave_thread.start()

    def start_health_check_thread(self):
        def health_check_loop():
            while self.training_active:
                time.sleep(self.health_check_interval)
                if self.training_active:
                    if not self.check_system_health():
                        print("System health check failed. Initiating recovery...")
                        self.initiate_recovery()
        self.health_check_thread = threading.Thread(target=health_check_loop)
        self.health_check_thread.start()

    def check_system_health(self):
        # Implement system health checks here
        # For example, check GPU memory, CPU usage, disk space, etc.
        return True  # Return False if health check fails

    def initiate_recovery(self):
        # Implement recovery logic here
        # For example, reload from the last checkpoint, reduce batch size, etc.
        pass

    def train_with_automation(self, epochs, steps_per_epoch, train_fn):
        self.training_active = True
        self.start_autosave_thread()
        self.start_health_check_thread()

        try:
            super().train_distributed(epochs, steps_per_epoch, train_fn)
        finally:
            self.training_active = False
            self.autosave_thread.join()
            self.health_check_thread.join()

# Example usage
trainer = AutomatedLLMTrainer(model, optimizer)
trainer.train_with_automation(epochs=10, steps_per_epoch=1000, train_fn=distributed_train_step)
```



