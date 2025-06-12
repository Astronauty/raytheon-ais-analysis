from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch
from tqdm import tqdm

class GPKernelShipClassificationTrainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
        log_dir = f"logs/kernel_classification/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self.writer = SummaryWriter(log_dir=log_dir)

    def train(self, num_epochs=20):
        for epoch in tqdm(range(num_epochs), desc="GP Kernel Ship Classification Training"):
            # Train
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for mmsi, kernel_params, group_id in self.train_loader:
                kernel_params = kernel_params.to(self.device)
                group_id = group_id.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(kernel_params)
                loss = self.criterion(outputs, group_id)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * kernel_params.size(0)
                _, predicted = outputs.max(1)
                correct += (predicted == group_id).sum().item()
                total += group_id.size(0)
                
            train_loss = running_loss / total
            train_acc = correct / total
            
            # Eval
            self.model.eval()
            test_loss = 0
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for mmsi, kernel_params, group_id in self.test_loader:
                    kernel_params = kernel_params.to(self.device)
                    group_id = group_id.to(self.device)
                    outputs = self.model(kernel_params)
                    loss = self.criterion(outputs, group_id)
                    test_loss += loss.item() * kernel_params.size(0)
                    _, predicted = outputs.max(1)
                    test_correct += (predicted == group_id).sum().item()
                    test_total += group_id.size(0)
            
            test_loss = test_loss / test_total
            test_acc = test_correct / test_total
            
        # --- Logging ---
        print(f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        self.writer.add_scalar("Loss/train", train_loss, epoch)
        self.writer.add_scalar("Accuracy/train", train_acc, epoch)
        self.writer.add_scalar("Loss/test", test_loss, epoch)
        self.writer.add_scalar("Accuracy/test", test_acc, epoch)
        
        self.writer.flush()
        self.writer.close()
