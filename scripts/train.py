import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from src.models.resnet import ResNet9
from src.optim.adamw import AdamW
from src.optim.lr_scheduler import CosineAnnealingLR
from scripts.download_cifar import get_cifar_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description="Train custom ResNet-9 on CIFAR-10")
    parser.add_argument("--epochs", "--epoch", type=int, default=20, dest="epochs", help="Number of training epochs")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=128, dest="batch_size", help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Max learning rate")
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=1e-2, dest="weight_decay", help="Decoupled weight decay for AdamW")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--output-dir", "--output_dir", type=str, default="weights", dest="output_dir", help="Directory to save checkpoint")
    return parser.parse_args()

def train_one_epoch(model, train_loader, criterion, optimizer, schedular, device):
    model.train()
    running_loss=0.0
    correct=0
    total=0
    
    for xb,yb in train_loader:
        xb,yb=xb.to(device), yb.to(device)
        
        optimizer.zero_grad()
        preds=model(xb)
        loss=criterion(preds,yb)
        loss.backward()
        optimizer.step()
        schedular.step()
        
        running_loss+=loss.item() * xb.size(0)
        _, predicted =preds.max(1)
        total+=yb.size(0)
        correct+=predicted.eq(yb).sum().item()
    epoch_loss=running_loss/total
    epoch_acc=100.0 * correct/total
    return epoch_loss,epoch_acc

def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss=0
    correct=0
    total=0
    
    with torch.no_grad():
        for xb,yb in test_loader:
            xb,yb=xb.to(device), yb.to(device)
            preds=model(xb)
            loss=criterion(preds, yb)
            
            running_loss+=loss.item() * xb.size(0)
            _, predicted=preds.max(1)
            total+=yb.size(0)
            correct+=predicted.eq(yb).sum().item()
    val_loss = running_loss / total
    val_acc = 100.0 * correct / total
    return val_loss, val_acc

def main():
    args=parse_args()
    device=torch.device(args.device)
    print(f"[*] Training on target device: {device}")
    
    train_loader, test_loader=get_cifar_dataloaders(batch_size=args.batch_size)
    
    model=ResNet9(num_classes=10).to(device)
    
    criterion=nn.CrossEntropyLoss()
    optimizer=AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    total_steps=args.epochs*len(train_loader)
    schedular=CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    
    output_path=Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    best_acc=0
    checkpoint_file=output_path/"cifar10_resnet.pt"
    
    print(f"[*] Starting training for {args.epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(1,args.epochs+1):
        epoch_start=time.time()
        train_loss, train_acc=train_one_epoch(model, train_loader,criterion,optimizer, schedular, device)
        val_loss, val_acc=evaluate(model, test_loader, criterion, device)
        epoch_duration=time.time()-epoch_start
        
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_duration:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {optimizer.lr:.6f}"
        )
        
        if val_acc>best_acc:
            best_acc=val_acc
            torch.save(model.state_dict(), checkpoint_file)
            
    total_duration = time.time() - start_time
    print(f"\n[+] Training completed in {total_duration / 60:.2f} minutes.")
    print(f"[+] Peak Validation Accuracy: {best_acc:.2f}%")
    print(f"[+] Best model checkpoint saved to: {checkpoint_file}")
    
    if best_acc >= 90.0:
        print("[SUCCESS] Stage 6 Exit Gate Passed: Validation Accuracy >= 90.0% achieved!")
    else:
        print("[WARNING] Target 90.0% not reached. Consider training for 25 epochs or tuning learning rate.")

if __name__ == "__main__":
    main()
    