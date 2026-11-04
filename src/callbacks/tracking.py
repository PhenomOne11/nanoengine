import torch
from src.callbacks.base import Callback

class TrackLossCallback(Callback):
    order=10
    
    def before_fit(self):
        self.train_losses=[]
        self.valid_losses=[]
    
    def before_epoch(self):
        self.batch_losses=[]
        
    def after_loss(self):
        loss_val=self.loss.detach().item() if isinstance(self.loss, torch.Tensor) else float(self.loss)
        self.batch_losses.append(loss_val)
        
    def after_epoch(self):
        avg_loss=sum(self.batch_losses)/max(len(self.batch_losses),1)
        if self.training:
            self.train_losses.append(avg_loss)
            print(f"Epoch {self.epoch + 1}/{self.epochs} - Train Loss: {avg_loss:.4f}")
        else:
            self.valid_losses.append(avg_loss)
            print(f"Epoch {self.epoch + 1}/{self.epochs} - Valid Loss: {avg_loss:.4f}")
            
class AccuracyCallback(Callback):
    order=20
    
    def before_epoch(self):
        self.correct=0
        self.total=0
        
    def after_pred(self):
        preds_class = self.preds.argmax(dim=-1)
        self.correct += (preds_class == self.yb).sum().item()
        self.total += self.yb.shape[0]
            
    def after_epoch(self):
        if not self.training and self.total > 0:
            acc = (self.correct / self.total) * 100.0
            print(f"Epoch {self.epoch + 1}/{self.epochs} - Valid Accuracy: {acc:.2f}%")