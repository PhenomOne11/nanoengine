import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from src.layers.linear import Linear
from src.layers.activations import ReLU
from src.layers.containers import Sequential
from src.optim.sgd import SGD
from src.callbacks.runner import Learner
from src.callbacks.tracking import TrackLossCallback

def test_stage5_learner_pipeline():
    torch.manual_seed(42)
    
    xb=torch.randn(16,10)
    yb=torch.randn(16,2)
    dummy_yl=[(xb,yb) for _ in range(4)]
    dls=(dummy_yl,dummy_yl)
    
    model=Sequential(
        Linear(10,16),
        ReLU(),
        Linear(16,2)
    )
    
    opt=SGD(model.parameters(), lr=0.01)
    
    loss_fn=nn.MSELoss()
    tracker=TrackLossCallback()
    Learn=Learner(model, dls, loss_fn, opt, cbs=[tracker])
    
    Learn.fit(epochs=1)
    
    assert len(tracker.train_losses) == 1, "TrackLossCallback failed to log train loss!"
    assert len(tracker.valid_losses) == 1, "TrackLossCallback failed to log valid loss!"
    
if __name__ == "__main__":
    test_stage5_learner_pipeline()
    print("Stage 5 Learner and Callback pipeline verified successfully!")