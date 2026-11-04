import torch
from src.callbacks.base import (
    Callback,
    CancelFitException,
    CancelEpochException,
    CancelBatchException,
)

class Learner:
    def __init__(self, model, dls, loss_func, opt, cbs=None):
        self.model=model
        self.dls=dls
        self.loss_func=loss_func
        self.opt=opt
        self.cbs=[]
        self.add_cbs(cbs or [])
        
        self.epoch=0
        self.epochs=0
        self.iter=0
        self.training=True
        self.xb=None
        self.yb=None
        self.preds=None
        self.loss=None
        
    def add_cbs(self,cbs):
        for cb in cbs:
            cb.set_runner(self)
            self.cbs.append(cb)
        self.cbs.sort(key=lambda x:getattr(x, "order", 0))
        
    def _call_cbs(self, method_name):
        for cb in self.cbs:
            method = getattr(cb, method_name, None)
            if method:
                method()
                
    def one_batch(self,i,xb,yb):
        self.iter=i
        self.xb,self.yb=xb,yb
        try:
            self._call_cbs("before_batch")
            self.preds=self.model(self.xb)
            self._call_cbs("after_preds")
            
            self.loss=self.loss_func(self.preds, self.yb)
            self._call_cbs("after_loss")
            
            if not self.training:
                return
            
            self.opt.zero_grad()
            self._call_cbs("before_step")
            
            if hasattr(self.model, "backward"):
                if hasattr(self.loss_func, "backward"):
                    dloss=self.loss_func.backward()
                else:
                    dloss=torch.ones_like(self.preds)/ self.preds.shape[0]
                self.model.backward(dloss)
            elif hasattr(self.loss,"backward"):
                self.loss.backward()
                
            self.opt.step()
            self._call_cbs("after_step")
    
        except CancelBatchException:
            pass
        finally:
            self._call_cbs("after_batch")
            
    def all_batches(self,dl):
        for i, (xb,yb) in enumerate(dl):
            self.one_batch(i,xb,yb)
            
    def one_epoch(self, training):
        self.training=training
        dl=self.dls[0] if training else self.dls[1]
        if hasattr(self.model, "train") and hasattr(self.model, "eval"):
            self.model.train() if training else self.model.eval()
            
        try:
            self._call_cbs("before_epoch")
            self.all_batches(dl)
        except CancelEpochException:
            pass
        finally:
            self._call_cbs("after_epoch")
            
    def fit(self, epochs):
        self.epochs=epochs
        try:
            self._call_cbs("before_fit")
            for epoch in range(epochs):
                self.epochs=epoch
                self.one_epoch(training=True)
                if len(self.dls)>1 and self.dls[1] is not None:
                    with torch.no_grad():
                        self.one_epoch(training=False)
        except CancelFitException:
            pass
        finally:
            self._call_cbs("after_fit")
            