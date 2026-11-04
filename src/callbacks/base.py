class CancelFitException(Exception):
    pass

class CancelEpochException(Exception):
    pass

class CancelBatchException(Exception):
    pass

class Callback:
    order=0
    
    def set_runner(self,runner):
        self.run=runner
        
    def __getattr__(self,k):
        return getattr(self.run, k)
    
    def before_fit(self):
        pass
    
    def before_epoch(self):
        pass
    
    def before_batch(self):
        pass
    
    def after_pred(self):
        pass
    
    def after_loss(self):
        pass
    
    def before_step(self):
        pass
    
    def after_step(self):
        pass
    
    def after_batch(self):
        pass
    
    def after_epoch(self):
        pass
    
    def after_fit(self):
        pass
    