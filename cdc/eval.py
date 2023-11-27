import numpy as np
from tqdm import tqdm


class EvaluateOneTask:
    def __init__(self, model, x:np.ndarray, y:np.ndarray, typ:str) -> None:
        
        self.model = model
        self.x:np.ndarray = x
        self.y:np.ndarray = y
        self.classes:np.ndarray = np.unique(y)
        self.typ:str = typ
        self.y_true:list = []
        self.y_pred:list = []
        self.y_probas:list = []
    
    def evaluate(self, Samplesize:int) -> tuple:
        iters:list = tqdm(range(len(self.x)))
        for index in iters:
            if self.typ == 'text_as_array':
                sample:np.ndarray = np.array(self.x[index])
            else:
                sample:np.ndarray = self.x[index]
                
            y:np.ndarray = self.y[index]
            kw_samples:list = []
            
            for classIndex in range(len(self.classes)):
                mask:np.ndarray = np.where(self.y == classIndex)[0]
                
                while True:
                    random_indexs:np.ndarray = np.random.choice(mask,
                                                                size=Samplesize,
                                                                replace=True)    
                    if index not in random_indexs:
                        break
                
                kw_samples.append(self.x[random_indexs])
            
            class_, preds, _ = self.model.predict(sample, kw_samples)
            
            self.y_pred.append(class_)
            self.y_true.append(y)
            self.y_probas.append(preds)
        
        return self.y_true, self.y_pred, self.y_probas


class EvaluateMultipleTasks:
    pass