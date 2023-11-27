import numpy as np
from typing import List
from typing import Union
from numpy.linalg import norm
from scipy.spatial.distance import cdist

from .matrix import Matrix
from .distance import Distance
from .compressor import Compressor


class BaseModel:
    
    def __call__(self, x:np.ndarray) -> np.ndarray:
        return self.predict(x)
    
    def softmax(self, x:np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
    def __forward__(self, x:np.ndarray) -> np.ndarray:
        ...
    
    def predict(self, x:np.ndarray) -> np.ndarray:
        pred:np.ndarray = self.__forward__(x)
        return self.softmax(pred)


class EucCDC(BaseModel):
    def __repr__(self) -> str:
        return 'Euc() (Euclidean)'
    
    def __forward__(self, xs:np.ndarray) -> np.ndarray:
        distances:np.ndarray = np.zeros(shape=(len(xs)),
                                        dtype=np.float32)
        
        for index, x in enumerate(xs):
            samples:np.ndarray = x[:-1, :].mean(axis=0).reshape(1, -1)
            query:np.ndarray = x[-1:, :].reshape(1, -1)
            
            euclidean:np.ndarray = cdist(samples, query, metric='euclidean')
            distances[index] = euclidean.item()
        
        return -distances


class CosCDC(BaseModel):
    def __repr__(self) -> str:
        return 'Cos() (Cosine)'
    
    def __forward__(self, xs:np.ndarray) -> np.ndarray:
        
        distances:np.ndarray = np.zeros(shape=(len(xs)),
                                        dtype=np.float32)
        
        for index, x in enumerate(xs):
            samples:np.ndarray = x[:-1, :].mean(axis=0).reshape(1, -1)
            query:np.ndarray = x[-1:, :].reshape(1, -1)
            
            cos:np.ndarray = (samples @ query.T) / (norm(samples) * norm(query))
            distances[index] = 1.0 - cos.item()
        
        return -distances


class MinMaxCDC(BaseModel):
    def __repr__(self) -> str:
        return 'Min Max() (Euclidean + Cosine)'
    
    def __forward__(self, x:np.ndarray) -> np.ndarray:
        distances:np.ndarray = np.zeros(shape=(len(x)),
                                        dtype=np.float32)
        
        for index, x in enumerate(x):
            samples:np.ndarray = x[:-1, :]
            query:np.ndarray = x[-1:, :]
            
            min_, max_ = samples.min(axis=0).reshape(1, -1), samples.max(axis=0).reshape(1, -1)
            
            euc_min:np.ndarray = cdist(min_, query, metric='euclidean').item()
            euc_max:np.ndarray = cdist(max_, query, metric='euclidean').item()
            
            cos_min:np.ndarray = 1 - ((min_ @ query.T) / (norm(min_) * norm(query))).item()
            cos_max:np.ndarray = 1 - ((max_ @ query.T) / (norm(max_) * norm(query))).item()
            cos:np.ndarray = (cos_max + cos_min)/2
            euc:np.ndarray = (euc_max + euc_min)/2

            distances[index] = (euc + cos)
        return -distances

class EucCosCDC(BaseModel):
    def __repr__(self) -> str:
        return 'EucCos() (Euclidean + Cosine)'
    
    def __forward__(self, xs:np.ndarray) -> np.ndarray:
        distances:np.ndarray = np.zeros(shape=(len(xs)),
                                        dtype=np.float32)
        
        for index, x in enumerate(xs):
            samples:np.ndarray = x[:-1, :].mean(axis=0).reshape(1, -1)
            query:np.ndarray = x[-1:, :].reshape(1, -1)
            
            euclidean:np.ndarray = cdist(samples, query, metric='euclidean').item()
            cos:np.ndarray = 1.0 - ((samples @ query.T) / (norm(samples) * norm(query))).item()
            distances[index] = (euclidean + cos)
        
        return -distances

class CDCEnsabmle:
    def __init__(self, model, compressors:Union[str, List], distances:Union[str, List], typ:str) -> None:
        self.model = model
        if compressors == '__all__':
            compressors = ['bz2', 'gzip', 'zlib']
        if distances == '__all__':
            distances = ['cdm', 'clm', 'ncd']
            
        if isinstance(compressors, str):
            compressors = [compressors]  
        if isinstance(distances, str):
            distances = [distances]
            
        self.compressors = compressors
        self.distances = distances
        self.type = typ
    
    def __repr__(self) -> str:
        return f'''CDCMulti(Model:{self.model}, Compressors:{", ".join(self.compressors)}, Distances:{", ".join(self.distances)})'''
        
    def predict(self, sample:Union[str, np.ndarray], classes:List[List]) -> np.ndarray:
        
        preds:list = []
        
        for compressor_index, compressor in enumerate(self.compressors):
            current_compressor:Compressor = Compressor(compressor)
            for distance_index, distance in enumerate(self.distances):
                current_distance:Distance = Distance(distance)
                calc_matrixs:np.ndarray = Matrix(compressor=current_compressor,
                                                 distance=current_distance,
                                                 typ=self.type)
                
                matrixs:np.ndarray = calc_matrixs.get_matrix(sample=sample,
                                                             classes=classes)
                
                pred:np.ndarray = self.model(matrixs)
                preds.append(pred)
                
        all_preds:np.ndarray = np.array(preds)
        preds = np.array(all_preds).mean(axis=0)
        votes:dict = dict(zip(self.compressors, all_preds.tolist()))
        
        return preds.argmax().item(), preds, votes