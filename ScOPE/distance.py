class Distance:
    """Distance class for calculating distance between two compressed strings"""
    def __init__(self, distance:str) -> None:
        """Initializes Distance class

        Args:
            distance (str): Distance to use for calculating distance between two compressed strings
            Currently supported distances are:
                - ncd: Normalized Compression Distance
                - cdm: Compression-based Distance Measure
                - clm: Compression-based Length Measure
        Raises:
            ValueError: If distance is not implemented.
        """
        
        __supported_distances__ = {
            'ncd': self.__ncd__,
            'cdm': self.__cdm__,
            'clm': self.__clm__,
        }
        
        if distance not in __supported_distances__:
            print(f'Error: {distance} is not a supported distance')
            print(f'Please choose one of the following: {__supported_distances__.keys()}')
            raise ValueError
        
        self.distance = __supported_distances__[distance]
        self.distance_name = distance
    
    def __repr__(self) -> str:
        return f'Distance({self.distance_name})'
    
    def __str__(self) -> str:
        return f'Distance({self.distance_name})'
    
    def __ncd__(self, x1:float, x2:float,
                x1x2:float) -> float:
        """Normalized Compression Distance (NCD)

        Args:
            x1 (float): Size of x1 after compression
            x2 (float): Size of x2 after compression
            x1x2 (float): Size of x1 + x2 after compression

        Returns:
            float: NCD distance between x1 and x2
        """
        
        denominator:float = max(x1, x2)
        numerator:float = x1x2 - min(x1, x2)
        return numerator / denominator
    
    def __cdm__(self, x1:float, x2:float,
                x1x2:float) -> float:
        """Compression-based Distance Measure (CDM)

        Args:
            x1 (float): Size of x1 after compression
            x2 (float): Size of x2 after compression
            x1x2 (float): Size of x1 + x2 after compression

        Returns:
            float: CDM distance between x1 and x2
        """
        denominator:float = x1 + x2
        numerator:float = x1x2
        return numerator / denominator
    
    def __clm__(self, x1:float, x2:float,
                x1x2:float) -> float:
        """Compression-based Length Measure (CLM)

        Args:
            x1 (float): Size of x1 after compression
            x2 (float): Size of x2 after compression
            x1x2 (float): Size of x1 + x2 after compression

        Returns:
            float: CLM distance between x1 and x2
        """
        
        denominator:float = x1x2
        numerator:float = 1 -(x1 + x2 - x1x2)
        return numerator / denominator
    
    def __check_values__(self, **kwargs) -> bool:
        return all(isinstance(value, (float)) for value in kwargs.values())
    
    def __call__(self, **kwargs) -> float:
        """ 
        Args:
            x1 (float): Size of x1 after compression
            x2 (float): Size of x2 after compression
            x1x2 (float): Size of x1 + x2 after compression, Ignored for MSE
            
        Returns:
            float: Distance between x1 and x2
        """
        
        if self.__check_values__(**kwargs) is False:
            print('Error: All values must be floats')
            raise ValueError
        
        return self.distance(**kwargs)

if __name__ == '__main__':
    from compressor import Compressor
    compressor = Compressor('gzip')
    distance = Distance('ncd')
    
    x1 = compressor('Hola')
    x2 = compressor('Adios')
    x1x2 = compressor('Hola Adios')
    print(distance(x1=x1, x2=x2, x1x2=x1x2))
    
    x1 = compressor(['Hola'])
    x2 = compressor(['Adios'])
    x1x2 = compressor(['Hola Adios'])
    print(distance(x1=x1, x2=x2, x1x2=x1x2))
    
    # x1 = compressor([1, 2, 3, 4, 5])
    # x2 = compressor([1, 2, 3, 4, 5, 6])
    # x1x2 = compressor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    # print(distance(x1=x1, x2=x2, x1x2=x1x2))