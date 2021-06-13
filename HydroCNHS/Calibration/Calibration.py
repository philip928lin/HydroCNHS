from .Convertor import Convertor
from .DMC import DMC
from .KGCA import KGCA
from .GA import GA
class Cali():
    DMC = DMC
    KGCA = KGCA
    GA = GA
    Convertor = Convertor
    
    def __init__(self) -> None:
        pass
    # @staticmethod
    # def DMC():
    #     return DMC
    # @staticmethod
    # def KGCA():
    #     return KmeansGA