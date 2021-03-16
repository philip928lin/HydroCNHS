from .Convertor import Convertor
from .DMC import DMC
from .KmeansGA import KmeansGA

class Cali():
    DMC = DMC
    KGCA = KmeansGA
    Convertor = Convertor
    
    def __init__(self) -> None:
        pass
    # @staticmethod
    # def DMC():
    #     return DMC
    # @staticmethod
    # def KGCA():
    #     return KmeansGA