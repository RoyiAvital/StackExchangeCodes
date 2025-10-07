# Python STD
import math

# Typing
from typing import List, Tuple, Union

def CeilFloat( inVal: float, numDigits: int = 0 ) -> float:
    
    return math.ceil( inVal * (10 ** numDigits) ) / (10 ** numDigits)

def FloorFloat( inVal: float, numDigits: int = 0 ) -> float:
    
    return math.floor( inVal * (10 ** numDigits) ) / (10 ** numDigits)