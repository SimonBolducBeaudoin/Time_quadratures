#!/bin/python
# -*- coding: utf-8 -*-

import numpy as _np
from fractions import Fraction as _Fraction 
from typeguard import typechecked as _typechecked

@_typechecked
def get_period(f: int,sampling_rate: int = 32000000000) -> int:
    """
    Returns the period at which the sampling rate will be commensurate with f.
    Example f = 12 sampling_rate=32.
        ==> Each 8 points we do a period which correspond to 3 cycles of f.
    """
    return _Fraction(f,sampling_rate).denominator
   
@_typechecked
def get_ordered_index(f: int,sampling_rate: int = 32000000000) -> _np.ndarray:
    """
    Returns the index that are ordering the times inside each cycle
    """
    fraction=_Fraction(f,sampling_rate)
    Period = fraction.denominator
    Cycle = fraction.numerator
    t_cycle=(_np.r_[0:Period]*Cycle/Period)%(1)
    return _np.argsort(t_cycle)
    
@_typechecked
def get_cycle_time(f: int,sampling_rate: int = 32000000000) -> _np.ndarray:
    """
    Returns the index that are ordering the times inside each cycle
    """
    fraction=_Fraction(f,sampling_rate)
    Period = fraction.denominator
    Cycle = fraction.numerator
    t_cycle=(_np.r_[0:Period]*Cycle/Period)%(1)
    return t_cycle