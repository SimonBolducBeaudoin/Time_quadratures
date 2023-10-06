#!/bin/env/python
#! -*- coding: utf-8 -*-
import numpy as _np

def gen_labels(ps, format_str='{:0.3f}'):
    """
    Generate labels as formatted strings from the input array.
    
    Returns:
    --------
    labels : numpy.ndarray
        Array of labels as strings with shape = ps.shape.
        
    Example:
    --------
    >>> ps.shape = (3,3)
    >>> labels = gen_labels(ps[...,0])
    >>> print(labels)
    array([['5.500'],
       ['6.000'],
       ['6.500']], dtype='|S5')
    """
    labels = _np.array([format_str.format(p) for p in ps])
    labels = labels[..., _np.newaxis]
    return labels

def gen_labels_from_pair(p1s, p2s,format_str='{:0.3f}',separator='&'):
    labels = _np.array([format_str.format(p1)+separator+format_str.format(p2) for p1, p2 in zip(p1s, p2s)])
    labels = labels[..., _np.newaxis]
    return labels
    
def generate_labels_from_matrix(ps, separator='&', format_str='{:0.3f}'):
    labels = _np.array([separator.join([format_str.format(p) for p in row]) for row in ps])
    return labels