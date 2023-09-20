#!/bin/env/python
#! -*- coding: utf-8 -*-

import os as _os, platform as _platform
mingw_path = 'C:\\cygwin64\\usr\\x86_64-w64-mingw32\\sys-root\\mingw\\bin'
if _os.name == 'nt':
    if _platform.python_version_tuple()[0]== '2' : #python 2
        if ( mingw_path not in _os.environ['PATH'] ) :
            _os.environ['PATH'] = mingw_path+_os.path.pathsep+_os.environ['PATH']
    else : # python 3      
        _os.add_dll_directory("C:/cygwin64/usr/x86_64-w64-mingw32/sys-root/mingw/bin")
del mingw_path

__all__ = ["time_quadratures","TimeQuadrature_helper","Deprecated"]
 