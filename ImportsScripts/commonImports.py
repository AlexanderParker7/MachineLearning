from __future__ import print_function
import numpy as np                   # for numerical computation
import sympy as sp                   # sympy module for computation
import numpy.linalg as lg            # numpy linear algebra module
import scipy.linalg as sg            # scipy linear algebra module
import scipy.integrate as si
from scipy.stats import ortho_group   
import importlib
import itertools 
import inspect
import csv
import pandas as pd
import random
from IPython.display import display, Math

import autograd.numpy as anp                     # Thinly−wrapped numpy
from autograd import grad

from grcodes import drawArrow, plotfig, printM, printx # frequently used

import math as ma
from sympy import sin, cos, tan, sinh, cosh, exp, log, symbols, lambdify, latex
from sympy import pi, Matrix, sqrt, oo, integrate, diff, Derivative, Rational, Float
from sympy import MatrixSymbol, simplify, nsimplify, Function, init_printing  
from sympy import factor, expand, nsimplify, Matrix, ordered, hessian
from sympy.plotting import plot as splt
init_printing(use_unicode=True) # for latex-like qualityprinting formate

from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt      # for plotting figures
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

#np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
#np.set_printoptions(precision=8)#, suppress=True)
#cmap = {'complex_kind': '{:>6.3f}'.format}
#cmap = {'complex_kind': '{:13.3f}'.rstrip('0').rstrip('.').format}
#with np.printoptions(suppress=True,formatter=cMap): 
#    print(f"Eigenvalues of skew-sy. part of random matrix A:\n{e}")