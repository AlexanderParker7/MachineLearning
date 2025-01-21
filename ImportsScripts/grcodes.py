import sys
import numpy as np
import sympy as sp
import scipy
import numpy.linalg as lg   
import math as ma
import scipy.integrate as si
from scipy.special import p_roots
import itertools

import inspect
import matplotlib.pyplot as plt      # for plotting figures
plt.rcParams['figure.dpi'] = 500
plt.rcParams.update({'font.size': 6})
plt.rcParams.update({'figure.figsize':(6,4)})
#plt.rcParams['axes.linewidth'] = 0.3

from IPython.display import display, Math
from sympy import sin, cos, symbols, lambdify, latex, Matrix, Function
from mxnet import nd
from collections import OrderedDict
import matplotlib.colors as colors 


def printx(name):                            
    """ This prints out both the name and its value together.  
    usage: name = 88.0;  printx('name') 
    Ref: https://stackoverflow.com/questions/32000934/
    python-print-a-variables-name-and-value"""

    frame = sys._getframe(1)
    print(name, '=', repr(eval(name, frame.f_globals, frame.f_locals)))

#x = 1 
#printnve('x')

#def printnv(name): 
#    print (name, '=', repr(eval(name)))

def principalS(S):
    '''Compute the principal stresses/strains & their direction cosines.
    inputs: 
       S: given stress/strain tensor, numpy array
    return: 
       principal stresses/strain (eigenValues), their direction cosines
       (eigenVectors) ranked by its values. Right-hand-rule is enforced
    '''
    eigenValues, eigenVectors = lg.eig(S)  

    #Sort in order
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    print('Principal stress/strain (Eigenvalues):\n',eigenValues,'\n')
    
    # make the first element in the first vector positive (optional):
    #eigenVectors[0,:] = eigenVectors[0,:]/np.sign(eigenVectors[0,0])

    # Determine the sign for given eigenVector-1 and eigenVector-3
    eigenVectors[:,2] = np.cross(eigenVectors[:,0], eigenVectors[:,1])

    angle = np.arccos(eigenVectors[0,0])*180/np.pi       # in degree 
    print(f'Principal stress/strain directions:\n{eigenVectors}\n')
    print(f"Possible angles (n1,x)={angle}∘ or {180-angle}∘")
    
    return eigenValues, eigenVectors

def principal2angle(𝜎2d): 
    '''Compute the principal stresses for a given 2D stress state 𝜎2d.
    '''
    𝜃p1 = 0.5*np.arctan(2*𝜎2d[0,1]/(𝜎2d[0,0]-𝜎2d[1,1]))  
    #print(f'Angle (X,x) = {𝜃p1}(rad), {np.rad2deg(𝜃p1)}∘')

    𝜎XX,𝜎YY,𝜎XY=tensorT2angle(𝜎2d[0,0],𝜎2d[1,1],𝜎2d[0,1],𝜃p1,py='numpy')
    #print(f'Principal stresses (MPa) :\n{𝜎XX, 𝜎YY, 𝜎XY}')
    return 𝜎XX, 𝜎YY, 𝜎XY, 𝜃p1

def principalS2D(S):
    '''Compute the principal stresses/strains and their directions.
    inputs: 
       S: given stress/strain tensor, numpy array
    return: 
       principal stresses/strains (eigenValues), their direction cosines
       (eigenVectors) ranked by its values. Right-hand-rule is enforced
    '''
    eigenValues, eigenVectors = lg.eig(S)  

    #Sort in order
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    print('Principal stress/strain (Eigenvalues):\n',eigenValues,'\n')
    
    angle = np.arccos(eigenVectors[0,0])*180/np.pi         # in degree 
    print(f'Principal stress/strain directions:\n{eigenVectors}\n')
    print(f"Possible angles (n1,x)={angle}∘ or {180-angle}∘")
    
    return eigenValues, eigenVectors

def M_eigen(stressMatrix):
    '''Compute the eigenvalues & eigenvectors for a given stress 
    (or strain) matrix. 
    Input stressMatrix: the stress/strain matrix with 9 components.
    Return: the eigenvalues are sorted in descending
    order, and the eigenvectors will be sorted accordingly. The sign 
    for eigenVectors are also adjusted based on the right-hand rule. 
    '''
    eigenValues, eigenVectors = lg.eig(stressMatrix)  

    # Sort in descending order
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]

    # Determine the sign for given eigenVector-1 and eigenVector-3
    eigenVectors[2,:] = np.cross(eigenVectors[0,:], eigenVectors[1,:])
    return eigenValues, eigenVectors


def M_eigen2D(stressMatrix):
    '''Compute eigenvalues & eigenvectors using 2D stress/strain matrix 
    Input stressMatrix: the stress matrix with 4 components.
    Return: the eigenvaluesare sorted in descending
    order, and the eigenvectors will be sorted accordingly.  
    '''
    eigenValues, eigenVectors = lg.eig(stressMatrix)  

    #Sort in descending order
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]

    return eigenValues, eigenVectors

def von_Mises(e):                    
    '''Compute the von Mises stress using principal stresses
    Input e: eigenvalues of the stress matrix or principal stresses
    '''
    vonMises=np.sqrt(((e[0]-e[1])**2+(e[0]-e[2])**2+(e[1]-e[2])**2)/2.)
    return vonMises

def Oct_stress(e):                    
    '''Compute the octahedral shear stress using principal stresses
    Input e: eigenvalues of the stress matrix or principal stresses
    '''
    sigma_oct= np.sum(e)/3. 
    tau_oct=np.sqrt((e[0]-e[1])**2+(e[0]-e[2])**2+(e[1]-e[2])**2)/3. 
    return sigma_oct, tau_oct

def energy_stress(e,nu):                    
    '''Compute the strain-energy using principal stresses
    Input e: eigenvalues of the stress matrix or principal stresses
         nu: Poisson's ratio.
    '''
    U_stress=np.sqrt(e[0]**2+e[1]**2+e[2]**2-2*nu*(e[1]*e[2]+
                     e[0]*e[2]+e[0]*e[1]))
    return U_stress

def M_invariant(e):                    
    '''Compute the 3 stress/strain invariants using principal stresses
    Input e: eigenvalues of the stress matrix or principal stresses
    '''
    I1 = np.sum(e)
    I2 = e[0]*e[1]+e[1]*e[2]+e[0]*e[2]
    I3 = e[0]*e[1]*e[2]
    return I1, I2, I3

def M_invariant2D(e):                    
    '''Compute the 2 stress/strain invariants using principal stresses
    Input e: eigenvalues of the stress matrix or principal stresses
    '''
    I1 = np.sum(e)
    I2 = e[0]*e[1]
    return I1, I2

def meanDeviator(stressM,e):                    
    '''Compute the mean stress, mean stresses, deviatoric stresses.
    input stressM: given stress (strain or any) matrix with 9 components
    Input e: eigenvalues of stress/strain matrix or principal stresses
    Input stressM: The stress/strain matrix
    '''
    meanStress = np.mean(e)
    meanMatrix = np.diag([meanStress,meanStress,meanStress])
    deviatorM = stressM - meanMatrix
    return meanStress, meanMatrix, deviatorM

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    ''' provided by ChatGPT'''
    diff = abs(sp.N(a) - sp.N(b))
    return diff <= max(rel_tol * max(abs(sp.N(a)), abs(sp.N(b))), abs_tol)

def printAn(A, n, p_text="A"):
    '''print randomly selected n elements in array A
    '''
    if A.ndim == 1:
        rows = A.shape
        cols =1
    else:
        rows, cols = A.shape
        
    random_idx = np.random.randint(0, rows*cols, size=n)
    random_ele = A.flatten()[random_idx]
    
    print(f'Randomly selected {n} elements: {p_text}:\n{random_ele}')

def printc(c, n_d=4):
    '''print a complex number c with n_d number of significant digits 
       after the decimal place for both real and imaginary parts.
    Inputs: 
       c  : the complex number 
       n_d: number of digits to keep after the decimal place.
    '''
    print(f"{c.real:.{str(n_d)}f} + {c.imag:.{str(n_d)}f}j")

def permuP(perm, module='Numpy'):
    '''Create a permutation matrix P for a given permutation
    Input:   perm: list, given permutations needed.
    module:  if Sympy, P will be Sympy Matrix 
             else, P will be a numpy array
    Return:  P, the corresponding permutation matrix generated. 
    Example: [0, 1, 2]       # original order
              ⇓  ⇓  ⇓         
    perm  =  [2, 0, 1]       # the given permuted order
    '''
    n = len(perm)                              # dimension of the matrix
    if module == 'Sympy':
        I = sp.eye(n)                                  # identity matrix 
        P = sp.zeros(n)              # initialize the permutation matrix
    else: 
        I = np.eye(n)                      
        P = np.zeros((n,n))            

#    for j in range(n):
#        P[:, perm[j]] = I[:,j]                                 # permute 

    P = I[perm,:]
        
    return P                  

def permuP_sp(perm):
    '''Create a permutation matrix P for a given permutation
    Input  : perm: list, given permutations needed.
    Return : P, Sympy Matrix, the corresponding permutation matrix 
             to be generated. 
    Example: [0, 1, 2]       # original order
              ⇓  ⇓  ⇓         
    perm   = [2, 0, 1]       # the given permuted order
    '''
    n = len(perm)                              # dimension of the matrix
    I = sp.eye(n)                                     # identity matrix 
    P = sp.zeros(n)                  # initialize the permutation matrix 

    for j in range(n):
        P[:, perm[j]] = I[:,j]                                # permute 
        
    return P                                                            

def newton_iter(f, df, guess_roots, limit=1000, atol=1e-6):  
    ''' Finds the roots of a function via the Newton iteration.
        inputs: f--the given function;
                df--the derivative of f; 
                guess_roots--possible locations of the roots;
                limit: max number of interations;
                atol:  tolerance to control convergence;
        return: the roots found.'''  
    i = 0;   roots = np.array([])
    for r in guess_roots:
        while lg.norm(f(r)) > atol and i < limit:
            r = r - f(r)/df(r)
            i += 1
        roots = np.append(roots, r)
        
    if lg.norm(f(r)) > atol: 
        print(f'Not yet converged:{atol}!!!')
    else:
        print(f'Converged at {i}th iteration to {atol}')
        
    return roots

def find_roots(sp_f, xL, xR, n_division = 200):
    '''Find the roots of a 1D sympy function, sp_f, within the domain 
       of (xL, xR)
    '''
    dx = (xR-xL)/n_division
    X = np.arange(xL, xR+dx, dx)               # discretize the domain
    np_f = lambdify(x, sp_f, "numpy")           # convert sp_f to np_f
    
    # compute the sign of np_f
    signs = np.sign(np_f(X))         
    # find the indices where sign changes: gives guesses of the roots. 
    indices = np.where(np.diff(signs))[0] + 1 
    guesses = X[indices]                        # guesses of the roots.
    roots = [sp.nsolve(sp_f, x, guess) for guess in guesses] 
    #print(f" Approximated roots = {roots}")  
    return roots 

def HermiteR(n, x):   
    '''Generate the Laguerre polynomials using the recursion 
    formula, degree (n-1) '''
    if   n == 0: return sp.S.One
    elif n == 1: return 2*x
    else:  
        Her = 2*x*HermiteR(n-1,x)-2*(n-1)*HermiteR(n-2,x) 
    return Her 

def LegendreR(n, x):   
    '''Generate the Legendre polynomials using the Bonnet’s recursion 
    formula, degree (n-1) '''
    if   n == 0: return sp.S.One
    elif n == 1: return x
    else:
        leg = ((2*n-1)*x*LegendreR(n-1,x)-(n-1)*LegendreR(n-2,x))/n 
        return leg.expand() 


def PolyAprox(f_true, Ns, x_nodes, Pfit, Px):
    '''To plot the curve fitted using a set of basis functions, and 
    with the true function to be approximated.
    
    f_true:  the true function
    Pfit:    numpy polynomial fitting object for fitting
    Px:      numpy polynomial object 
    Ns:      list, degrees of approximation
    x_nodes: array like, nodes on x-axis used for fitting
    '''
    fx_nodes = [f_true(xi) for xi in x_nodes]  # true f vals for fitting

    for i in range(len(Ns)):
        P_coef = Pfit(x_nodes, fx_nodes, Ns[i])       # fit f for coeffs
        P_ap_x   = Px(P_coef)         # compute the fitted function of x

    return P_ap_x 


def plot_PolyAprox(f_true, Ns, x_nodes, X, Pfit, Px, P_type='Chebyshev'):
    '''To plot the curve fitted by a set of basis functions, to gether 
    with the original function to be approximated.
    
    f_true:  the true function
    Pfit:    numpy polynomial fitting object for fitting
    Px:      numpy polynomial object 
    Ns:      list, degrees of approximation
    x_nodes: array like, nodes on x-axis used for fitting
    X:       array like, points on x-axis for plots
    P_type:  string, tpye of the basis polynomial used, for file name. 
    '''
    fX = [f_true(xi) for xi in X]              # ture function for plots

    fx_nodes = [f_true(xi) for xi in x_nodes]  # true f vals for fitting
    
    plt.rcParams.update({'font.size': 6})
    fig_s = plt.figure(figsize=(6,5))   

    for i in range(len(Ns)):
        ax = fig_s.add_subplot(2,2,i+1)
        P_coef = Pfit(x_nodes, fx_nodes, Ns[i])       # fit f for coeffs
        P_ap_x = Px(P_coef)                          # fitted polynomial
        P_apX   = P_ap_x(X)            # Compute the fitted results at X
                                                          # for plotting
        ax.plot(X,P_apX,'mo',ms=2,label=P_type+' deg.'+str(Ns[i])+
                f', rmse: {sqrt(np.mean((P_apX-fX)**2)):.4e}')
        ax.plot(X, fX, c='k', label='Original Function')
        if i==2 or i==3: ax.set_xlabel('x')
        if i==0 or i==2: ax.set_ylabel('$f(x)$')
        ax.legend()        #loc='center right', bbox_to_anchor=(1, 0.5)) 
        
    plt.savefig('images/'+P_type+'.png',dpi=500,bbox_inches='tight')
    plt.show() 
    
    return P_ap_x 


def cheby_T(n, x):   
    '''Generate the first kind Chebyshev polynomials of degree (n-1) '''
    if   n == 0: return sp.S.One
    elif n == 1: return x
    else:        return (2*x*cheby_T(n-1,x)-cheby_T(n-2,x)).expand()

def cheby_U(n, x):   
    '''Generate the 2nd kind Chebyshev polynomials of degree (n-1) '''
    if   n == 0: return sp.S.One
    elif n == 1: return 2*x
    else:        return (2*x*cheby_U(n-1,x)-cheby_U(n-2,x)).expand()

def node_shape_f(𝜉):
    '''Linear nodal shape function in natural coordinate 𝜉.
    This shape function has 4 pieces.''' 
    N = np.piecewise(𝜉, [𝜉<-1, (𝜉>=-1)&(𝜉<0),  (𝜉>=0)&(𝜉<=1),  𝜉>1],\
                        [0,    lambda 𝜉: 1+𝜉,  lambda 𝜉: 1-𝜉,   0]) 
    return N

def uniqueX(X, atol=1.e-6): 
    '''Remove elements in X that has close values within atol'''
    u_ndcs=np.unique([np.isclose(X, v, atol=atol).argmax() for v in X])
    return X[u_ndcs]
## Example: 
#X = np.array([1.0, 2.0, 2.1, 2.2, 3.0, 3.05, 3.1, 5.3, 6.0])
#print(uniqueX(X, atol=1.e-1))

def gauss_df(x,mu,sigma):               
    '''define the Gauss function: 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu)/sigma)**2)'''
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu)/sigma)**2)

def errors(a, b): 
    '''Computing the MSE-, Min-, Max-Errors between two 1D arrays
       a and b.
    return: mse, min_error, max_error
    '''
    if a.shape != b.shape: 
        print(f'Warning: a.shape != b.shape: {a.shape, b.shape}')
    
    af = a.flatten(); bf = b.flatten()
    mse = mean_squared_error(af, bf)                # Mean squared error
     
    a_norm=(np.linalg.norm(af,np.inf)-np.linalg.norm(af,-np.inf))/2+1e-8
    #rel_error = np.linalg.norm(af - bf)/a_norm/len(af)
    min_error = np.min((af-bf))/a_norm  
    max_error = np.max((af-bf))/a_norm  
    return mse, min_error, max_error

def LagrangeP(x, xi):
    '''Create shape function matrix N(x)=[l_0(x), l_1(x),...l_degree(x)] 
    in as list for a 1D interval (element) [a=x_0, x_1, ...,  b=x_n], 
    using the Lagrange interpolator.
    '''
    N = []                       # l_0(x) is a shape function N(x)
    for i in range(len(xi)):
        nodes = [n for n in range(len(xi))]
        nodes.pop(i)
        N.append(np.prod([(x-xi[k])/(xi[i]-xi[k]) for k in nodes]))
        
    return N


def LaguerreR(n, x):   
    '''Generate the Laguerre polynomials using the recursion 
    formula, degree (n-1) '''
    if   n == 0: return sp.S.One
    elif n == 1: return -x+1
    else:
        Lag = ((2*n-1-x)*LaguerreR(n-1,x)-(n-1)*LaguerreR(n-2,x))/n 
        return Lag.expand()


def pad_last(arr, c_inc, maxlen):
    '''Extend the given arr to with maxlen columns, by repeating the 
    last row in the array. The c_inc column (usually the time) is set 
    to increase with the same rate of the last two elments in the column
    of the orginal arr.
    Return: arr_expanded.
    '''
    row_add = maxlen - arr.shape[0]      # number of rows to add to arr
    if arr[-2, c_inc] == 0:
        rate = 0
    else:
        rate = (arr[-1, c_inc] - arr[-2, c_inc]) / arr[-2, c_inc]

    # repeat the last row to expand the arr to shape (n+row_add,m)
    arr_add = np.repeat(arr[-1:, :], row_add, axis=0)

    # gradually increase the values in the 1st column of  extended array
    for i in range(row_add):
        arr_add[i,c_inc] = arr[-1,c_inc] + (i+1)*rate*arr[-1,c_inc]

    # concatenate the original array with the expanded array
    arr_expanded = np.concatenate((arr, arr_add), axis=0)
    return arr_expanded

## Example: 
# n=9; m=5; row_add=2; c_inc = 0 
# arr = np.arange(n*m).reshape(n, m)   # create a numpy array
# print(f'Original array: {arr.shape} \n{arr}') 
# arr_expanded = pad_last(arr, c_inc, n+row_add)
# print(f'Extended array: {arr_expanded.shape} \n{arr_expanded}') 

def array_to_list(dictionary):
    """ Recursively convert all NumPy arrays in a dictionary to lists.
        Suggested by ChatGPT. 
    """
    for k, v in dictionary.items():
        if isinstance(v, np.ndarray): dictionary[k] = v.tolist()
        elif     isinstance(v, dict): array_to_list(v)


def flattenL(nested_list):
    '''flatten lists with multiple depth, recursively. 
       Suggested by ChatGPT'''
    flat_list = []
    for item in nested_list:
        if isinstance(item, list): flat_list.extend(flatten_list(item))
        else:                      flat_list.append(item)
    return flat_list

def roundS0(expr, n_d):
    '''Usage: roundS(expr, n_d), where n_d: number of digits to keep.
    Ref: https://stackoverflow.com/questions/48491577/
    printing-the-output-rounded-to-3-decimals-in-sympy'''
    return expr.xreplace({n.evalf():round(n,n_d) 
                          for n in expr.atoms(sp.Number)})

def roundS(expr, n_d):           # to be used in maxminS() 
    '''To limit the number of digits to keep in a variable. 
    Usage: roundS(expr, n_d), where n_d: number of digits to keep.
    Ref: https://stackoverflow.com/questions/48491577/
    printing-the-output-rounded-to-3-decimals-in-sympy
    '''
    return expr.xreplace({n.evalf():round(n,n_d) 
                          for n in expr.atoms(sp.Number)})


def solver1D2(E, A, bf, l, bc0, bcl, key='disp'):
    '''Solves mechanics problems for 1D bars subjected to body 
    force bf: u_xx=-bf(x)/EA, with BCs: u(0)=bc0; u(l)=bcl or fR=bcl.
    Input: bf, body force; l, the length of the bar; 
           bcl, condition at x=l; 
           key="disp", displacement condition at a boundary;
           otherwise a force boundary condition
    Return: u (displacement), E*u_x (stress), u_xx (body force term)
    Example: Consider a bar fixed at x=0, loaded with fR at x=L and q: 
             bf = q  
             u = solver1D2(E, A, bf, l, uL, fR, key='force')
    ''' 
    x, c0, c1 = sp.symbols('x, c0, c1')
    
    # Integrate twice to obtain the general solutions in u_x and u: 
    # u_x: du/dx
    u_x = sp.integrate(-bf/E/A,(x))+ c0  #c0: integration constant
    u   = sp.integrate(u_x,(x))+ c1      #c1: integration constant
    
    # Solve for the integration constants: 
    if key == "disp":
        consts=sp.solve([u.subs(x,0)-bc0, u.subs(x,l)-bcl],  [c0, c1])
    else:
        consts=sp.solve([u.subs(x,0)-bc0,u_x.subs(x,l)-bcl/E/A],[c0,c1])
        
    # Substitute the constants back to the general solutions, to obtain
    # solutions that satisfy the BCs:
    u   =   u.subs({c0:consts[c0],c1:consts[c1]})
    u   =   u.expand().simplify()
    u_x = u_x.subs({c0:consts[c0],c1:consts[c1]})
    u_x = u_x.expand().simplify()
    u_xx = sp.diff(u_x,x)
    
    print(f'output from solver1D2(): u, 𝜎, u_xx')
    return u.expand(), (E*u_x).expand(), u_xx.expand()

def maxminS(f, title="value"):
    '''Find maximum location and values of a symbolic function f.
    '''
    x = sp.symbols('x')
    ndg = 8                                 # number of digits
    df = sp.diff(f, x)
    ddf = sp.diff(df, x)
    df0_points = sp.solve(df, x)       #find stationary points
    df0s = [roundS(point.evalf(),ndg) for point in df0_points]
    print(f"Stationary points for {title}: {df0s}")

    for point in df0s:
        fv = roundS((f.subs(x,point)).evalf(), ndg)
        ddfv = ddf.subs({x:point})
        if ddfv < 0:
            print(f"At x={point}, local maximum {title}={fv}")
        elif ddfv > 0:
            print(f"At x={point}, local minimum {title}={fv}")
        else:
            print(f"At x={point}, max or min {title}={fv}")

    df0s.append(0)        # Add in points on the boundaries
    df0s.append(1)
    fs_df0 = [f.subs(x, point).evalf() for point in df0s]
    f_max = roundS(max(fs_df0), ndg)
    x_max=[pnt for pnt in df0s if ma.isclose(f.subs(x,pnt),max(fs_df0))]

    print(f"\nAt x={x_max}, Max {title}={f_max}\n")

def maxminS0(f, x, title="value"):
    ndg = 8           # number of digits
    df = sp.diff(f, x)
    ddf = sp.diff(df, x)
    df0_points = sp.solve(df, x)     #find stationary points
    df0s = [roundS(point.evalf(),ndg) for point in df0_points]
    print(f"Stationary points for {title}: {df0s}")

    for point in df0s:
        fv = roundS((f.subs(x,point)).evalf(), ndg)
        ddfv = ddf.subs(x, point)
        if ddfv < 0:
            print(f"At x={point}, local maximum {title}={fv}")
        elif ddfv > 0:
            print(f"At x={point}, local minimum {title}={fv}")
        else:
            print(f"At x={point}, max or min {title}={fv}")

    df0s.append(0)        # Add in points on the boundaries
    df0s.append(1)
    fs_df0 = [f.subs(x, point).evalf() for point in df0s]
    f_max = roundS(max(fs_df0), ndg)
    x_max=[pnt for pnt in df0s if ma.isclose(f.subs(x,pnt),max(fs_df0))]

    print(f"\nAt x={x_max}, Max {title}={f_max}\n")

def plot2curveS(u, xL=0., xR=1., title="f_title"):
    '''Plot the maximum values and loctions, as well as stationary 
    points, and the values at the stationary points, and boundaries
    input: u, sympy function defined in [xL, xR]
    '''
    x = sp.symbols('x')
    dx = 0.01; dxr = dx*10       # x-step
    xi = np.arange(xL, xR+dx, dx)
    uf = sp.lambdify((x), u[0], 'numpy')   #convert Sympy f to numpy f 
    yi = uf(xi)
    if type(yi) != np.ndarray:       #in case, uf is a constant
        #type(yi) == int or type(yi) == float: # or len(yi)==1:  
        xi = np.arange(xL, xR+dxr, dxr)
        yi = float(yi)*np.ones_like(xi)

    fig, ax1 = plt.subplots(figsize=(5.,1.), dpi=300)
    fs = 8          # fontsize
    color = 'black'
    ax1.set_xlabel('location', fontsize=fs)
    ax1.set_ylabel(title[0], color=color, fontsize=fs)
    ax1.plot(xi, yi, color=color)
    ax1.grid(color='r',ls=':',lw=.3, which='both') # Use both tick 
    ax1.tick_params(axis='x', labelcolor=color, labelsize=fs)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=fs)

    vmax = yi[yi.argmax()]
    max_l = np.argwhere(yi == vmax)
    ax1.plot(xi[max_l],  yi[max_l], 'r*', markersize=4)
    print(f'Maximum {title[0]} value={vmax:.3e}, at x={xi[max_l][0][0]}')
    
    uf = sp.lambdify((x), u[1], 'numpy') #convert Sympy f to numpy f 
    xi = np.arange(xL, xR+dx, dx)
    yi2 = uf(xi)
    if type(yi2) != np.ndarray: # or len(yi2) == 1:
        xi = np.arange(xL, xR+dxr, dxr)
        yi2 = float(yi2)*np.ones_like(xi)

    m1, m2, m3 = np.partition(abs(yi2), 2)[0:3]
    msl=[np.where(abs(yi2)==m1)[0][0],np.where(abs(yi2)==m2)[0][0],
             np.where(abs(yi2)==m3)[0][0]]
    
    vmax = yi2[yi2.argmax()]
    max_l = np.argwhere(yi2 == vmax)
    print(f'Maximum {title[1]} value={vmax:.3e}, at x={xi[max_l][0][0]}')
        
    if abs(xi[msl[2]]-xi[msl[1]])<2*dx:
        if abs(yi2[msl[2]]-0.)<abs(yi2[msl[1]]-0.): msl.pop(1)
        else: msl.pop(2) 
    if len(msl) > 2:
        if abs(xi[msl[2]]-xi[msl[0]])<2*dx:
            if abs(yi2[msl[2]]-0.)<abs(yi2[msl[0]]-0.): msl.pop(0)
            else: msl.pop(2)  
    if len(msl) > 1:
        if abs(xi[msl[1]]-xi[msl[0]])<2*dx:
            if abs(yi2[msl[1]]-0.)<abs(yi2[msl[0]]-0.): msl.pop(0)
            else: msl.pop(1)
         
    ax2 = ax1.twinx()  # instantiate second axes sharing the same x-axis

    color = 'blue'
    ax2.set_ylabel(title[1], color=color, fontsize=fs) 
    ax2.plot(xi, yi2, color=color)
    ax2.plot(xi[max_l], yi2[max_l], 'r*', markersize=4)
    ax2.plot(xi[msl], yi2[msl], 'r*', markersize=4)
    ax1.plot(xi[msl],  yi[msl], 'r*', markersize=4)
    ax2.plot(xi[0], yi2[0], 'ro', markersize=2)
    ax1.plot(xi[0],  yi[0], 'ro', markersize=2)
    ax2.plot(xi[-1], yi2[-1], 'ro', markersize=2)
    ax1.plot(xi[-1],  yi[-1], 'ro', markersize=2)
    ax2.grid(color='r',ls=':',lw=.5, which='both') # Use both tick  
    ax2.tick_params(axis='x', labelcolor=color, labelsize=fs)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=fs)
    np.set_printoptions(formatter={'float': '{: 0.3e}'.format})
    print(f'Extreme {title[0]} values={yi[msl]},\n    at x={xi[msl]}')
    print(f'Critical {title[1]} values={yi2[msl]},\n    at x={xi[msl]}')
    print(f'{title[0]} values at boundary ={yi[0], yi[-1]}')
    print(f'{title[1]} values at boundary ={yi2[0], yi2[-1]}\n')

def plot2curveS0(u, xL, xR, title="f_title"):
    dx = 0.01
    xi = np.arange(0., 1.+dx, dx)
    uf = sp.lambdify((x), u[0], 'numpy') #convert Sympy f to numpy f 
    yi = uf(xi)
    if type(yi) == int: # or len(yi) == 1:
        yi = float(yi)*np.ones_like(xi)

    fig, ax1 = plt.subplots(figsize=(5.,1.), dpi=300)
    fs = 8     # fontsize
    color = 'black'
    ax1.set_xlabel('location x', fontsize=fs)
    ax1.set_ylabel(title[0], color=color, fontsize=fs)
    ax1.plot(xi, yi, color=color)
    ax1.grid(color='r',ls=':',lw=.3, which='both') # Use both tick 
    ax1.tick_params(axis='x', labelcolor=color, labelsize=fs)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=fs)

    uf = sp.lambdify((x), u[1], 'numpy') #convert Sympy f to numpy f 
    yi2 = uf(xi)
    if type(yi2) == int: # or len(yi2) == 1:
        yi2 = float(yi2)*np.ones_like(xi)
        
    m1, m2, m3 = np.partition(abs(yi2), 2)[0:3]
    msl=[np.where(abs(yi2)==m1)[0][0],np.where(abs(yi2)==m2)[0][0],
             np.where(abs(yi2)==m3)[0][0]]
    
    vmax = yi[yi.argmax()]
    max_l = np.argwhere(yi == vmax)
    ax1.plot(xi[max_l],  yi[max_l], 'r*', markersize=4)
    print(f'Maximum {title[0]} value={vmax:.3e}, at x={xi[max_l][0][0]}')
    vmax = yi2[yi2.argmax()]
    max_l = np.argwhere(yi2 == vmax)
    print(f'Maximum {title[1]} value={vmax:.3e}, at x={xi[max_l][0][0]}')
        
    if abs(xi[msl[2]]-xi[msl[1]])<2*dx:
        if abs(yi2[msl[2]]-0.)<abs(yi2[msl[1]]-0.): msl.pop(1)
        else: msl.pop(2) 
    if len(msl) > 2:
        if abs(xi[msl[2]]-xi[msl[0]])<2*dx:
            if abs(yi2[msl[2]]-0.)<abs(yi2[msl[0]]-0.): msl.pop(0)
            else: msl.pop(2)  
    if len(msl) > 1:
        if abs(xi[msl[1]]-xi[msl[0]])<2*dx:
            if abs(yi2[msl[1]]-0.)<abs(yi2[msl[0]]-0.): msl.pop(0)
            else: msl.pop(1)
         
    ax2 = ax1.twinx()  # instantiate second axes sharing the same x-axis

    color = 'blue'
    ax2.set_ylabel(title[1], color=color, fontsize=fs) 
    ax2.plot(xi, yi2, color=color)
    ax2.plot(xi[max_l], yi2[max_l], 'r*', markersize=4)
    ax2.plot(xi[msl], yi2[msl], 'r*', markersize=4)
    ax1.plot(xi[msl],  yi[msl], 'r*', markersize=4)
    ax2.plot(xi[0], yi2[0], 'ro', markersize=2)
    ax1.plot(xi[0],  yi[0], 'ro', markersize=2)
    ax2.plot(xi[-1], yi2[-1], 'ro', markersize=2)
    ax1.plot(xi[-1],  yi[-1], 'ro', markersize=2)
    ax2.grid(color='r',ls=':',lw=.5, which='both') # Use both tick  
    ax2.tick_params(axis='x', labelcolor=color, labelsize=fs)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=fs)
    np.set_printoptions(formatter={'float': '{: 0.3e}'.format})
    print(f'Extreme {title[0]} values={yi[msl]},\n    at x={xi[msl]}')
    print(f'Critical {title[1]} values={yi2[msl]},\n    at x={xi[msl]}')
    print(f'{title[0]} values at boundary ={yi[0], yi[-1]}')
    print(f'{title[1]} values at boundary ={yi2[0], yi2[-1]}\n')
    #Vmax = max(max(yi[msl]),yi[0], yi[-1])
    #V_xmax = max(max(yi2[msl]),yi2[0], yi2[-1])
    #print(f'Maximum {title[0]} values={Vmax}')
    #print(f'Maximum {title[1]} values={V_xmax}\n') 

def plotbeam2by2(u, axis,iax=0, jax=0, title="f_title"):
    '''Plots 4 curves in 2 by 2 arrangement. Example: 
    title = [["deflection", "rotation"], ["moment", "shear force"]]
    fig, axis = plt.subplots(2,2,figsize=(10.,5), dpi=300)
    for i in range(len(title)):
        for j in range(len(title[0])):
            plotbeam(vx, axis, iax=i, jax=j, title=title[i][j]) 
    fig.tight_layout()
    plt.savefig('images/beam_cq.png', dpi=500)
    plt.show()'''
    spacing = 0.5       # grid line spacing. 
    minorLocator = MultipleLocator(spacing)
    xi = np.arange(0., 1., .01)
    uf = lambdify((x), u[iax*2+jax], 'numpy') #convert Sympy f to numpy f 
    yi = uf(xi)
    if type(yi) == int or len(yi) == 1:
        yi = float(yi)*np.ones_like(xi)
    axis[iax,jax].plot(xi, yi, label=title)
    axis[iax,jax].set_axisbelow(True)
    axis[iax,jax].set_xlabel('x')
    axis[iax,jax].yaxis.set_minor_locator(minorLocator) # Set minor tick 
    axis[iax,jax].xaxis.set_minor_locator(minorLocator)
    axis[iax,jax].grid(color='r',ls=':',lw=.5, which='both')    
    #axis[iax,jax].set_title(title)
    axis[iax,jax].legend()

def plotbar0(u, axis,iax=0, title="f_title"):
    '''Plots 2 curves. Example:
    title = ["Displacement", "Strain"]
    fig, axis = plt.subplots(1,2,figsize=(10.,1.8), dpi=300)
    for i in range(len(title)):
        plotbar(ux, axis,iax=i, title=title[i]
    plt.show())''' 
    
    spacing = 0.5       # grid line spacing. 
    minorLocator = MultipleLocator(spacing)
    xi = np.arange(0., 1., .01)
    uf = lambdify((x), u[iax], 'numpy') #convert Sympy f to numpy f 
    yi = uf(xi)
    if type(yi) == int or len(yi) == 1:
        yi = float(yi)*np.ones_like(xi)
    axis[iax].plot(xi, yi)
    axis[iax].set_axisbelow(True)
    axis[iax].set_xlabel('x')
    axis[iax].yaxis.set_minor_locator(minorLocator) # Set minor tick 
    axis[iax].xaxis.set_minor_locator(minorLocator)
    axis[iax].grid(color='r',ls=':',lw=.5, which='both') # Use both tick  
    axis[iax].set_title(title)

def transferM(theta, about = 'z'):
    '''Create a transformation matrix for coordinate transformation\
    Input theta: rotation angle in degree \
          about: the axis of the rotation is about \
    Return: numpy array of transformation matrix of shape (3,3)'''
    from scipy.stats import ortho_group
    
    n = 3   # 3-dimensonal problem
    c, s = np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))
    
    if about == 'z':
        # rotates about z by theta 
        T = np.array([[ c, s, 0.],
                      [-s, c, 0.],
                      [0.,0., 1.]]) 
    elif about == 'y':
        # rotates about y by theta 
        T = np.array([[ c, 0.,-s],
                      [ 0, 1.,0.],
                      [ s, 0., c]])  
    elif about == 'x':
        # rotates about x by theta 
        T = np.array([[ 1.,0., 0.],
                      [ 0., c, s],
                      [ 0.,-s, c]])
    else: # randomly generated unitary matrix as transformation matrix:
        T = ortho_group.rvs(dim=n)          # Generate a random matrix
        T[2,:] = np.cross(T[0,:], T[1,:]) # Enforce the right-hand rule
        
    return T, about


def rotationM(theta, about = 'z'):
    '''Create a rotation matrix for coordinate transformation (numpy)\
    Input theta: rotation angle \
          about: the axis of the rotation is about \
    Return: numpy array of rotation matrix of shape (3,3)'''
    from scipy.stats import ortho_group
    
    n = 3        # 3-dimensonal problem
    c, s = np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))
    #T = np.zeros((n,n))
    
    if about == 'z':
        # rotates about z by theta 
        T = np.array([[ c, s, 0.],
                      [-s, c, 0.],
                      [0.,0., 1.]]) 
    elif about == 'y':
        # rotates about y by theta 
        T = np.array([[ c, 0., s],
                      [0., 1.,0.],
                      [-s, 0., c]])  
    elif about == 'x':
        # rotates about x by theta 
        T = np.array([[ 1.,0., 0.],
                      [ 0., c, s],
                      [ 0.,-s, c]])
    else: # randomly generated unitary matrix->rotation matrix, no theta
        T = ortho_group.rvs(dim=n)          # Generate a random matrix
        T[2,:] = np.cross(T[0,:], T[1,:])   # Enforce the righ-hand rule
        
    return T.T, about   # R=T.T # Rotation matrix

def rotationMs():
    '''define the transformation matrix T for symbolic computation'''

    a11, a12, a13 = sp.symbols("a11, a12, a13")
    a21, a22, a23 = sp.symbols("a21, a22, a23")
    a31, a32, a33 = sp.symbols("a31, a32, a33")
    

    T = sp.MatrixSymbol('T', 3, 3)
    T = sp.Matrix([[a11, a12, a13],
                [a21, a22, a23],
                [a31, a32, a33]])
    return T.T  # Rotation matrix R=T.T      

def rotationR(𝜃):
    '''Create a 2D sympy rotation matrix. 
    Input 𝜃: rotation angle, symbolic. \
    Return: Rotation matrix R of shape (2,2)'''
    R = sp.Matrix([[ sp.cos(𝜃), -sp.sin(𝜃)],
                   [ sp.sin(𝜃),  sp.cos(𝜃)]])  
    return R

def np_R(𝜃):
    '''Create a 2D numpy rotation matrix. 
    Input 𝜃: degree, rotation angle. \
    Return: Rotation matrix R of shape (2,2)'''
    R = np.array([[np.cos(np.deg2rad(𝜃)), -np.sin(np.deg2rad(𝜃))],
                  [np.sin(np.deg2rad(𝜃)),  np.cos(np.deg2rad(𝜃))]])  
    return R

def transferMs():
    '''define the transformation matrix T for symbolic computation'''

    a11, a12, a13 = sp.symbols("a11, a12, a13")
    a21, a22, a23 = sp.symbols("a21, a22, a23")
    a31, a32, a33 = sp.symbols("a31, a32, a33")
    
    T = sp.MatrixSymbol('T', 3, 3)
    T = sp.Matrix([[a11, a12, a13],
                [a21, a22, a23],
                [a31, a32, a33]])
    return T

def transferMts(𝜃):
    '''define the transformation matrix T for symbolic computation
       for given rotation angle 𝜃 with respect to z.'''

    T = sp.MatrixSymbol('T', 3, 3)
    T = sp.Matrix([[sp.cos(𝜃),  sp.sin(𝜃), 0],
                   [-sp.sin(𝜃), sp.cos(𝜃), 0],
                   [0,             0,      1]])
    return T

def transf_YPRs(𝜃, about = 'z'):
    '''Create a transformation matrix for coordinate transformation\
    Input:  𝜃, rotation angle, in Sympy \
            about, the axis of the rotation is about \
    Return: Sympy matrix of transformation matrix of shape (3,3)
    '''
    c, s = sp.cos(𝜃), sp.sin(𝜃)
    
    if about == 'z':
        # Yaw: rotates about z by 𝜃 
        T = sp.Matrix([[ c, s, 0],
                       [-s, c, 0],
                       [ 0, 0, 1]]) 
    elif about == 'y':
        # Pitch: rotates about y by 𝜃 
        T = sp.Matrix([[ c, 0,-s],
                       [ 0, 1, 0],
                       [ s, 0, c]])  
    elif about == 'x':
        # roll: rotates about x by 𝜃 
        T = sp.Matrix([[ 1, 0, 0],
                       [ 0, c, s],
                       [ 0,-s, c]])
    else: # randomly generated unitary matrix as transformation matrix:
        print("Rotation axis must be given: z, y, or x")
        
    return T, about

def Tensor2_transfer(T,S):
    '''Sybolic coordinate transformation for 2nd order tensors
    '''
    S = np.tensordot(T, S, axes=([1],[0]))
    S = np.tensordot(S, T, axes=([1],[1]))
    return S

def mgs_QR(A):
    '''QR decomposition via modified Gram-Schmidt orthogonalization.
    This is a minimalistic version for principle demonstration. 
    A(m, n): Inputs, an numpy array 
    Q(m, n) and R(n, n): Outputs, numpy arrays 
    Usage: Q, R = mgs_QR(A)''' 
    
    m, n = A.shape               # get the dimensions 
    dtypeA = A.dtype             # get the data type of A
    v = A.copy().astype(dtypeA)    
    Q = np.zeros((m, n),dtype = dtypeA)      # initialization
    R = np.zeros((n, n),dtype = dtypeA)
    
    for i in range(n):           # n iterations needed in GS via
        R[i,i] = lg.norm(v[:,i]) # recursive improvements
        Q[:,i] = v[:,i]/R[i,i]   # normalization
        for j in range(i+1, n):  # 𝐏⊥𝑢 on each component just once
            R[i,j] = np.dot(Q[:,i], v[:,j])  
            v[:,j] -= R[i,j]*Q[:,i]

    return Q, R


def A_exam(A, atol = 1.e-10, n_dgt=4):
    '''Function to examing the properties of a given matrix A'''
    from sympy import Matrix 
    np.set_printoptions(precision=n_dgt,suppress=True)
    print(f'The original matrix:\n{A}')
    B = Matrix(A) 
    A_rank = B.rank()      
    print(f'Rank of the original matrix:{A_rank}')  
    
    e, V = lg.eig(A)
    print(f'Eigenvalues e:{e}')
    print(f'Eigenvectors V:\n{V}')
    print(f'Condition number of V={lg.cond(V)}')
    B = Matrix(V)
    V_rank = B.rank(iszerofunc= lambda x: abs(x)<atol) 
    print(f'Rank of eigenvector matrix V ={V_rank}')  
    print(f'Determinant of the original matrix ={np.prod(e)}')
    return e, V, A_rank, V_rank 

def AI_exam(A, e, ie, atol = 1.e-10, n_dgt=4):
    ''' Function to examing the properties of [A-lambda*I]'''
    from sympy import Matrix 
    np.set_printoptions(precision=n_dgt,suppress=True)
    AI = A-e[ie]*np.eye(len(e))
    print(f'Matrxi [A-lambda*I]:\n{AI}')
    B = Matrix(AI)
    nullity = B.shape[1] - B.rank(iszerofunc= lambda x: abs(x)<atol) 
    print(f'Nullity of [A-("+str(e[ie])+")*I]={nullity}')  

    # If want to further examine the nullpace: 
    if nullity == 0:               
        print(f'The nullpace has only the zero-vector!')
    else: 
        NullSpace=B.nullspace(iszerofunc= lambda x: abs(x)<atol) # list
        print(f'Null Space:{NullSpace[0]}')         # Select the Matrix 
        print(f'Is Ax=0? {B@NullSpace[0]}')     # check if in nullspace

def A_diag(A, e, V, n_dgt=4):
    '''Produce diagonalized A, given e and V can then check the \
    correctness'''
    np.set_printoptions(precision=n_dgt,suppress=True)
    print(f'Inverse of the eigvenvector matrix:\n{lg.inv(V)}') 
    print(f'Original matrix:\n{A}')
    E = np.diag(e) # matrix Lambda with eigenvalues at diagonal
    print(f'Diagonalized, with eigenvalues at diagonal:\n{E}')
    A_recovered = V@E@lg.inv(V)
    print(f'Reconstructed A via EVD:\n{A_recovered}')

def eig_diag(A, atol = 1.e-10, n_dgt=4):
    '''Perform EVD for matrix A, and check the correctness'''
    np.set_printoptions(precision=n_dgt,suppress=True)
    print(f'Original matrix:\n{A}',f'Rank={lg.matrix_rank(A)}')
    e, V = lg.eig(A)
    V_inv = lg.inv(V)               
    print(f'Eigenvalues, diagonalized matrix:\n{np.diag(e)}')
    print(f'The eigenvector matrix:\n{V}',f'Cond={lg.cond(V)}')
    print(f'Inverse of the eigenvector matrix:\n{V_inv}')
    Ae = V@np.diag(e)@V_inv  # Reconstructed matrix
    print(f'Are eigenvalues correct?\n{np.isclose(Ae,A,atol=atol)}')
    return e, V, V_inv 

def printE(e, n_dgt=4):
    '''Print out eigenvalues with multiplicities for sympy.dict
    Ref: https://stackoverflow.com/questions/48491577/
    printing-the-output-rounded-to-3-decimals-in-sympy'''
    np.set_printoptions(precision=n_dgt,suppress=True)
    fstr = '{: 0.'+str(n_dgt)+'f}'
    np.set_printoptions(formatter={'float': fstr.format})
    #print(f'Eigenvalues and multiplicities:{e}')
    #ea = np.array(list(e.items()))
    ei=np.array(list(e.keys()),dtype=float) #dict->list->numpy array
    me=np.array(list(e.values()), dtype=int)
    print(f'Eigenvalues={ei}, m={me}')
    n_e = np.sum(me, axis=0)             # number of eigenvalues
    print(f'Totoal number of eigenvalues ={n_e}') 
    #printM(sp.Matrix(ei).T)
    return ei, me, n_e                   #in numpy array

def printM0(A, p_text='Matrix A', n_dgt=4):
    '''Print out a sympy.Matrix A in latex form
    Ref: https://stackoverflow.com/questions/48491577/
    printing-the-output-rounded-to-3-decimals-in-sympy'''
    from sympy import init_printing, Float, Number
    init_printing(wrap_line=True,scale=0.85)
    print(p_text)
    return A.xreplace({n.evalf() : n if type(n)==int else 
           Float(n, n_dgt) for n in A.atoms(Number)}) 

def printM(A, *text, n_dgt=4):
    '''Print out a sympy.Matrix A in latex form
    Ref: https://stackoverflow.com/questions/48491577/
    printing-the-output-rounded-to-3-decimals-in-sympy'''
    from sympy import init_printing, Float, Number
    init_printing(wrap_line=True,scale=0.85)
    if len(text) !=0: print(text[0])
    return A.xreplace({n.evalf() : n if type(n)==int else 
           Float(n, n_dgt) for n in A.atoms(Number)}) 

def printCM(y_test,y_predict):
    print(f"Classification Report: \n" 
          f"{classification_report(y_test,y_predict,labels=list(range(10)))}")

    # Compute the Modified confusion matrix: 
    digids = [i for i in range(11)] #; digids[-1] = '/'
    digids = np.array(digids)[:,np.newaxis]
    allLabels = [str(i) for i in range(10)] #print(y_test, y_predict, allLabels)
    c_matrix = confusion_matrix(y_test,y_predict, labels=allLabels)
    supports = np.sum(c_matrix, axis=1)
    TP_FN = np.sum(c_matrix, axis=1)  # supports
    c_matrix = np.hstack((c_matrix, TP_FN[:,np.newaxis]))
    TP_FP = np.sum(c_matrix, axis=0)
    c_matrix = np.vstack((c_matrix, TP_FP[np.newaxis,:]))
    c_matrix = np.hstack((digids, c_matrix))
    #print(f"Confusion matrix:\n{c_matrix}"); print(f"Supports:\n{supports}")
    print(f"Extended confusion matrix:\n{c_matrix}")

def eval_svm(y_test, y_pred, warning = False):
    '''Evaluate of the performance of a model using Sklearn.'''
    # Overall accuracy of the prediction
    print(f"Overall accuracy = {accuracy_score(y_test, y_pred)*100} %")

    # Detailed classification results and performance report
    if warning: warnings.filterwarnings("ignore")

    from sklearn.metrics import classification_report, confusion_matrix
    print(f"Classification Report: \n" 
          f"{classification_report(y_test,y_pred,labels=list(range(10)))}")

    # Compute the Modified confusion matrix: 
    digids = [i for i in range(11)] #; digids[-1] = '/'
    digids = np.array(digids)[:,np.newaxis]
    allLabels = [str(i) for i in range(10)] #print(y_test, y_pred)
    
    c_matrix = confusion_matrix(y_test,y_pred, labels=allLabels)
    supports = np.sum(c_matrix, axis=1)
    
    TP_FN = np.sum(c_matrix, axis=1)  # supports
    c_matrix = np.hstack((c_matrix, TP_FN[:,np.newaxis]))
    TP_FP = np.sum(c_matrix, axis=0)
    
    c_matrix = np.vstack((c_matrix, TP_FP[np.newaxis,:]))
    c_matrix = np.hstack((digids, c_matrix))
 
    print(f"Modified confusion matrix:\n{c_matrix}")

def printl(list_num, *arg, n_d=4):
    '''print given list, list_num, with numerical numbers in format of .n_df'''
    str_list = [f"{num:.{str(n_d)}f}" for num in list_num]
    str_list = [float(num) for num in str_list]
    print(*arg, str_list)

def printt(tuple_num, *arg):
    '''print given tuple_num, with numerical numbers in format of .4f'''
    str_tuple = [f"{num:.4f}" for num in tuple_num]
    str_tuple = [float(num) for num in str_tuple]
    print(*arg, str_tuple)

def printA(A, p_text='Matrix A:', n_dgt=4):
    '''Print out a numpy.Matrix A in latex form
    Ref: https://stackoverflow.com/questions/48491577/
    printing-the-output-rounded-to-3-decimals-in-sympy'''
    from sympy import init_printing, Float, Number, Matrix
    np.set_printoptions(precision=n_dgt,suppress=True)
    init_printing(wrap_line=True,scale=0.85)
    if p_text != "None":
        print(p_text)
    return Matrix(A).xreplace({n.evalf() : n if type(n)==int else 
           Float(n, n_dgt) for n in Matrix(A).atoms(Number)}) 

def A_eig(A, p_Av=True, atol=1e-12, n_dgt=4):
    np.set_printoptions(precision=n_dgt, suppress=True)
    print(f'Original matrix:\n{A}')
    e, V = lg.eig(A)      # Invoke np.linalg.eig to compute eigenvalues
    print(f'Eigenvalues of the matrix:{e}')
    print(f'Eigenvectors of the matrix:\n{V}')
    if p_Av: 
        print(f'Av:\n{A@V}')   # or use A.dot(v)
        print(f'Lambda*v:\n{V@np.diag(e)}')            # to get e_i*v_I
        
    print(f'Is Av=Lambda*v?\n{np.isclose(A@V,V@np.diag(e),atol=1e-12)}') 
    return e, V

def printEig(e, V):
    '''Print out eigenvalues and its corresponding eigenvectors'''
    np.set_printoptions(precision=3)
    for eig_val, eig_vec in zip(e, V.T):
        print(f'e={eig_val:0.3f}, ev={eig_vec}')

def sp_vcts(V_sp):
    '''Get eigenvectors from sympy eigenvectors V_sp in a dict,\
       and its geometric multiplicity nv_np'''
    V_list = []
    nv = []
    for tup in V_sp:
        nv.append(len(tup[2]))
        for v in tup[2]:
            V_list.append(list(v))
    V_np = np.array(V_list)
    nv_np =np.array(nv)
    return V_np.T, nv_np  

def sp_nulls(V_sp):              #in numpy array
    '''Get eigenvectors from sympy matrix V_sp in a list'''
    V_list = []
    for v in V_sp:
        V_list.append(list(v))
    V_np = np.array(V_list)  
    return V_np.T 

def svd_diag(A, atol = 1.e-10):
    '''Perform SVD for arbitrary matrix A and check the correctness'''
    U, s, VT = lg.svd(A)           #SVD
    print(f"Original Matrix:\n{A}, lg rank={lg.matrix_rank(A)}")
    S = np.diag(s)
    r = len(s)                     # rank of the matrix
    print(f"SVD, left-singular vectors (after cut-off):\n{U[:,0:r]}")
    print(f"SVD, Singular values (diagonal):\n{s}")
    print(f"SVD, right-singular vectors (after cut-off):\n{VT[0:r,:]}")
    As = U[:,:r]@S[:r,:r]@VT[:r,:] #Reconstruction of the matrix
    print(f"SVD, U@S@VT:\n{As}")
    print(f"Is U@S@VT the original? \n{np.isclose(A,As,atol=atol)}")
    return U, s, VT

def null_exam0(A):
    '''Examination of numpy array A by performing row-echolen-from.  
    Tested on two singular marices: 
    #A = np.array([[1,-0.2, 0], [0.1, 1,-0.5], [0.1, 1,-0.5]]) and 
    #A=np.array([[1,-0.2,0,0],[0.1,1,-0.5,0],[0.1,1,-0.5,0],[0.1,1,-0.5,0]])
    And one nonsingular matrix:
    #A = np.array([[1, 1], [0, 1]])        # the shear matrix
    #A = np.array([[1,-0.2, 0], [0.1, 1,-0.5], [0.1, 1,-0.8]])
    '''
    from sympy import Matrix
    #np.set_printoptions(precision=3)
    n = np.shape(A)
    M = Matrix(A)
    M_rref = M.rref()
    print("Row echelon form of A with pivot columns:{}".format(M_rref))
    # Basis for null-space (see the textbook by Gilbert Strong)
    r = np.size(M_rref[1])
    print(A,'shape:',n,'\nRank=',r,', or dimension of C(A)')
    nd = n[1] - r
    print('Dimension of the null-space N(A), or nullity =',nd)
    if nd == 0: 
        print('Thus, the null-space has only the zero vector')
    else: 
        N = - M_rref[0][:,r:n[1]+1]
        print('Basis vector for the null-space:\n',N)
        print(n[1],np.shape(N), np.eye(nd))
        N[r:n[1],0:nd] = np.eye(n[1]-r)
        print('Basis vector for the null-space:\n',N)
        print('Transformed basis vector of the null-space by A:\n', A.dot(N),
              ', N is orthogonal to all the row-vecors of A: A@N=0')
        print('N is also orthogonal to the rref of A:\n',M_rref[0]@N)

###########################################################################

def plot_fs(x, y, labels, xlbl, ylbl, *p_name, ioff=False):
    '''plot multiple x-y curves in one figure.
    x: x data, list of 1D numpy array;
    y: y data, list of 1D numpy array 
    labels: labels for these curves
    xlbl: for ax.set_xlabel()
    ylbl: for ax.set_ylabel()
    p_name: string, name of the plot. 
    '''
    plt.ion()
    if ioff: plt.ioff()

    colors = ['b', 'r', 'g', 'c', 'm','k','y','b','r','k','m'] 
    plt.rcParams.update({'font.size': 5})         # settings for plots
    fig_s = plt.figure(figsize=(4,2.5))   
    ax = fig_s.add_subplot(1,1,1)
    for i in range(len(y)):
        plt.plot(x[i], y[i], label=labels[i], color=colors[i],lw=0.9) 
    
    ax.grid(c='r', linestyle=':', linewidth=0.5)
    ax.axvline(x=0, c="k", lw=0.6);ax.axhline(y=0, c="k", lw=0.6)
    ax.set_xlabel(xlbl); ax.set_ylabel(ylbl)
    ax.legend() #loc='center right', bbox_to_anchor=(1, 0.5)) 
    #plt.title('x-y curves'+p_name+'')

    #plt.savefig('images/'+p_name[0]+'.png',dpi=500,bbox_inches='tight')
    plt.savefig(p_name[0]+'.png',dpi=500,bbox_inches='tight')
    if not ioff: plt.show()

def plot_fs0(x, y, labels, xlbl, ylbl, *p_name):
    '''plot multiple x-y curves in one figure.
    x: x data points, 1D numpy array;
    y: y data, list of 1D numpy array 
    labels: labels for these curves
    xlbl: for ax.set_xlabel()
    ylbl: for ax.set_ylabel()
    p_name: string, name of the plot. 
    '''
    colors = ['b', 'r', 'g', 'c', 'm'] 
    plt.rcParams.update({'font.size': 5})         # settings for plots
    fig_s = plt.figure(figsize=(4,2.5))   
    ax = fig_s.add_subplot(1,1,1)
    for i in range(len(y)):
        plt.plot(x, y[i], label=labels[i], color=colors[i]) 
    
    ax.grid(c='r', linestyle=':', linewidth=0.5)
    ax.axvline(x=0, c="k", lw=0.6);ax.axhline(y=0, c="k", lw=0.6)
    ax.set_xlabel(xlbl); ax.set_ylabel(ylbl)
    ax.legend() #loc='center right', bbox_to_anchor=(1, 0.5)) 
    #plt.title('x-y curves'+p_name+'')

    plt.savefig('images/'+p_name[0]+'.png',dpi=500,bbox_inches='tight')
    plt.show()

def plot_fij(fig_s, x, y, labels, xlbl, ylbl,fij):
    '''plot multiple x-y curves in multiple figures.
    fig_s: a plt.figure() object
    x: x data points, 1D numpy array
    y: y data, list of 1D numpy array 
    labels: labels for these curves
    xlbl: for ax.set_xlabel()
    ylbl: for ax.set_ylabel()
    fij: figure grids. First 2 idexes gives the grid, 3rd is for 
         figure number. The last one controls whether plot xlabel. 
    '''
    colors = ['b', 'r', 'g', 'c', 'm', 'k'] 

    ax = fig_s.add_subplot(fij[0],fij[1],fij[2])  #(1,1, 1)
    for i in range(len(y)):
        ax.plot(x, y[i], label=labels[i], color=colors[i]) 
    
    ax.grid(c='r', linestyle=':', linewidth=0.5)
    #ax.axvline(x=0, c="k", lw=0.6);ax.axhline(y=0, c="k", lw=0.6)
    if fij[3] == 'xl': ax.set_xlabel(xlbl) 
    if fij[4] == 'yl': ax.set_ylabel(ylbl)
    ax.legend() #loc='center right', bbox_to_anchor=(1, 0.5)) 
    
    #plt.title('x-y curves'+p_name+'')
    #plt.savefig('images/'+p_name[0]+'.png',dpi=500,bbox_inches='tight')
    #plt.show()


def plotfig(axis, iax=0, title="f_title", xylim = 6):
    from matplotlib.ticker import MultipleLocator 
    xl, xr = -xylim, xylim                  # x left, x right for plots
    yf, yc = -xylim, xylim                  # y floor, y ceiling for plots
    spacing = 1        # grid line spacing. 
    minorLocator = MultipleLocator(spacing)
    axis[iax].set_axisbelow(True)
    axis[iax].yaxis.set_minor_locator(minorLocator) # Set minor tick locations
    axis[iax].xaxis.set_minor_locator(minorLocator)
    axis[iax].grid(ls=':',lw=.5, which = 'both')              # Use both ticks 
    axis[iax].legend()
    axis[iax].axis('square')       
    axis[iax].plot(0, 0, 'ro', markersize=2)
    axis[iax].set_xlim([xl,xr])
    axis[iax].set_ylim([yf,yc])
    axis[iax].set_title(title)


def drawArrow(axis,origin,vector,color='k',iax=0,xylim = 6,label='None'):
    aw, lw = 0.015*xylim, 0.09*xylim # 0.07, 0.3      # arrow width, line-width
    if label == 'None':
        axis[iax].arrow(origin[0],origin[1],vector[0],vector[1],width=aw,lw=lw, 
            color=color, fill=False, length_includes_head=True)
        labels = label
    else:
        labels = axis[iax].arrow(origin[0],origin[1],vector[0],vector[1],
                 width=aw,lw=lw, color=color, label=label, fill=False, 
                 length_includes_head=True)
    return labels
    
def drawArrow0(axis,origin,vector,color='k',iax=0,xylim = 6, label='No'):
    aw, lw = 0.015*xylim, 0.09*xylim # 0.07, 0.3      # arrow width, line-width
    if label == 'None':
        axis[iax].arrow(origin[0],origin[1],vector[0],vector[1],width=aw,lw=lw, 
            color=color, fill=False, length_includes_head=True)
    else:
        axis[iax].arrow(origin[0],origin[1],vector[0],vector[1],width=aw,lw=lw,
            color=color, label=label, fill=False, length_includes_head=True)
        #axis[iax].legend()
       
###############################################################################

def plotfig0(axis,iax=0, title="f_title",axislim = 6):
    xl, xr = -axislim, axislim                     # x left, x right for plots
    yf, yc = -axislim, axislim                  # y floor, y ceiling for plots
    spacing = 1        # grid line spacing. 
    minorLocator = MultipleLocator(spacing)
    axis[iax].set_axisbelow(True)
    axis[iax].yaxis.set_minor_locator(minorLocator) # Set minor tick locations
    axis[iax].xaxis.set_minor_locator(minorLocator)
    axis[iax].grid(ls='--',lw=1.2, which = 'both')    # Use both tick locations 
    axis[iax].legend()
    axis[iax].axis('square')
    axis[iax].plot(0, 0, 'ro', markersize=6)
    axis[iax].set_xlim([xl,xr])
    axis[iax].set_ylim([yf,yc])
    axis[iax].set_title(title)
    
def drawArrow00(axis,origin,vector,color='k',iax=0,label='None'):
    aw, lw = 0.08, 1.2                               # arrow width, line-width
    if label == 'None':
        axis[iax].arrow(origin[0],origin[1],vector[0],vector[1],width=aw,lw=lw, 
            color=color, fill=False, length_includes_head=True)
    else:
        axis[iax].arrow(origin[0],origin[1],vector[0],vector[1],width=aw,lw=lw,
            color=color, label=label, fill=False, length_includes_head=True)

def sequence_trunkI(rows, clms,start = 0):
    '''Use interges to create a sequence trunk matrix'''
    ST = [j for j in range(start,clms+start)]
    for i in range(1,rows):
        si = start + i
        STr= [j for j in range(si,clms+si)] 
        ST = np.vstack((ST,STr))
    return ST

def sequence_trunkA(A, clms, start = 0):
    '''A: a given 1D np.array; to create sequence trunk matrix'''
    rows = int(A.shape[0]/clms)
    ST = [j for j in A[start:clms+start]]
    for i in range(1,rows):
        si = start + i
        STr= [j for j in A[si:clms+si]]
        ST = np.vstack((ST,STr))
    return ST

def null_exam(A): #, A_name='A'):
    '''Examination of numpy array A by computing its row-echolen-from. 
    input: array A; return: N, null-space of A; nd: nullity of A.
    '''
    #A_name=varName(A, globals())[0]
    A_name=varName2(A)[0]
    n = np.shape(A)
    M_rref = sp.Matrix(A).rref()
    # Basis for null-space (see the textbook by Gilbert Strong)
    r = np.size(M_rref[1])
    print(f"Row echelon form of {A_name} with pivot columns:{M_rref}; "\
          f'{A_name}.shape:',n,'\nRank=',r,f'or dimension of C({A_name})')
    nd = n[1] - r
    print(f"Dimension of the null-space N({A_name}), or nullity = {nd}")
    
    i_old = [i for i in range(n[1])]    # original variables order
    iFC = [i for i in i_old if i not in set(M_rref[1])]
    i_new = list(M_rref[1]) + list(iFC) # changed variables order
    print(f"Free columns = {iFC}, i_old= {i_old}, i_new= {i_new}")
    
    # Form the basis vectors for the Null space: N=[-F  I].T
    if nd == 0: 
        N = 0; print(f"Thus, the null-space has only the zero vector")
    else: 
        N =np.array(-M_rref[0][:r, iFC])   # free columns for pivot vars 
        N = np.append(N, np.eye(nd), axis=0)  # add I for free variables
        print(' N shape:',np.shape(N), 'I(nd).shape:', np.eye(nd).shape)
        print(f"Basis vectors for the null-space:\n{N}")
        
        # Swap rows, if variable order is changed 
        for i in range(n[1]):
            if i_new[i] != i_old[i]:
                N[[i_new[i], i_old[i]], :] = N[[i_old[i], i_new[i]], :]
                i_new[i_new[i]], i_new[i_old[i]] = \
                i_new[i_old[i]], i_new[i_new[i]]
                print(f"Basis vectors for null-space after swap:\n{N}")
                
        print(f"Transformed basis vector of the null-space by {A_name}:\
              \n{A.dot(N)}, N is orthogonal to all row-vecors of {A_name}.")
        print(f"N is also orthogonal to rref of {A_name}:\n{M_rref[0]@N}")
        
    return N, nd

def simplifyM(M):     
    '''To simplify a Sympy matrix, by factor() each of its elements'''
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            M[i,j] = sp.factor(M[i,j]).subs(1.,1) # simplify the expression
            M[i,j] = sp.factor(M[i,j]).subs(1.,1) # simplify it again
    return M


def simplifyM0(M):     
    '''To simplify a Sympy matrix, by factor() each of its elements'''
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            M[i,j] = M[i,j].simplify()  # simplify the expression
    return M

def simplifyMd(M):                      
    '''To deep-simplify a Sympy matrix with fractions, by working on
    each of its elements'''
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            Mn,Md=sp.fraction(M[i,j])     # Simplify the expression
            Mn,Md=Mn.expand(),Md.expand() # for both numerator 
            Mn,Md=Mn.evalf(),Md.evalf()   # and denominator.
            M[i,j] = Mn/Md                # Put it back.
    return M

def simplifyMm(M):                        # added to grcords
    '''To multi-simplify a Sympy matrix with fractions, by working on
    each of its elements using factor(), expand(), evalf(), etc.'''
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):    # simplify the expression
            M[i,j] = sp.factor(M[i,j]).subs(1.0, 1) 
            M[i,j] = M[i,j].evalf().subs(1.0, 1)
            M[i,j] = sp.factor(M[i,j]).subs(1.0, 1)
            M[i,j] = sp.expand(M[i,j]).subs(1.0, 1)
            M[i,j] = sp.factor(M[i,j]).subs(1.0, 1)
    return M

def varName(var, allVarNames):
    '''return the name of a variable. Usage:varName(var,globals())[0]
    ref: https://stackoverflow.com/questions/18425225/
    getting-the-name-of-a-variable-as-a-string'''
    return [name for name in allVarNames if allVarNames[name] is var]

def varName2(var):
    '''return the name of a variable. Usage:varName2(var)[0]
    ref: https://stackoverflow.com/questions/18425225/
    getting-the-name-of-a-variable-as-a-string'''
    current_vars = inspect.currentframe().f_back.f_back.f_locals.items()
    return [name for name, value in current_vars if value is var]

def cholesky(A):
    '''Cholesky decomposition.
    This is only a minimalistic code for principle demonstration. 
    Input:   A(n, n), SPD, numpy array 
    Returns: L(n, n), L lower triangular 
    Usage:   L = Cholesky(A) ''' 
    n = A.shape[0]                 # get the dimension of A
    L = A.copy().astype(A.dtype)
    for i in range(n):
        for j in range(i+1,n):
            L[j:,j] = L[j:,j]-L[j:,i]*L[j,i].conj()/L[i,i]
            #L[j,j:]=L[j,j:]-L[i,j:]*L[i,j].conj()/L[i,i] # upper
            
        L[i:,i]=L[i:,i]/np.sqrt(L[i,i])
        #L[i,i:]=L[i,i:]/np.sqrt(L[i,i]) # upper
    return np.tril(L, k = 0)

def gs_LU(A):
    '''LU decomposition via Gaussin Elimination without Pivoting.
    This is only a minimalistic code for principle demonstration. 
    Input:   A(n, n), numpy array 
    Returns: L(n, n), lower triangular; U(n, n), upper
    Usage:   L, U = gs_LU(A) ''' 
    n = A.shape[0]                 # get the dimension of A
    U = A.copy().astype(A.dtype)
    L = np.eye(n,n).astype(A.dtype)
    for i in range(n-1):
        for j in range(i+1,n):
            L[j,i] = U[j,i]/U[i,i]
            U[j,i:]=U[j,i:]-L[j,i]*U[i,i:]
    return L, U

def det_minor(A, i, j):                        # in grcodes.py
    '''compute the determinant of the minor_ij of an numpy array''' 
    return lg.det(np.delete(np.delete(A,i,axis=0), j, axis=1))

def pure_QR0(A, atol=1.e-8, n_iter=1000):
    '''The pure QR for the Schur factorization. To be further tested!!!. 
    A(n, n): Inputs, an numpy array 
    A(n, n): Schur form; and Q(n, n): Unitary matrix''' 
    m, n = A.shape                           # get the dimensions 
    dtypeA = A.dtype                         # get the data type of A
    Acopy = A.copy().astype(dtypeA)    
    Q = np.zeros((m, n), dtype=dtypeA)       # initialization
    R = np.zeros((n, n), dtype=dtypeA)
    Qr = np.eye(n, dtype=dtypeA)          
    
    for i in range(n_iter):                  # control by iterations 
        Q, R = mgs_QR(Acopy) # or lg.qr() & h_QR() for QR factorization
        Acopy = R@Q                 # Recombine factors in revise order
        Qr = Qr@Q                   # buid up Q
        if lg.norm(Acopy[n-1,:n-1]) < atol:  # control by accuracy
            print(f"Converged at iteration = {i}"); break   
    return Acopy, Qr

def pure_QR(A, atol=1.e-12, n_iter=1000):
    '''The pure QR for the Schur factorization . 
    A(n, n): Inputs, an numpy array 
    A(n, n): Schur form; and Q(n, n): Unitary matrix''' 
    m, n = A.shape; dtypeA = A.dtype   # get dimensions & data-type of A
    Qr = np.eye(n, dtype=dtypeA)          
    for i in range(n_iter):                  # control by iterations 
        Q, R = lg.qr(A)                      # QR factorization
        A = R@Q                     # Re-combine factors in revise order
        Qr = Qr@Q                            # buid up Q
        if lg.norm(A[n-1,:n-1]) < atol:      # control by accuracy
            print(f"Converged at iteration = {i}"); break   
    return A, Qr

def forward_subst(L, b):
    '''Primary code for solve Ly=b via forward substitution.
    Inputs:  L(n, n)-- lower triagonal matrix, numpy array 
             b(n)--vector on the right-handside
    Leturns: x(n)--solution
    usage:   x = forward_subst(L, b)''' 
    n, x = b.shape[0], np.zeros_like(b) 
    x[0] = b[0]/L[0,0]
    for i in range(1,n): 
        r = 0.
        for j in range(0,i):
            r += L[i,j]*x[j]
        x[i] = (b[i]-r)/L[i,i]
    return x

def back_subst(R, b):                        # in grcodes.py
    '''Primary code for solve Rx=b via backward substitution.
    Inputs:  R(n, n)-- upper triagonal matrix, numpy array 
             b(n)--vector on the right-handside
    Returns: x(n)--solution
    usage:   x = back_subst(R, b)''' 
    n, x = b.shape[0], np.zeros_like(b) 
    x[-1] = b[-1]/R[-1, -1]
    for i in range(n-2,-1,-1):
        r = 0
        for j in range(i+1,n):
            r += R[i,j]*x[j]
        x[i] = (b[i]-r)/R[i,i]
    return x

def find_orth(A):
    '''This finds a vector that orthogonal to a given matrix A of m byn.
    returns: the normalized vector. 
    https://stackoverflow.com/questions/50660389/generate-a-vector-\
    that-is-orthogonal-to-a-set-of-other-vectors-in-any-dimension'''
    x0 = np.random.randn(A.shape[0], 1)
    A = np.hstack((A, x0))
    b = np.zeros(A.shape[1])
    b[-1] = 1
    print(A.shape, A.shape[1], b.shape)
    x = lg.lstsq(A.T, b, rcond=None)[0]
    return x/lg.norm(x)

def mapAA(A, f_name='images/paralgramA.png',fig_size=4,font_size=5):
    '''Plot the mapped parallelograms by given matrix A[i,j].
    It maps a square to k quadrants.'''
    
    plt.rcParams.update({'font.size': font_size})
    I = np.array([[1, 0], 
                  [0, 1]])
    fig, axis = plt.subplots(1,1,figsize=(fig_size,fig_size)) 
    axis.set_aspect(1)
    #axis.set_title('Mapped the unit square') 
    axis.grid(color='r', linestyle=':', linewidth=0.5)

    x1 = [0, A[0,0], A[0,0]+A[0,1],A[0,1], 0]  
    x2 = [0, A[1,0], A[1,0]+A[1,1],A[1,1], 0]
    axis.plot(x1, x2, label="Maped by A")
    
    x1 = [0, I[0,0], I[0,0]+I[0,1],I[0,1], 0]  
    x2 = [0, I[1,0], I[1,0]+I[1,1],I[1,1], 0]
    axis.plot(x1, x2, label="Square")
    
    mpl.rcParams['text.usetex'] = True # False
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'
    xloc = .01*fig_size; yloc = 0.2*fig_size
    dx = 3.8*font_size/72; dy = .6*font_size/72
    
    axis.text(xloc+dx, yloc+dy, str(A[0,0]))
    axis.text(xloc+dx+3.5*dy, yloc+dy, str(A[0,1]))
    
    axis.text(xloc+dx, yloc-1.2*dy, str(A[1,0]))
    axis.text(xloc+dx+3.5*dy, yloc-1.2*dy, str(A[1,1]))
    
    axis.text(xloc, yloc,\
              r'$\mathbf{A}=\begin{bmatrix}\;\;&\;\;\\&\end{bmatrix}$')
    
    axis.legend()#loc='lower center')#, bbox_to_anchor=(1.1, 0.9)) 
    plt.savefig(f_name, dpi=500)
    plt.show()

def lagrange_interp(x, y, x_eval):
    """Computes the Lagrange interpolating polynomial for a given set of 
    data points. Suggested by ChapGPT.
    ----------    
    Parameters
    x : array-like
        The x-coordinates of the data points.
    y : array-like
        The y-coordinates of the data points.
    x_eval : float or array-like
        The point(s) at which to evaluate the interpolating polynomial.
    -------        
    Returns
    p : float or array-like
        The value(s) of the Lagrange interpolating polynomial at x_eval.
    """
    n = len(x)
    p = 0.0
    
    for i in range(n):
        p_i = 1.0
        for j in range(n):
            if i == j:
                continue
            p_i *= (x_eval - x[j]) / (x[i] - x[j])
        p += y[i] * p_i
    
    return p


def lagrange_interp_diff(x, y, x_eval):
    """Computes the derivatives of the Lagrange interpolating polynomial 
    for a given set of data points. Suggested by ChapGPT. 
    ----------
    Parameters: 
    x : array-like
        The x-coordinates of the data points.
    y : array-like
        The y-coordinates of the data points.
    x_eval : float or array-like
        The point(s) at which to evaluate the derivatives of the 
        interpolating polynomial.
    -------
    Returns:
    p_deriv : float or array-like
        The derivative(s) of the interpolating polynomial at x_eval.
    """
    n = len(x)
    p_deriv = 0.0
    
    for i in range(n):
        p_i = 1.0
        p_i_deriv = 0.0
        
        for j in range(n):
            if i == j:
                continue
            p_i *= (x_eval - x[j]) / (x[i] - x[j])
            p_i_deriv += 1.0 / (x_eval - x[j])
        
        p_deriv += y[i] * p_i_deriv * p_i
    
    return p_deriv

## Example: 
#x = np.array([-1, 0, 1])  # Define a set of data points
#y = np.array([1, 0, 1])

## Evaluate interpolating polynomial and its derivative at some points
#x_eval = np.linspace(-1.5, 1.5, num=100)
#p = np.array([lagrange_interp(x, y, xi) for xi in x_eval])
#p_deriv = np.array([lagrange_interp_diff(x, y, xi) for xi in x_eval])

#plt.plot(x_eval, p, label='Lagrange Interpolating Polynomial')
#plt.plot(x_eval, p_deriv, label='Derivative of Interpolating Polynomial')
#plt.plot(x, y, 'ro', label='Data Points')
#plt.legend(); plt.show()

def centroid_I(hb, wft, wfb, tft, tfb, tw):
    '''Compute the centroid of an I-beam made of rectangles. 
    The reference coordinate is on the bottom edge with the left-bottom 
    corner as the origin. The code can be used for any composite area 
    with up to 3 rectangles. For a T beam, for example, just set tfb=0. 
    The whole area must be left-right symmetric. 
    Inputs: 
    hb:  beam height (total)
    wft: width of flange on top
    wfb: width of flange at bottom
    tft: thickness of flange on top
    tfb: thickness of flange at bottom
    tw:  web thickness
    Return: 
    x_bar, y_bar: the centroid of the composite area. 
    A: The total area
    Ai: areas of each rectangles 
    '''
    width_ai  = np.array([wft, tw,         wfb])            
    height_bi = np.array([tft, hb-tft-tfb, tfb])            
    xi_bar    = np.array([wfb/2,  wfb/2,  wfb/2])
    yi_bar    = np.array([hb-tft/2, (hb-tft-tfb)/2+tfb, tfb/2])

    Ai = width_ai*height_bi
    A = np.sum(Ai)
    print(f"Areas of sub-areas = {Ai}[L^2]")
    print(f"Areas of I-beam = {A}[L^2])")

    x_bar = np.sum(xi_bar*Ai)/A
    y_bar = np.sum(yi_bar*Ai)/A

    print(f"The centroid [L]):\n x = {x_bar} \n y = {y_bar}")
    return x_bar, y_bar, A, Ai

def secondMoments_I(hb, wft, wfb, tft, tfb, tw, cxi, cyi, sy=True):
    '''Compute the second moments of I-beam cross-sections made of  
    rectangles. The reference coordinate is on the bottom edge with  
    cthe left-bottom orner as the origin and x-axis on the bottom edge. 
    The code can be used for any composite area 
    with up to 3 rectangles. For a T beam, for example, just set tfb=0. 
    Inputs: 
    hb:  beam height (total)
    wft: width of flange on top
    wfb: width of flange at bottom
    tft: thickness of flange on top
    tfb: thickness of flange at bottom
    tw:  web thickness
    cxi, cyi: centers of each sub-area wrt the common coordinates. 
    sy:  if True (I-, T-beam), cxi, cyi are not used, computed here in. 
         Otherwise, the provided cxi, cyi will be used here.
    Return: 
    Ixx, Iyy, Ixy: 2nd moments
    x_bar, y_bar: the centroid  
    A: The total area
    Ai: areas of each rectangles
    '''
    width_ai  = np.array([wft, tw,         wfb])   
    height_bi = np.array([tft, hb-tft-tfb, tfb])   
    
    # when whole area is symmetric with respect to y-axis:     
    if sy:    #compute the centers (cxi, cyi) for each sub-area. 
        cxi = np.array([wfb/2,  wfb/2,  wfb/2])
        cyi = np.array([hb-tft/2, (hb-tft-tfb)/2+tfb, tfb/2])

    print(f"Center of sub-areas [L]:\n x = {cxi} \n y = {cyi}")
    
    Ai = width_ai*height_bi
    A = np.sum(Ai)
    print(f"Areas of sub-areas = {Ai} [L^2]")
    print(f"Areas of I-beam = {A} [L^2]")

    x_bar = np.sum(cxi*Ai)/A
    y_bar = np.sum(cyi*Ai)/A

    print(f"The centroid = ({x_bar}, {y_bar}) [L]")
    
    # shifts the coordinates to the centroid of the whole area:
    xi_bar = cxi - x_bar
    yi_bar = cyi - y_bar
    
    print(f"New center of sub-areas [L]:\n x = {xi_bar}\n y = {yi_bar}")
    
    Ixx = np.sum(width_ai*height_bi**3/12 + yi_bar**2 * Ai)
    Iyy = np.sum(height_bi*width_ai**3/12 + xi_bar**2 * Ai)
    Ixy = np.sum(xi_bar*yi_bar * Ai)
    
    print(f"2nd moments [L^4]:\n Ixx = {Ixx}\n Iyy = {Iyy}\n Ixy = {Ixy}")
    
    return Ixx, Iyy, Ixy, x_bar, y_bar, xi_bar, yi_bar, A, Ai


def Ixx2IXX(Ixx, Iyy, Ixy, 𝜃):
    '''Coordinate transformation for second moments of area.
    Ixx, Iyy, Ixy: for original (x,y) 
    IXX, IYY, IXY: for (X,Y) that is 𝜃-rotated from (x,y)
    '''
    c = np.cos(𝜃); s = np.sin(𝜃)
    c2= c**2;      s2= s**2;  cs = c*s
    IXX = Ixx*c2 + Iyy*s2 - 2*Ixy*cs 
    IYY = Ixx*s2 + Iyy*c2 + 2*Ixy*cs 
    IXY = (Ixx-Iyy)*cs + Ixy*(c2 - s2) 
    
    return IXX, IYY, IXY

def q_web(y, Q_Dweb, tw, hwu, hwl):
    '''computing the shear flow q at any location s in the web of 
    an I-beam.
    Q_Dweb: Q value of the web at the top junction
    tw: thickness of the web
    hwu: height of the web above the neutral axis
    hwl: height of the web blow  the neutral axis
    Example: Computing the force in the web of I-beam. 
    # Because q_web() has parameters, use partial() to fix the parameters: 
    q_web_params = partial(q_web, Q_Dweb=Q_Dweb, tw=tw, hwu=hwu, hwl=hwl)
    F_webG, error = si.quad(q_web_params, hwl, hwu)       # use Gauss int.
    print(f"Force in the web (kN): {F_webG*1e-3}, intgral error= {error}")
    print(f"The true shear sorce V (kN): {V*1e-3}")
    One may use a lambda function. for this, if it is used just for once.
    '''
    Q_sweb = Q_Dweb + tw*(hwu - y)*(hwu + y)/2
    q_s_web = V/Izz * Q_sweb
    return q_s_web
 

def solver1D4(E, I, by, l, v0, 𝜃0, vl, 𝜃l, V0, M0, Vl, Ml, key='c-c'):
    '''Solves the Beam Equation for integrable distributed body force: 
           u,x4=-by(x)/EI, with various boundary conditions (BCs): 
           s-s, c-c, s-c, c-s, f-c, c-f. 
    Input: EI: bending stiffness factor; by, body force; l, the length 
           of the beam; v0, 𝜃0, V0, M0, deflection, rotation, 
           (internal) shear force, (internal) moment at x=0; 
           vl, 𝜃l, Vl, Ml, are those at x=l. 
    Return: u, v_x, v_x2, v_x3, v_x4   up to 4th derivatives of v
    ''' 
    x = sp.symbols('x')
    c0, c1, c2, c3 = sp.symbols('c0, c1, c2, c3') #integration constant
    EI = E*I                   # I is the Iz in our chosen coordinates. 
    
    # Integrate 4 times:     
    v_x3= sp.integrate(by/EI,(x, 0, x))+ c0   #ci: integration constant   
    v_x2= sp.integrate( v_x3,(x, 0, x))+ c1          
    v_x = sp.integrate( v_x2,(x, 0, x))+ c2   
    v   = sp.integrate(  v_x,(x, 0, x))+ c3   
    
    # Solve for the 4 integration constants: 
    if   key == "s-s":
        cs=sp.solve([v.subs(x,0)-v0, v_x2.subs(x,0)-M0/EI,
                     v.subs(x,l)-vl, v_x2.subs(x,l)-Ml/EI],[c0,c1,c2,c3])
    elif key == "c-c":
        cs=sp.solve([v.subs(x,0)-v0, v_x.subs(x,0)-𝜃0,
                     v.subs(x,l)-vl, v_x.subs(x,l)-𝜃l], [c0,c1,c2,c3])
    elif key == "s-c":
        cs=sp.solve([v.subs(x,0)-v0, v_x2.subs(x,0)-M0/EI,
                     v.subs(x,l)-vl, v_x.subs(x,l)-𝜃l], [c0,c1,c2,c3])
        #print('solver1D4:',cs[c0],cs[c1],cs[c2],cs[c3])
    elif key == "c-s":
        cs=sp.solve([v.subs(x,0)-v0, v_x.subs(x,0)-𝜃0,
                     v.subs(x,l)-vl, v_x2.subs(x,l)-Ml/EI],[c0,c1,c2,c3])
    elif key == "c-f":
        cs=sp.solve([v.subs(x,0)-v0, v_x.subs(x,0)-𝜃0,
           v_x3.subs(x,l)+Vl/EI, v_x2.subs(x,l)-Ml/EI], [c0,c1,c2,c3])
    elif key == "f-c":
        cs=sp.solve([v_x3.subs(x,0)+V0/EI, v_x2.subs(x,0)-M0/EI,
                     v.subs(x,l)-vl, v_x.subs(x,l)-𝜃l], [c0,c1,c2,c3])
    else:
        print("Please specify boundary condition type.")
        sys.exit()
        
    # Substitute the constants back to the integral solutions   
    v   =   v.subs({c0:cs[c0],c1:cs[c1],c2:cs[c2],c3:cs[c3]})
    v   =   v.expand().simplify().expand()
    
    v_x = v_x.subs({c0:cs[c0],c1:cs[c1],c2:cs[c2],c3:cs[c3]})
    v_x = v_x.expand().simplify().expand()
    
    v_x2=v_x2.subs({c0:cs[c0],c1:cs[c1],c2:cs[c2],c3:cs[c3]})
    v_x2=v_x2.expand().simplify().expand()
    
    v_x3=v_x3.subs({c0:cs[c0],c1:cs[c1],c2:cs[c2],c3:cs[c3]})
    v_x3=v_x3.expand().simplify().expand()
    
    v_x4 = sp.diff(v_x3,x).expand()
    
    print("Outputs form solver1D4(): v, 𝜃, M, V, qy")
    return v,v_x,(EI*v_x2).expand(),(-EI*v_x3).expand(),(v_x4).expand()

def create_DPF3D(order):
    '''Creates polynomial Laplace basis functions that satisfies 
       ∇^2 = 0 with a desirved order that must be >=2.
       '''
    x, y, z = sp.symbols("x, y, z")
    𝜓 = sp.Function("𝜓")(x, y, z)           # a displacement-function
    c = sp.symbols("c0:99")    # coeffs, define more than what we need
    
    # create tri-nomials of "order"
    basis = Matrix([x**i*y**j*z**k for i in range(order+1) for j in 
            range(order+1) for k in range(order+1) if i+j+k==order])
    #print(f"Polynomials bases, complete order {order}:\n {basis.T}")
    
    cs = Matrix([c[i] for i in range(len(basis))])     # coeffs needed
    num_cs = (order+2)*(order+1)/2
    print(f'order={order}, No. of constants={len(cs)}, ' 
          f'should be (order+2)*(order+1)/2: {num_cs}')
    
    𝜓 = (cs.T@basis)[0]    # create the function by linear combination
    #print(f"Initial 𝜓 with complete order {order} for 𝜓:\n {𝜓}")
    
    # form the order-reduced polynomial bases after applying ∇^2:
    basis2 = Matrix([x**i*y**j*z**k for i in range(order+1) for j in 
            range(order+1) for k in range(order+1) if i+j+k==order-2])
    #print(f"Polynomials bases after ∇^2:\n {basis2.T} {len(basis2)}")
    
    nab2𝜓=(𝜓.diff(x,2) + 𝜓.diff(y,2) + 𝜓.diff(z,2)).expand() # ∇^2

    # collect terms for order-reduced bases, to group the coeffs  
    for i in range(len(basis2)):       
        nab2𝜓 =  nab2𝜓.collect(basis2[i]) #print(nab2𝜓)
    
    # form equations of relations of the coeffs. 
    eqs = [nab2𝜓.coeff(basis2[i]) for i in range(len(basis2))]  
    if len(eqs)==1: eqs = [nab2𝜓]
        
    sln_cs=sp.solve(eqs, cs)  #print(sln_cs)
    keys_sln = list(sln_cs.keys())                     # solution keys
    
    # get the remaining coefficients in the expression
    keys_csi  = list(filter(lambda x: x not in keys_sln, cs))
    ns = order*(order-1)/2
    print(keys_sln,"\n", len(keys_sln),"is order*(order-1)/2:",ns,"\n",
          keys_csi,"\n", len(keys_csi),"should be 2*order+1")
    
    𝜓_ci = 𝜓.subs(sln_cs).expand()           # get the 𝜓 with keys_csi
    
    # collect terms based on keys_csi, to group polynomials  
    for i in range(len(keys_csi)):
        𝜓_ci = 𝜓_ci.collect(keys_csi[i]) 
    
    # group polynomials, based on keys_csi
    DPFs=Matrix([(𝜓_ci.coeff(keys_csi[i])).expand() 
                 for i in range(len(keys_csi))]) 
    
    nab2𝜓=(DPFs.diff(x,2)+DPFs.diff(y,2)+DPFs.diff(z,2)).expand() #∇^2
    
    print(f'Is ∇^2 𝜓=0? {nab2𝜓}!\n  The number of bases = {len(DPFs)}')
    return DPFs

def create_BBFs(order):
    '''Creates polynomial biharmonic basis functions that satisfies 
       ∇^4=0 with a desirved order that must be larger or equal to 4.
       '''
    x, y = sp.symbols("x, y")
    𝜙 = sp.Function("𝜙")(x, y)               # a displacement-function
    c = sp.symbols("c0:20")    # coeffs, define more than what we need
    cs = Matrix([c[i] for i in range(order+1)]) #  coefficients needed
    print(f"order:{order}, terms:{len(cs)}, should be (order+1)")

    # form the complete polynomial bases for 𝜙: 
    basis = Matrix([x**i * y**(order-i) for i in range(order+1)])
    #print(f"Polynomials bases, complete order {order}:\n {basis.T}")

    # form the remaining polynomial bases after applying ∇^4:
    basis4 = Matrix([x**i * y**(order-4-i) for i in range(order-4+1)])
    #print(f"Polynomials bases after applying ∇^4:\n {basis4.T}")
    
    𝜙 = (cs.T@basis)[0]    # create the function by linear combination
    #print(f"Initial 𝜙 with complete order {order} for 𝜙:\n {𝜙}")

    nab2𝜙=(𝜙.diff(x,2) + 𝜙.diff(y,2)).simplify()                # ∇^2
    nab4𝜙=(nab2𝜙.diff(x,2)+nab2𝜙.diff(y,2)).simplify()          # ∇^4 

    # collect terms for remaining bases, so that the coeffs are grouped. 
    for i in range(len(basis4)):       
        nab4𝜙 =  nab4𝜙.collect(basis4[i]) 

    # form (order+1-4) equations of relations of coeffs. 
    eqs = [nab4𝜙.coeff(basis4[i]).expand() for i in range(len(basis4))]
    if len(eqs)==1: eqs = [nab4𝜙]
    sln_cs=sp.solve(eqs, cs)   
    keys_sln = list(sln_cs.keys())                     # solution keys
    
    # get the remaining coefficients in the expression
    keys_csi  = list(filter(lambda x: x not in keys_sln, cs))
    print(keys_sln, len(keys_sln),",  should be (order-4+1)","\n",
          keys_csi, len(keys_csi),",  should be:", 4)

    𝜙_final = 𝜙.subs(sln_cs).expand()               # get the final 𝜙

    # collect terms based on remaining coeffs, polynomials are grouped. 
    for i in range(len(keys_csi)):
        𝜙_final = 𝜙_final.collect(keys_csi) 

    # bases that satisfies ∇^4 = 0. They are with the last 4 coeffs:
    BBFs=Matrix([(𝜙_final.coeff(keys_csi[i])).expand() 
                 for i in range(len(keys_csi))]) 

    # check the results: 
    nab2𝜙=(BBFs.diff(x,2)+BBFs.diff(y,2)).expand()               # ∇^2
    nab4𝜙=(nab2𝜙.diff(x,2)+nab2𝜙.diff(y,2)).expand()            # ∇^4 
    print(f'Is ∇^4𝜙 = 0? {nab4𝜙.T}! Number of bases: {len(nab4𝜙)}')
    return BBFs

def field_plot0(expr, x_mesh, z_mesh, title=''):
    """Plot the field, by Guarín-Zapata, Nicolás, under MIT license.
    at https://github.com/nicoguaro/continuum_mechanics
    """

    # Lambdify the function
    var = [x, z]
    arr = [x_mesh, z_mesh]
    vals = OrderedDict(zip(var, arr)).values()
    expr_fun = lambdify(var, expr, "numpy")
    expr_mesh = expr_fun(*vals)

    # Determine extrema
    vmin = np.min(expr_mesh)
    vmax = np.max(expr_mesh)
    print("Minimum value in the domain: {:g}".format(vmin))
    print("Maximum value in the domain: {:g}".format(vmax))
    vmax = max(np.abs(vmax), np.abs(vmin)) 
    #print(f"log10(vmax)={vmax},{float(vmax)}")
    vmax = float(vmax)

    # Plotting
    fig = plt.gcf()
    levels = np.logspace(-1, np.log10(vmax), 20)
    levels = np.hstack((-levels[-1::-1], [0], levels))
    cbar_ticks = ["{:.2g}".format(level) for level in levels]
    cont = plt.contourf(x_mesh, z_mesh, expr_mesh, levels=levels,
           cmap="RdYlBu_r", norm=colors.SymLogNorm(0.1)) #,base=10))
    cbar = fig.colorbar(cont, ticks=levels[::2])
    cbar.ax.set_yticklabels(cbar_ticks[::2])
    plt.axis("image")
    plt.gca().invert_yaxis()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$z$")
    plt.title(title)
    return cont

def field_plot(expr, X, x_mesh, z_mesh, title=''):
    """Plot the field"""

    # Lambdify the function
    #var = [x, y]
    arr = [x_mesh, z_mesh]
    vals = OrderedDict(zip(X, arr)).values()
    expr_fun = lambdify(X, expr, "numpy")
    expr_mesh = expr_fun(*vals)

    # Determine extrema
    vmin = np.min(expr_mesh)
    vmax = np.max(expr_mesh)
    print("Minimum value in the domain: {:g}".format(vmin))
    print("Maximum value in the domain: {:g}".format(vmax))
    vmax = max(np.abs(vmax), np.abs(vmin)) 
    #print(f"log10(vmax)={vmax},{float(vmax)}")
    vmax = float(vmax)

    # Plotting
    fig = plt.gcf()
    levels = np.logspace(-1, np.log10(vmax), 20)
    levels = np.hstack((-levels[-1::-1], [0], levels))
    cbar_ticks = ["{:.2g}".format(level) for level in levels]
    cont = plt.contourf(x_mesh, z_mesh, expr_mesh, levels=levels,
           cmap="RdYlBu_r", norm=colors.SymLogNorm(0.1)) #,base=10))
    cbar = fig.colorbar(cont, ticks=levels[::2])
    cbar.ax.set_yticklabels(cbar_ticks[::2])
    plt.axis("equal")
    #plt.gca().invert_yaxis()
    #plt.xlabel(r"$x$")
    #plt.ylabel(r"$z$")
    plt.title(title)
    return cont

def contour_plot(X, Y, Z, x_lbl='x-axis', y_lbl='y-axis',title='title',lvls=40,flip='No',ioff=False):
    
    '''Generate a simple contour plot for given X, Y, Z
    '''
    vmin = np.nanmin(Z)                                 # find extrema
    vmax = np.nanmax(Z)
    print(f"Minimum value in the domain: {vmin:g}")
    print(f"Maximum value in the domain: {vmax:g}")
    
    plt.ion()
    if ioff: plt.ioff()
    fig, ax = plt.subplots()
    cntur = ax.contourf(X, Y, Z, cmap='RdYlBu_r',levels=lvls) #'viridis'
    plt.colorbar(cntur, label='Color Value')

    if flip =='y': 
        ax.invert_yaxis()
    elif flip == 'x': 
        ax.invert_xaxis()
    elif flip == 'xy': 
        ax.invert_xaxis()
        ax.invert_yaxis()

    ax.set_aspect('equal')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title(title)
    if not ioff: plt.show()
    
    return cntur

def curvaturef(F):
    '''Compute the curvature an implicit expression of a curve: F(x,y)=0.
    return: curvature
    Example: 
    from sympy import symbols, diff 
    x, y, a, b = symbols('x, y, a, b')                # Define symbols
    F = x**2/a**2 + y**2/b**2 - 1            # Define F for an ellipse
    curvature = curvaturef(F); print(curvature) # should be: 
    #Abs(8*y**2/(a**2*b**4)+8*x**2/(a**4*b**2))/(4*y**2/b**4+4*x**2/a**4)**1.5
    '''
    f_x = diff(F, x)                               # first derivatives
    f_y = diff(F, y)
    
    f_xx = diff(f_x, x)                           # second derivatives
    f_yy = diff(f_y, y)
    f_xy = diff(f_x, y)
    
    curvature = abs(f_xx*f_y**2+f_x**2*f_yy)/(f_x**2+f_y**2)**(3/2) 
    return curvature            # ⇑ the curvature formula


def hyper2e(x, s='s'):
    '''produces exponential form of hyperbolic functions for given x'''
    if s == 's': return (sp.exp(x) - sp.exp(-x))/2
    if s == 'c': return (sp.exp(x) + sp.exp(-x))/2
    if s == 't': return (sp.exp(x) - sp.exp(-x))/(sp.exp(x)+sp.exp(-x))
    if s =='ct': return (sp.exp(x) + sp.exp(-x))/(sp.exp(x)-sp.exp(-x))
    
def hyper2to1_dic(x):
    '''produces a dictionary for trigonometric expansion of 
    4 hyperbolic functions
    '''
    dic = {sp.sinh(x): sp.expand_trig(sp.sinh(x)),
           sp.cosh(x): sp.expand_trig(sp.cosh(x)),
           sp.tanh(x): sp.expand_trig(sp.tanh(x)),
           sp.coth(x): sp.expand_trig(sp.coth(x))}
    return dic

def hyper2e_dic(𝜉0, 𝜉, 𝜂):
    '''produces a dictionary for converting hyperbolic functions to
       exponential functions
    '''
    dic_e={sp.cosh(2*𝜉-2*𝜉0+2*sp.I*𝜂): hyper2e(2*𝜉-2*𝜉0+2*sp.I*𝜂, 'c'),
           sp.sinh(2*𝜉-2*𝜉0+2*sp.I*𝜂): hyper2e(2*𝜉-2*𝜉0+2*sp.I*𝜂, 's'),
           sp.cosh(2*sp.I*𝜂-2*𝜉-2*𝜉0): hyper2e(2*sp.I*𝜂-2*𝜉-2*𝜉0, 'c'),
           sp.sinh(2*sp.I*𝜂-2*𝜉-2*𝜉0): hyper2e(2*sp.I*𝜂-2*𝜉-2*𝜉0, 's'),
           sp.cosh(2*sp.I*𝜂+2*𝜉): hyper2e(2*sp.I*𝜂+2*𝜉, 'c'),
           sp.sinh(2*sp.I*𝜂+2*𝜉): hyper2e(2*sp.I*𝜂+2*𝜉, 's'),
           sp.cosh(𝜉-sp.I*𝜂): hyper2e(𝜉-sp.I*𝜂, 'c'),
           sp.sinh(𝜉-sp.I*𝜂): hyper2e(𝜉-sp.I*𝜂, 's'),
           sp.cosh(𝜉+sp.I*𝜂): hyper2e(𝜉+sp.I*𝜂, 'c'),
           sp.sinh(𝜉+sp.I*𝜂): hyper2e(𝜉+sp.I*𝜂, 's'),
           sp.tanh(𝜉+sp.I*𝜂): hyper2e(𝜉+sp.I*𝜂, 't'),
         
           sp.cosh(𝜉0-sp.I*𝜂): hyper2e(𝜉0-sp.I*𝜂, 'c'),
           sp.sinh(𝜉0-sp.I*𝜂): hyper2e(𝜉0-sp.I*𝜂, 's'),
           sp.cosh(𝜉0+sp.I*𝜂): hyper2e(𝜉0+sp.I*𝜂, 'c'),
           sp.sinh(𝜉0+sp.I*𝜂): hyper2e(𝜉0+sp.I*𝜂, 's'),
           sp.tanh(𝜉0+sp.I*𝜂): hyper2e(𝜉0+sp.I*𝜂, 't')}
    return dic_e

def hyper02e_dic(𝜉0):
    '''produces a dictionary for substituting hyperbolic functions with
       exponential functions
    '''
    dic={sp.cosh(𝜉0):hyper2e(𝜉0,'c'), sp.cosh(4*𝜉0):hyper2e(4*𝜉0,'c'),
         sp.sinh(𝜉0):hyper2e(𝜉0,'s'), sp.sinh(4*𝜉0): hyper2e(4*𝜉0,'s'),
         sp.cosh(2*𝜉0): hyper2e(2*𝜉0, 'c'),
         sp.sinh(2*𝜉0): hyper2e(2*𝜉0, 's'),
         sp.tanh(𝜉0): hyper2e(𝜉0, 't'), sp.coth(𝜉0): hyper2e(𝜉0, 'ct')}
    return dic

#def eI_sc(alpha):
#    return {sp.exp(sp.I*alpha): sp.cos(alpha)+sp.I*sp.sin(alpha)}

def deepsimp(expr):
    '''perform deep simplification to a complex expression
    ''' 
    # Extract real and imaginary parts
    re_part = sp.re(expr); im_part = sp.im(expr)

    # Simplify each part
    simp_re_part = sp.simplify(re_part, deep=True).expand().simplify()
    simp_im_part = sp.simplify(im_part, deep=True).expand().simplify()

    # Reconstruct the simplified complex expression
    return simp_re_part + I * simp_im_part

def print_limit(expr_name, expression, max_line_length):
    '''print a variable/expression. Each line has a max_line_length.
    input: expr_name, the name of the expression.
    Example: 
    expr = 𝜎*(exp(4*𝜉) - 2*exp(2*𝜉)*cos(2*𝜂) + 1)*(-8*exp(2*𝜉0))  
    print_limit("expr = ", expr, 70) 
    '''
    print(expr_name)                   # print this a the beginning 
    for i in range(0, len(str(expression)), max_line_length):
        print(str(expression)[i:i+max_line_length])
        
def printLimit(expr_name, expression, max_line_length):
    '''print a variable/expression. Each line as a max_line_length, 
       and breaks only at a word.
    Example: 
    expr = 𝜎*(exp(4*𝜉) - 2*exp(2*𝜉)*cos(2*𝜂) + 1)*(-8*exp(2*𝜉0))  
    printLimit("expr = ", expr, 70) 
    '''
    print(expr_name, end='')
    words = str(expression).split()
    current_line_length = 0
    for word in words:
        if current_line_length + len(word) <= max_line_length:
            print(word, end=" ")
            current_line_length += len(word)+1  # +1: space btwn words
        else:
            print()  # start a new line
            print(word, end=" ")
            current_line_length = len(word) + 1

def writeRead(folderName, fileName, expression): 
    '''Write a variable/expression into a txt file with the same 
       fillename and then read the same variable/expression back in.
    inputs: folderName, fileName: all string.
    return: variable/expression: Python/Sympy expression
    '''
    with open(folderName+"/"+fileName+".txt","w",encoding="utf-8") as file:
        file.write(str(expression))

    with open(folderName+"/"+fileName+".txt","r",encoding="utf-8") as file:
        read_expression = file.read()

    return eval(read_expression)  # convert back to variable/expression

def xy2𝜉𝜂(x, y, c):
    '''inverse mapping from z-plane to 𝜁-plane'''
    z = x + 1j*y                        # complex z-plane in Cartesian
    𝜁 = np.arccosh(z/c)               # to complex 𝜁-plane in elliptic
    return 𝜁.real, 𝜁.imag 

def readFrFile(folderName, fileName, oldStr, newStr): 
    '''Read a variable/expression in a txt file.
    inputs: folderName, fileName: all string.
            oldStr in the expression is to be replace by newStr
    return: variable/expression: Python/Sympy expression
    '''
    with open(folderName+"/"+fileName+".txt","r",encoding="utf-8") as file:
        expression_ = file.read()

    for i in range(len(oldStr)):
        expression_ = expression_.replace(oldStr[i], newStr[i])
    return expression_   # convert back to variable/expression

def writeToFile(folderName, fileName, expression): 
    '''Write a variable/expression into a txt file with the same 
       fillename. Inputs: folderName, fileName: all string.
    '''
    with open(folderName+"/"+fileName+".txt","w",encoding="utf-8") as file:
        file.write(str(expression))

def readFromFile(folderName, fileName): 
    '''Read a variable/expression in a txt file.
    inputs: folderName, fileName: all string.
    return: variable/expression: Python/Sympy expressionSS
    '''
    with open(folderName+"/"+fileName+".txt","r",encoding="utf-8") as file:
        read_expression = file.read()

    return read_expression    # convert back to variable/expression

def stressOnN00(S,N):
    '''Compute the traction (stress vector), normal stress, and \
    magnitude of the shear stress on a surface with normal vector N
    Input S: stress tensor in matrix form with 9 components\
          N: normal vector of the surface
    Return: S_N (traction), S_NN (normal stress), S_NS (shear stress)
    '''

    N = N/np.linalg.norm(N) # normalize vector N -> unit normal vector

    # Compute the traction vector on the surface with normal N
    S_N = np.dot(N,S)
    
    # Compute the normal stress on surface in the normal N direction
    S_NN = np.dot(S_N,N)
    
    # compute the magnitude of the shear stress on the surface
    S_NS = np.sqrt(S_N.dot(S_N)-S_NN**2)
    
    return S_N, S_NN, S_NS

def stressOnN0(σ,n):
    '''Compute the traction (stress vector), normal stress, and \
    magnitude of the shear stress on a surface with normal vector n
    Input σ: stress tensor in matrix form with 9 components\
          n: normal vector of the surface
    Return: σ_n (traction), σ_nn (normal stress), σ_ns (shear stress)
    '''

    n = n/np.linalg.norm(n) # normalize vector n -> unit normal vector

    # Compute the traction vector on the surface with normal n
    σ_n = np.dot(n,σ)
    
    # Compute the normal stress on surface in the normal n direction
    σ_nn = np.dot(σ_n,n)
    
    # compute the magnitude of the shear stress on the surface
    σ_ns = np.sqrt(σ_n.dot(σ_n)-σ_nn**2)
    
    return σ_n, σ_nn, σ_ns

def stressOnN(σ, n, q=False):
    '''Compute the traction (stress vector), normal stress, and \
    magnitude of the shear stress, and the shear stress on a surface 
    with normal vector n.
    Input σ: stress tensor in matrix form with 9 components\
          n: normal vector of the surface
          q: a unit vector on the surface; an vector r orthogonal to q
             will be generated based on the right-hand rule, and 
             the shear stresses in both q&r directions will be computed.
             If not provided, the shear stresses are not computed. 
    Return: σ_n (traction), σ_nn (normal stress), σ_ns (shear stress)
            σ_nq (shear stress in given q-direction)
            σ_nr(shear stress in r-direction)
    '''
    n = n/np.linalg.norm(n) # normalize vector n -> unit normal vector
    
    # Compute the traction vector on the surface with normal n
    σ_n = np.dot(n,σ)
    
    # Compute the normal stress on surface in the normal n direction
    σ_nn = np.dot(σ_n,n) 
    
    # compute the magnitude of the shear stress on the surface
    σ_ns = np.sqrt(σ_n.dot(σ_n)-σ_nn**2)  
    
    # Compute the shear stresses on the surface, if q is given
    if isinstance(q, (list, tuple, np.ndarray)):
        q = q/np.linalg.norm(q)                 # normalize vector q 
        ndq = np.dot(n,q)
    
        if ndq > 1e-12: 
            print(f'n and q are not orthogonal: {ndq}; σnq & σnr=None.')
            σ_nq = None; σ_nr = None  
        else: 
            r = np.cross(n, q)      # generate r via right-hand rule
            print(f'Unit vector r on surface-n = {r}\n')
    
            # Compute the shear stress on the surface direction q
            σ_nq = np.dot(σ_n,q)

            # Compute the shear stress on the surface direction q
            σ_nr = np.dot(σ_n,r)
    elif q == False:
        print(f'Shear stress will not be computed.')
        σ_nq = None; σ_nr = None    
    else:
        print(f'q must be array-like')
        σ_nq = None; σ_nr = None  
    
    return σ_n, σ_nn, σ_ns, σ_nq, σ_nr


def eigv_signAlter(St):
    '''Compute the eigenvalue and eigenvectors of stress tensor, St
       And the alter the signs of eigenvectors to see the effects. 
    Input: St, stress tensor, array like, 3 by 3.
    return: S_recovered stress tensor. 
    # Examples:     
    #St= np.array([[-10, 15, 0], [15, 30, 0], [0, 0, 0]])   # stresses
    #St =  np.array([[10, 15, 0], [15, -30, 0], [0, 0, 0]]) 
    #St =  np.array([[10, 15, 8], [15, -30, 5], [8, 5, 9]])  
    #S_recovered = eigv_signAlter(St)
    #S_recovered = eigv_signAlter(St)
    '''
    eigenValues, eigenVectors = principalS(St)
    signs = [(1., 1), (-1., 1), (1., -1),(-1., -1)]
    eigenVectors_=np.empty_like(eigenVectors)
    for i in range(len(signs)): 
        eigenVectors_[:,0] = signs[i][0]*eigenVectors[:,0]
        eigenVectors_[:,1] = signs[i][1]*eigenVectors[:,1]
        eigenVectors_[:,2] = eigenVectors[:,2]
        #eigenVectors_[:,2] = np.cross(eigenVectors_[:,0], eigenVectors_[:,1])
        S_recovered = eigenVectors_.T@St@eigenVectors_
        ei_diff = np.array([S_recovered[i,i]-eigenValues[i] 
                            for i in range(len(eigenValues))])
        diff_sum = np.sum(np.abs(ei_diff))
        print(f'i={i}, {signs[i][1]} diff_sum={diff_sum}')
        #if diff_sum < 1.e-129: 
        #    print(f'i={i}, diff_sum={diff_sum}')
        #    break

    print(f'Principal stress directions:\n{eigenVectors_}\n')
    return S_recovered 

def tensorT2angle(𝜎xx, 𝜎yy, 𝜎xy, 𝜃, py='sympy'):
    '''
    Perform coordinate transformation for 2nd order tensors (stress and
    strain) using double angle of 𝜃. 
    inputs: 𝜎xx, 𝜎yy, 𝜎xy in x-y,
            𝜃: angle (rad) of rotation (counterclockwise) to form X-Y
            py: if 'sympy', return sympy formulas
                if 'numpy', return numpy object
    return: 𝜎XX, 𝜎YY, 𝜎XY in X-Y
    ''' 
    if py == 'sympy':
        c2𝜃 = sp.cos(2*𝜃); s2𝜃 = sp.sin(2*𝜃)
    elif py == 'numpy':
        c2𝜃 = np.cos(2*𝜃); s2𝜃 = np.sin(2*𝜃)
    else: 
        print("Please set py == 'sympy' or 'numpy'")
        
    𝜎XX = σxx*c2𝜃/2 + σxx/2 + σxy*s2𝜃/2 + σxy*s2𝜃/2 - σyy*c2𝜃/2 + σyy/2
    𝜎YY =-σxx*c2𝜃/2 + σxx/2 - σxy*s2𝜃/2 - σxy*s2𝜃/2 + σyy*c2𝜃/2 + σyy/2
    𝜎XY =-σxx*s2𝜃/2 + σxy*c2𝜃/2 + σxy/2 + σxy*c2𝜃/2 - σxy/2 + σyy*s2𝜃/2
    
    return 𝜎XX, 𝜎YY, 𝜎XY

def Mohr_circle2D(𝜎, lw=1.0):
    '''
    Plot a 2D Mohr circle, for given 2×2 stress/strain matrix (tensor).
    input: 𝜎, numpy array.
           lw, the line width of the circle. 
    Example: 
    plt.figure(figsize=(6, 6)) 
    Mohr_circle2D(np.array([[88.,-35], [-35, 38.]]), lw=2.0)
    '''
    𝜎_avg = (𝜎[0,0]+𝜎[1,1])/2                            # mean stress
    center = (𝜎_avg, 0)                    # center of the Mohr circle
    R = np.sqrt((𝜎[0,0]-𝜎[1,1])**2/4+𝜎[0,1]**2)   # radius Mohr circle
    
    plt.scatter(𝜎_avg, 0, c='r', marker="o", zorder=2) # circle center  
    plt.scatter((𝜎[1,1],𝜎[0,0]),(𝜎[0,1],-𝜎[0,1])) #current state A & B
    
    plt.plot((𝜎[1,1],𝜎[0,0]),(𝜎[0,1],-𝜎[0,1]), color='blue', lw=0.5)
    plt.text(𝜎[1,1]+R/25., 𝜎[0,1]+R/25., "A")        # current state A 
    plt.text(𝜎[0,0]+R/25.,-𝜎[0,1]+R/25., "B")        # current state B
    
    circle = plt.Circle(center, R, color='blue', fill=False, lw=lw)
    plt.gca().add_patch(circle) #plt.gca().invert_yaxis()
    
    plt.axis('scaled')
    plt.xlim(𝜎_avg-R-R/10, 𝜎_avg+R+R/10)   
    plt.xlabel('$𝜎_{XX}$'); plt.ylabel('$𝜎_{XY}$')
    #plt.show()

def Mohr_circle2D(𝜎, lw=1.0):
    '''
    Plot a 2D Mohr circle, for given 2×2 stress/strain matrix (tensor).
    input: 𝜎: numpy array.
       mark: maker (string) of a stress state on the circle
       lw: the line width of the circle. 
    Example: 
    plt.figure(figsize=(6, 6)) 
    Mohr_circle2D(np.array([[88.,-35], [-35, 38.]]), lw=2.0)
    '''
    𝜎_avg = (𝜎[0,0]+𝜎[1,1])/2                            # mean stress
    center = (𝜎_avg, 0)                    # center of the Mohr circle
    R = np.sqrt((𝜎[0,0]-𝜎[1,1])**2/4+𝜎[0,1]**2)   # radius Mohr circle
    
    plt.scatter(𝜎_avg, 0, c='r', marker="o", zorder=2) # circle center  
    plt.scatter((𝜎[1,1],𝜎[0,0]),(𝜎[0,1],-𝜎[0,1])) #current state A & B
    
    plt.plot((𝜎[1,1],𝜎[0,0]),(𝜎[0,1],-𝜎[0,1]), color='blue', lw=0.5)
    plt.text(𝜎[1,1]+R/25., 𝜎[0,1]+R/25., "A")        # current state A 
    plt.text(𝜎[0,0]+R/25.,-𝜎[0,1]+R/25., "B")        # current state B
    
    circle = plt.Circle(center, R, color='blue', fill=False, lw=lw)
    plt.gca().add_patch(circle) #plt.gca().invert_yaxis()
    
    plt.axis('scaled')
    plt.xlim(𝜎_avg-R-R/10, 𝜎_avg+R+R/10)   
    plt.xlabel('$𝜎_{XX}$'); plt.ylabel('$𝜎_{XY}$')
    #plt.show()

def Mohr_circle3D(𝜎, lw=1.0):
    '''
    Plot a 3D Mohr circle, for given 3×3 stress/strain matrix (tensor) 
    𝜎 in numpy array. lw: the line width of the circle. 
    Example: 
    plt.figure(figsize=(6, 6))
    Mohr_circle3D([100., 20., -50.], lw=2.0)  
    '''
    p𝜎, _ = principalS(𝜎)                  # ranked principal stresses
    
    𝜎_avg12 = (p𝜎[0]+p𝜎[1])/2                     # 3 average stresses
    𝜎_avg13 = (p𝜎[0]+p𝜎[2])/2
    𝜎_avg23 = (p𝜎[1]+p𝜎[2])/2
    
    center1 = (𝜎_avg12, 0)                                 # 3 centers 
    center2 = (𝜎_avg13, 0)
    center3 = (𝜎_avg23, 0)
    
    R1 = (p𝜎[0]-p𝜎[1])/2                                     # 3 radii
    R2 = (p𝜎[0]-p𝜎[2])/2
    R3 = (p𝜎[1]-p𝜎[2])/2
    
    # centers of 3 circles:
    circle2 = plt.Circle(center2, R2, edgecolor='blue', \
                         facecolor='lightgreen', lw=lw)
    circle1 = plt.Circle(center1, R1, edgecolor='c', \
                         facecolor='white', lw=lw)
    circle3 = plt.Circle(center3, R3, edgecolor='orange', \
                         facecolor='white', lw=lw)
    
    plt.gca().add_patch(circle2)               # plot on the same plot
    plt.gca().add_patch(circle1)
    plt.gca().add_patch(circle3)

    plt.axis('scaled')
    plt.xlabel('$𝜎_{nn}$'); plt.ylabel('$𝜎_{ns}$')
    
    # three principal stresses:
    plt.scatter(p𝜎, (0,0,0), c='r', marker="o", s=6, zorder=2)
    
    # three centers: 
    centers = (𝜎_avg12,𝜎_avg13,𝜎_avg23)
    plt.scatter(centers, (0,0,0), c='k', \
                marker="o", facecolor='none', s=4, zorder=2)  

    plt.scatter(centers,( R1, R2, R3), c='r', marker="o",\
                 s=6, zorder=2)      # maximum shear stress (positive)
    plt.scatter(centers,(-R1,-R2,-R3), c='r', marker="o",\
                 s=6, zorder=2)      # maximum shear stress (negative)
    plt.show()

def J_rectangle(a, b, n):
    '''Compute the approximated J value for a rectangular cross-section
       In the process of testing......'''
    ta = a/2/n    # thinckness in horizotal direction
    tb = b/2/n    # thinckness in vetical direction
    ai = np.zeros(n); bi = np.zeros(n)
    ai[0] = a - ta; bi[0] = b - tb
    for i in range(1, n):
        ai[i] = ai[i-1] - 2*ta
        bi[i] = bi[i-1] - 2*tb
    print(ai)
    J = 4*ai**2*bi**2/(2*ai/ta+2*bi/tb)
    J = np.sum(J)
    return J

## Exmaple: 
#a = 2.; b = 2.; n = 25
#J_rctngl = J_rectangle(a, b, n)
#J_k1 = 0.140577*a**4        # the series solution
#print(f"J_rectangle = {J_rctngl}; J_k1 = {J_k1}")

def E_SnC2D_stress(E1, E2, m12, G12):
    '''Compute the S and C matrix in Voigt notation for given Young's  
    moduli and Poisson's ratios of orthotropic materials for 2D plan 
    stress problems.
    '''
    S = np.zeros((3,3))             #initialization
    m21 = m12/E1*E2 
    
    # compute the compliance matrix S 
    S[0,0], S[1,1] = 1./E1, 1./E2 
    S[0,1]         = -m21/E2 
    S[1,0]         = S[0,1] 
    S[2,2]         = 1./G12 

    # compute C matrix
    C = np.linalg.inv(S)
    
    return C, S

def grad_f(f, X):
    '''Compute the gradient of a given scalar function f.
       input: f, sp.Function. 
              X, coordinates, array-like of symbols. 
       return: grad_f, sp.Matrix, a matrix of 2×1 or 3×1 or len(X)×1.
       ## Example: 
        x, y, z = sp.symbols("x, y, z") 
        X = [x, y, z]
        f = 8*x**2 + 5*x*y -2*y + 3*y**2 + 2*z**2 - 5*z
        f_g = grad_f(f, X)
        f_g
    '''
    gradf = sp.Matrix([sp.diff(f,xi) for xi in X]) 
    return gradf  

def grad_vf(vf, X):
    '''Compute the gradient of a given vector function, vf
       input: vf, sp.Matrix, a column vector of 2 or 3. 
              X, coordinates, array-like of symbols
       return: grad_f, sp.Matrix, a matrix of 2×1 or 3×1 or len(X)×1.
       ## Example: 
       x, y, z = sp.symbols("x, y, z") 
       vf = sp.Matrix([y*sp.sin(x), z*sp.sin(y), x*sp.cos(z)]) 
       vf_g = grad_vf(vf, X)
    '''
    grad_f = sp.Matrix([sp.diff(vf,xi).T for xi in X]) 
    return grad_f  

def grad_vf0(vf):
    '''Compute the gradient of a given vector function, vf
       input: vf, sp.Matrix, a column vector of 2 or 3. 
       return: grad_f, sp.Matrix, a matrix of 2 by 2 or 3 by 3.
       ## Example: 
       x, y, z = sp.symbols("x, y, z") 
       vf = sp.Matrix([y*sp.sin(x), z*sp.sin(y), x*sp.cos(z)]) 
       vf_g_ = grad_vf(vf)
    '''
    x, y, z = sp.symbols("x, y, z")

    if   len(vf) == 2: 
        grad_f = sp.Matrix([[sp.diff(vf,x).T], [sp.diff(vf,y).T]])
    elif len(vf) == 3: 
        grad_f = sp.Matrix([[sp.diff(vf,x).T], [sp.diff(vf,y).T],\
                         [sp.diff(vf,z).T]])
    else: 
        print(f" The length of vf must be 2 or 3")
        grad_f = " Not computed !"
        
    return grad_f  

def div_vf(vf, X):
    '''Compute the divergence of a vector function vf.
       input: vf, vector function in, sp.Matrix
              X, vector of in dependent variables, sp.Matrix
       return: div_f
    '''
    div_f = sp.Add(*[sp.diff(vf[i], X[i]) for i in range(len(X))]) 
    return div_f           # divergence of a vector function


def principalS_ypr(St):    
    '''Diagonalization of a stress tensor in 3D, through coordinate 
    transformation via three Yaw, Pitch and Roll angles. 
    Input: St: Stress tensor, array like, 3 by 3
    return: 𝜃, 𝛽, 𝛾: Yaw, Pitch and Roll angles, in rad. 
            TS_ypr: Diagonalized stress matrix array like, 3 by 3
    '''
    from scipy.optimize import fsolve
    
    𝜃, 𝛽, 𝛾 = sp.symbols("𝜃, 𝛽, 𝛾")      # Angles for Yaw, Pitch and Roll

    # Create the matrices for Yaw, Pitch, and Roll for transformations: 
    Tz = transf_YPRs(𝜃, about = 'z')[0]
    Ty = transf_YPRs(𝛽, about = 'y')[0]
    Tx = transf_YPRs(𝛾, about = 'x')[0]

    # Construct the transformation matrix for Yaw, Pitch and Roll. 
    T_ypr = Tz@Ty@Tx
 
    # Perform coordinate transformation to the given stress tensor. 
    S_ypr = T_ypr@St@T_ypr.T

    # Create these three equations: 
    eqns = lambda x: [S_ypr[0,1].subs({𝜃:x[0], 𝛽:x[1], 𝛾:x[2]}),
                      S_ypr[0,2].subs({𝜃:x[0], 𝛽:x[1], 𝛾:x[2]}), 
                      S_ypr[1,2].subs({𝜃:x[0], 𝛽:x[1], 𝛾:x[2]})]

    # Give an initial guess randomly: 
    init_guess = np.random.randn(3)/999.               # make it small 
 
    # Solve the set of three equations numerically: 
    sln = fsolve(eqns, init_guess)
    print(f'Solution, 𝜃, 𝛽, 𝛾 = {sln} (rad)')
    print(f'Solution, 𝜃, 𝛽, 𝛾 = {sln*180/np.pi} (degree)') 

    # Finally, we can check the results, use 𝜃, 𝛽, 𝛾  found to 
    # perform coordinate trans formation: 
    T_ypr_ = T_ypr.subs({𝜃:sln[0], 𝛽:sln[1], 𝛾:sln[2]})
    S_ypr_ = T_ypr_@St@T_ypr_.T

    return 𝜃, 𝛽, 𝛾, S_ypr_ 


def div_tensorF(σ, X):
    '''Compute the divergence of a tensor function σ.
       input: σ, tensor function in sp.Matrix
       return: the divergence of a tensor function
    '''
    n = σ.shape[0]
    div_tf = [(sp.Add(*[sp.diff(σ[i,j], X[i]) for i in range(n)]))\
               for j in range(n)]
    
    return sp.Matrix(div_tf)         # divergence of a tensor function


def curl_vf(vf, X):
    '''Compute the curl of a given vector function vf in Sympy w.r.t X.
       For 2D&3D.
    '''
    if len(vf) == 2:
        return sp.diff(vf[1], X[0]) - sp.diff(vf[0], X[1])

    curl_f = sp.Matrix([sp.diff(vf[2], X[1]) - sp.diff(vf[1], X[2]),
                        sp.diff(vf[0], X[2]) - sp.diff(vf[2], X[0]),
                        sp.diff(vf[1], X[0]) - sp.diff(vf[0], X[1])]) 
    return curl_f

def J_cylinder(R, n):
    
    '''Compute the approximated J value for a cylindrical cross-section
    input: R, radius; n, number of divisions
    ## Example: 
    R = 2.; n = 25
    J_cylndr = J_cylinder(R, n)
    J_exact = np.pi*R**4/2            # the exact solution
    print(f"J_cylinder = {J_cylndr}; J_exact = {J_exact}")
    '''
    t_half = R/2./n          # half thickness of a thin-wall cylinder
    t = t_half*2.
    print("The thickness of the thin-walls =",t)
    
    Ri = np.zeros(n) 
    Ri[0] = R - t_half 
    for i in range(1, n):
        Ri[i] = Ri[i-1] - t 

    #print(Ri)
    J = 2*np.pi*t*Ri**3
    J = np.sum(J)
    return J

def combine_pdfs(files, output_file):
    '''Combine all pdf files in list of files.
    Example: 
    files = ['Temp/ICCM2023-updated20-06-2023.pdf', 'Temp/page12th.pdf']
    output_file = 'Temp/combined.pdf'
    combine_pdfs(files, output_file)
    '''
    merger = PyPDF2.PdfMerger()  # may need to import PyPDF2
    
    for file in files: 
        # Open the PDF file
        with open(file, 'rb') as fi:
            merger.append(fi) 

    # Write the merged PDF to the output file
    with open(output_file, 'wb') as output:
        merger.write(output)

def deletePDFpages(input_file, output_file, pages_to_delete):
    '''
    Example: 
    input_file = 'Temp/combined.pdf'
    output_file = 'Temp/combined2.pdf'
    pages_to_delete = [2, 5]  # deleting page 2 and page 5
    deletePDFpages(input_file, output_file, pages_to_delete)
    '''
    reader = PyPDF2.PdfReader(input_file) # may need to import PyPDF2
    writer = PyPDF2.PdfWriter()

    total_pages = len(reader.pages) #reader.numPages

    # Add pages to the writer object, excluding the ones to delete
    for page_number in range(total_pages):
        if page_number + 1 not in pages_to_delete:
            page = reader.pages[page_number] #reader.getPage(page_number)
            writer.add_page(page)

    # Write the modified PDF to the output file
    with open(output_file, 'wb') as output:
        writer.write(output)

def Tensor1_transfer(T,u):
    '''Transformation for 1st order tensor u using transformation 
        matrix T. 
        u and T are array alike. 
    '''
    S = np.tensordot(T, u, axes=([1],[0]))    # use numpy tensordot()
               # contracting axis-1 of T with axis-0 of u
    return S 

def Tensor4_transfer(T,C4):
    '''Transformation of 4th order tensors
    input: T, transformation matrix (aij); C4: tensor to be transformed
    return:C4, transformed tensor of 4th order'''
    
    C4 = np.tensordot( T, C4, axes=([1],[0]))  # contract i
    C4 = np.tensordot( T, C4, axes=([1],[1]))  # contract j
    C4 = np.tensordot(C4,  T, axes=([3],[1]))  # contract l
    C4 = np.tensordot(C4,  T, axes=([2],[1]))  # contract k

    return C4

def C2toC4(C2):
    '''To convert C(6,6) matrix (the Voigt notation) to  
       4th tensor C(3,3,3,3).
    '''
    C4 = np.zeros((3,3,3,3))                     #Initialization
             
    # Pass over all C(6,6) to parts of C(3,3,3,3)
    C4[0,0,0,0],C4[0,0,1,1],C4[0,0,2,2] = C2[0,0],C2[0,1],C2[0,2]
    C4[0,0,1,2],C4[0,0,0,2],C4[0,0,0,1] = C2[0,3],C2[0,4],C2[0,5]

    C4[1,1,0,0],C4[1,1,1,1],C4[1,1,2,2] = C2[1,0],C2[1,1],C2[1,2]
    C4[1,1,1,2],C4[1,1,0,2],C4[1,1,0,1] = C2[1,3],C2[1,4],C2[1,5]

    C4[2,2,0,0],C4[2,2,1,1],C4[2,2,2,2] = C2[2,0],C2[2,1],C2[2,2]
    C4[2,2,1,2],C4[2,2,0,2],C4[2,2,0,1] = C2[2,3],C2[2,4],C2[2,5]

    C4[1,2,0,0],C4[1,2,1,1],C4[1,2,2,2] = C2[3,0],C2[3,1],C2[3,2]
    C4[1,2,1,2],C4[1,2,0,2],C4[1,2,0,1] = C2[3,3],C2[3,4],C2[3,5]

    C4[0,2,0,0],C4[0,2,1,1],C4[0,2,2,2] = C2[4,0],C2[4,1],C2[4,2]
    C4[0,2,1,2],C4[0,2,0,2],C4[0,2,0,1] = C2[4,3],C2[4,4],C2[4,5]

    C4[0,1,0,0],C4[0,1,1,1],C4[0,1,2,2] = C2[5,0],C2[5,1],C2[5,2]
    C4[0,1,1,2],C4[0,1,0,2],C4[0,1,0,1] = C2[5,3],C2[5,4],C2[5,5]
                             
    # Impose (minor) symmetric conditions
    apply_symmetry(C4, key = "all", tol=1.e-4)
    
    return C4

def apply_symmetry(C4, key = "all", tol=1.e-2):
    '''Impose (minor) symmetric conditions
    '''
    if key == "all" or key == "ij":
        for k in range(3):
            for l in range(3):
                for i in range(3):
                    for j in range(i+1,3):
                        if abs(C4[j,i,k,l]) <= tol:
                            C4[j,i,k,l]=C4[i,j,k,l]
                            
    if key == "all" or key == "kl":
        for k in range(3):
            for l in range(k+1,3):
                for i in range(3):
                    for j in range(3):
                        if abs(C4[i,j,l,k]) <= tol:
                            C4[i,j,l,k]=C4[i,j,k,l]
                            
    if key == "all" or key == "ijkl":
        for k in range(3):
            for l in range(3):
                for i in range(k+1,3):
                    for j in range(l+1,3):
                        if abs(C4[i,j,k,l]) <= tol:
                            C4[i,j,k,l]=C4[k,l,i,j]
    return C4


def C4toC2(C4):
    '''To convert 4th tensor C(3,3,3,3) to C(6,6) matrix 
       (the Voigt notation).
    '''
    C2 = np.zeros((6,6))
    C2[0,0],C2[0,1],C2[0,2]=C4[0,0,0,0],C4[0,0,1,1],C4[0,0,2,2]
    C2[0,3],C2[0,4],C2[0,5]=C4[0,0,1,2],C4[0,0,0,2],C4[0,0,0,1]

    C2[1,0],C2[1,1],C2[1,2]=C4[1,1,0,0],C4[1,1,1,1],C4[1,1,2,2]
    C2[1,3],C2[1,4],C2[1,5]=C4[1,1,1,2],C4[1,1,0,2],C4[1,1,0,1]

    C2[2,0],C2[2,1],C2[2,2]=C4[2,2,0,0],C4[2,2,1,1],C4[2,2,2,2]
    C2[2,3],C2[2,4],C2[2,5]=C4[2,2,1,2],C4[2,2,0,2],C4[2,2,0,1]

    C2[3,0],C2[3,1],C2[3,2]=C4[1,2,0,0],C4[1,2,1,1],C4[1,2,2,2]
    C2[3,3],C2[3,4],C2[3,5]=C4[1,2,1,2],C4[1,2,0,2],C4[1,2,0,1]

    C2[4,0],C2[4,1],C2[4,2]=C4[0,2,0,0],C4[0,2,1,1],C4[0,2,2,2]
    C2[4,3],C2[4,4],C2[4,5]=C4[0,2,1,2],C4[0,2,0,2],C4[0,2,0,1]

    C2[5,0],C2[5,1],C2[5,2]=C4[0,1,0,0],C4[0,1,1,1],C4[0,1,2,2]
    C2[5,3],C2[5,4],C2[5,5]=C4[0,1,1,2],C4[0,1,0,2],C4[0,1,0,1]
    
    return C2

def E_SnC3D(E1, E2, E3, m12, m13, m23, G23, G13, G12):
    '''Numpy code to compute the S and C matrix in Voigt notation for   
    given Young's moduli and Poisson's ratios of orthotropic materials 
    for 3D problems.
    '''
    S = np.zeros((6,6))             #initialization
    
    m21, m31, m32 = m12/E1*E2, m13/E1*E3, m23/E2*E3
    
    # compute the compliance matrix S 
    S[0,0], S[1,1], S[2,2] = 1./E1, 1./E2, 1./E3 
    S[0,1], S[0,2], S[1,2] = -m21/E2, -m31/E3, -m32/E3
    S[3,3], S[4,4], S[5,5] = 1./G23, 1./G13, 1./G12 
    S[1,0], S[2,0], S[2,1] = S[0,1], S[0,2], S[1,2]
    
    # compute C matrix
    C = np.linalg.inv(S)
    
    return C, S

def E_SnC3Dsp(E1, E2, E3, m12, m13, m23, G23, G13, G12):
    '''Sympy code to compute the S and C matrix in Voigt notation for   
    given Young's moduli and Poisson's ratios of orthotropic materials 
    for 3D problems.
    '''
    S = sp.zeros(6,6)               #initialization
    
    m21, m31, m32 = m12/E1*E2, m13/E1*E3, m23/E2*E3
    
    # compute the compliance matrix S 
    S[0,0], S[1,1], S[2,2] = 1/E1, 1/E2, 1/E3 
    S[0,1], S[0,2], S[1,2] = -m21/E2, -m31/E3, -m32/E3
    S[3,3], S[4,4], S[5,5] = 1/G23, 1/G13, 1/G12 
    S[1,0], S[2,0], S[2,1] = S[0,1], S[0,2], S[1,2]
    
    # compute C matrix
    C = S.inv()
    
    return C, S

def E_SnC2D_stress(E1, E2, 𝜈12, G12):
    '''Numpy code to ompute S and C matrix in Voigt notation for given 
       Young's moduli Ei, and Poisson's ratios 𝜈12, shear modulus G12 
       of orthotropic materials for 2D plan stress problems.
    '''
    S = np.zeros((3,3))   # sp: S = sp.zeros(3,3)      #initialization
    𝜈21 = 𝜈12/E1*E2 
    
    # compute the compliance matrix S 
    S[0,0], S[1,1] = 1/E1, 1/E2 
    S[0,1]         = -𝜈21/E2 
    S[1,0]         = S[0,1] 
    S[2,2]         = 1/G12 

    # compute C matrix
    C = np.linalg.inv(S)  # sp: S = sp.inv(S)   
    
    return C, S

def E_SnC2D_strain(E1, E2, E3, 𝜈12, 𝜈13, 𝜈23, G12):
    '''Compute the S and C matrix in Voigt notation for given Young's  
    moduli Ei, Poisson's ratios 𝜈ij, and shear modulus G12 of
    orthotropic materials for 2D plan strain problems.
    '''
    S = np.zeros((3,3))                #initialization
    𝜈21 = 𝜈12/E1*E2 
    𝜈31 = 𝜈13/E1*E3
    𝜈32 = 𝜈23/E2*E3
    
    # compute the compliance matrix S 
    S[0,0] = (1. -𝜈13*𝜈31)/E1 
    S[0,1] =-(𝜈12+𝜈13*𝜈32)/E1 
    S[1,1] = (1. -𝜈23*𝜈32)/E2 
    S[1,0]         = S[0,1] 
    S[2,2]         = 1./G12 

    # compute C matrix
    C = np.linalg.inv(S)
    
    return C, S

def box_shearCenter(h, b, t_L, t_R, t_T): 
    '''Compute the shear center of a box-section with 4 thin edges wrt 
    the middle line of the right-edge. Its dimensions are measured 
    based on the middle lines of the edges. 
    h:   beam height; b: width of top- and bottom-edge; 
    t_T: thickness of top-edge; t_B thickness of bottom-edge: t_B=t_t
    t_L, t_R: thickness of left-edge and right-edges. 
    Return: 
    e: shear center; qP, qQ, qS, qA: q values at key points: 
    P(north-west), Q(west), S(east), A(north-east). 
    '''
    t_B = t_T
    # compute Izz 
    Izz = (1./12)*(t_L+t_R)*h**3 + b*t_T*(h/2)**2 + b*t_B*(h/2)**2
    print(f"2nd moment of inertia of area Izz = {Izz} [L^4]")

    # Compute shear flow at P and R (north-west, south-west corners)
    qP = b*t_T*(h/2)
    qR = b*t_B*(h/2)
    # qQ = qP + (h/2)*t_L*(h/4); qS = (h/2)*t_R*(h/4)
    #print(f"Shear flows at key points [Pa/m]: qP={qP:.4e}; qR={qR:.4e}")
 
    # Define the functions for 𝜏 = q/t values on all four edges: 
    qt_s = lambda s:-qP/b/t_T * s            # q function for top edge
    qb_s = lambda s:-qR/b/t_B * s         # q function for bottom edge
    ql_s = lambda s:-(qP + t_L*(h/2 - s)*(h/2 + s)/2)/t_L  # left edge
    qr_s = lambda s: t_R*(h/2 - s)*(h/2 + s)/2/t_R        # right edge

    # Compute total area of the shear stress 𝜏-map on all four edges: 
    A_t, _ = si.quad(qt_s, 0, b)              # use Gauss integration
    A_b, _ = si.quad(qb_s, 0, b)                        
    A_l, _ = si.quad(ql_s, -h/2, h/2)    
    A_r, _ = si.quad(qr_s, -h/2, h/2)    

    A_total = A_t + A_b + A_l + A_r
    #print(f"The total area of the 𝜏-map: {A_total:.4e}")

    # Compute total area of q/t for the unit qA=1 on all four edges: 
    A_qA = b/t_T + b/t_B + h/t_L + h/t_R
    #print(f"The total area of q/t for the unit qA=1: {A_qA:.4e}")

    qA = -A_total/A_qA    
    #print(f"q/t value at point A (north-east corner) qA = {qA:.4e}")

    # Compute the channel forces: 
    # Define the functions for (q+qA)/t values on all four edges: 

    qt_sA = lambda s:-qP/b*s + qA            # q function for top edge
    qb_sA = lambda s:-qR/b*s + qA            # q function for top edge
    ql_sA = lambda s:-(qP + t_L*(h/2-s)*(h/2+s)/2) + qA    # left edge
    qr_sA = lambda s: t_R*(h/2 - s)*(h/2 + s)/2 + qA      # right edge

    # Compute total area of the shear stress 𝜏-map on all four edges: 
    F_l = si.quad(ql_sA, -h/2, h/2)[0]    
    F_r = si.quad(qr_sA, -h/2, h/2)[0]   
    #print(f"Shear forces: F_l={F_l:.4e}; F_r={F_r:.4e}")
    V = -F_l+F_r 
    #print(f"The total shear forces in box-beam (V=Izz): V={V:.4e}")

    # Compute shear center wrt the middle line of the right edge. 
    F_t, _ = si.quad(qt_sA, 0, b)                     # use Gauss int.
    F_b, _ = si.quad(qb_sA, 0, b)                        
    #print(f"Shear forces: F_t={F_t:.4e}; F_b={F_b:.4e}")
    e = -(F_t*h + F_l*b)/V
    print(f"e wrt the middle line of the right edge: {e:.4e}")
    return e, qt_sA(b), ql_sA(0), qr_sA(0), qr_sA(h/2)


def gauss(f, n, a, b):
    '''Integrate f(x) over interval [a, b] using n Gauss points
    '''
    [x,w] = p_roots(n+1) # roots of the Legendre polynomial and weights
    G=0.5*(b-a)*sum(w*f(0.5*(b-a)*x+0.5*(b+a))) # in natural coordinate 
                # sample the function values at these roots and sum up. 
    return G

def cross2vectors(a, b, sympy=True):
    '''Cross-product of a×b, returns c
       if sympy=True: a, b, and c are all sympy Matrixes.
       otherwise: all in numpy arrays
    '''
    c1 = (a[1]*b[2] - a[2]*b[1])
    c2 = (a[2]*b[0] - a[0]*b[2])
    c3 = (a[0]*b[1] - a[1]*b[0])
    if sympy: 
        c = sp.Matrix([c1, c2, c3])
    else:
        c = np.array([c1, c2, c3])
    return c

def cross2v(a, b, sympy=True):
    '''Cross-product of a×b, returns c
       if sympy=True: a, b, and c are all sympy Matrixes.
       otherwise: all in numpy arrays
    '''
    if len(a) == 2:
        c = a[0]*b[1] - a[1]*b[0]
    else:     
        if sympy: 
            c = a.cross(b)
        else:
            c = np.cross(a, b) 
    return c

def N_Lagrange_fxy(nodexi,nodeyj): 
    '''Compute the 4 shape functions using lagrange interpolators.
       Use physical coordinate for square elements.  2D array form. 
    '''
    x, y =symbols('x, y')                 # define 2D physic coordinate

    lxs = [lx for lx in gr.LagrangeP(x, nodexi)]
    lys = [ly for ly in gr.LagrangeP(y, nodeyj)]
    Nijs = sp.Matrix(lxs)@ sp.Matrix(lys).T  

    return lxs, lys, Nijs

def N_Lagrange_f(nodexi,nodeyj): 
    '''Compute and plot shape functions generated using lagrange 
    interpolators. Use natural coordinates. 
    '''
    𝜉, 𝜂 =sp.symbols('𝜉, 𝜂')                 # define2D natrual coordinate

    lxs = [lx for lx in LagrangeP(𝜉, nodexi)]
    lys = [ly for ly in LagrangeP(η, nodeyj)]
    Nijs = sp.Matrix(lxs)@ sp.Matrix(lys).T  

    return lxs, lys, Nijs

def N_Lagrange_fN(nodexi,nodeyj): 
    '''Compute 4 shape functions generated using lagrange 
    interpolators. Use natural coordinates.  
    Nodes arrangement N-shape: 2   4
                               1   3
    '''
    𝜉, 𝜂 =sp.symbols('𝜉, 𝜂')                 # define2D natrual coordinate

    lxs = [lx for lx in LagrangeP(𝜉, nodexi)]
    lys = [ly for ly in LagrangeP(η, nodeyj)]
    Nij = sp.Matrix(lxs)@ sp.Matrix(lys).T  

    return lxs, lys, Nij 

def N_Lagrange_fC(nodexi,nodeyj): 
    '''Compute 4 shape functions generated using lagrange 
    interpolators. Use natural coordinates. 
    Nodes arrangement, counterclockwise: 4   3
                                         1   2
    '''  
    𝜉, 𝜂 =sp.symbols('𝜉, 𝜂')                 # define2D natrual coordinate

    lxs = [lx for lx in LagrangeP(𝜉, nodexi)]
    lys = [ly for ly in LagrangeP(η, nodeyj)]
    
    Nij = [lxs[0]*lys[0], lxs[1]*lys[0],
           lxs[1]*lys[1], lxs[0]*lys[1]] 
    Nij = [Ni.simplify() for Ni in Nij]

    return lxs, lys, Nij

def NQ4C(𝜉, 𝜂):
    '''
       Shape function for Q4 elements, all sympy objects. 
       return a row vector for four nodal shape functions. 
       Nodes arrangement, Counterclockwise: 4   3
                                            1   2     '''
    𝜉, 𝜂 =sp.symbols('𝜉, 𝜂')
    N4=sp.Matrix([[(1 - 𝜂)*(1 - 𝜉)], [(1 - 𝜂)*(𝜉 + 1)],
                  [(𝜂 + 1)*(𝜉 + 1)], [(1 - 𝜉)*(𝜂 + 1)]])/4
    return N4.T

def N_Lagrange3D_f(nodexi, nodeyj, nodeyk): 
    '''Compute and plot shape functions generated using lagrange 
    interpolators. Use natural coordinates in 3D. 
    '''
    𝜉, 𝜂, 𝜁 = sp.symbols('𝜉, 𝜂, 𝜁')     # define 3D natrual coordinates

    lxs = [lx for lx in LagrangeP(𝜉, nodexi)]
    lys = [ly for ly in LagrangeP(η, nodeyj)]
    lzs = [lz for lz in LagrangeP(𝜁, nodeyk)]
    Nijk= [lxs[0]*lys[0]*lzs[0], lxs[1]*lys[0]*lzs[0],
           lxs[1]*lys[1]*lzs[0], lxs[0]*lys[1]*lzs[0],
           lxs[0]*lys[0]*lzs[1], lxs[1]*lys[0]*lzs[1],
           lxs[1]*lys[1]*lzs[1], lxs[0]*lys[1]*lzs[1]] 
    Nijk= [Ni.simplify() for Ni in Nijk]

    return lxs, lys, lzs, Nijk

def area_T3(x1, y1, x2, y2, x3, y3):
    '''Compute the area of a triangle with three nodal coordinates
    '''
    return abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))/2.


def area_Q4N(x1, y1, x2, y2, x3, y3, x4, y4):
    '''Compute the area of a Q4 quadrilateral with 4 nodal coordinates
    Nodes arrangement N-shape: 2   4
                               1   3
    '''
    return abs((x1-x4)*(y3-y2) + (x3-x2)*(y4-y1))/2.

def area_Q4C(x1, y1, x2, y2, x3, y3, x4, y4):
    '''Compute the area of a Q4, quadrilateral with 4 nodal coordinates
    Nodes arrangement, counterclockwise: 4   3
                                         1   2
    '''
    return abs((x1-x3)*(y2-y4) + (x2-x4)*(y3-y1))/2.

def polygonA_xy(x, y, absOn=True):
    '''
    Compute the area of a polygon element for given 
    its verticescoordinates in array x and y. Example formula for Q4: 
    area = (1/2) |-(x_3-x_1)(y_2-y_4) + (x_2-x_4)(y_3-y_1)|
         = (1/2) |(x1*y2+x2*y3+x3*y4+x4*y1)−(y1*x2+y2*x3+y3*x4+y4*x1)|
                  cyclic permutation using np.roll()
    if absOn == True, take the absolute value, otherwise the raw value.
    The raw value provides additional information on node numbering. 
    '''
    if absOn: 
        area=np.abs(x@np.roll(y,1)-y@np.roll(x,1))/2.
    else:
        area=(x@np.roll(y,-1)-y@np.roll(x,-1))/2. # roll backwards by 1
        
    return area


def comp_area_T3(T3_x_y):
    '''Compute the area of a triangular (T3) element for given 
    its vertices given in (3,2), 3 nodes with x, y coordinates.
    area = (1/2) |x1(y2 − y3) + x2(y3 − y1) + x3(y1 − y2)|
         = (1/2) |(x1*y2+x2*y3+x3*y1)−(y1*x2+y2*x3+y3*x1)|
                  cyclic permutation using np.roll()
    '''
    x = T3_x_y[:,0];  y = T3_x_y[:,1]
    area = .5*np.abs(x@np.roll(y, 1)-y@np.roll(x, 1))
    
    return area


def mc_int(f_values, area):
    """Compute the integral of a scalar function using the Monte Carlo 
    method.
    Imputs: f: Values of the scalar function to be integrated.
            Aare: area of the function is sampled.
    Returns: The estimate of the definite integral.
    """
    f_average = np.mean(f_values)        # averaged the function values  
    return f_average*area             # multiply the area of the domain
                  # zeroth-order approximation of the definite integral

def jacobQ4C(x1, y1, x2, y2, x3, y3, x4, y4):
    '''Compute the Jacobian matrix of a Q4, quadrilateral with 4 nodal 
    coordinates. Nodes arrangement, counterclockwise: 4   3
                                                      1   2
    '''
    𝜉, 𝜂 =sp.symbols('𝜉, 𝜂')
    J = sp.Matrix([
            [-x1*(1 - 𝜂)/4 + x2*(1 - 𝜂)/4 + x3*(𝜂 + 1)/4 - x4*(𝜂+1)/4, 
             -x1*(1 - 𝜉)/4 - x2*(𝜉 + 1)/4 + x3*(𝜉 + 1)/4 + x4*(1-𝜉)/4], 
            [-y1*(1 - 𝜂)/4 + y2*(1 - 𝜂)/4 + y3*(𝜂 + 1)/4 - y4*(𝜂+1)/4, 
             -y1*(1 - 𝜉)/4 - y2*(𝜉 + 1)/4 + y3*(𝜉 + 1)/4 + y4*(1-𝜉)/4]])
    return J 

def NQ4N(𝜉, 𝜂):
    '''Shape function for Q4 elements, all sympy objects. 
       return a row vector. 
       Nodes arrangement N-shape: 2   4
                                  1   3
    '''
    𝜉, 𝜂 =sp.symbols('𝜉, 𝜂')
    N4=sp.Matrix([[(1 - 𝜂)*(1 - 𝜉)], [(1 - 𝜉)*(𝜂 + 1)],
                  [(1 - 𝜂)*(𝜉 + 1)], [(𝜂 + 1)*(𝜉 + 1)]])/4
    return N4.T

def NQ4C(𝜉, 𝜂):
    '''Shape function for Q4 elements, all sympy objects. 
       return a row vector. 
       Nodes arrangement, counterclockwise: 4   3
                                            1   2
    '''
    𝜉, 𝜂 =sp.symbols('𝜉, 𝜂')
    N4=sp.Matrix([[(1 - 𝜂)*(1 - 𝜉)], [(1 - 𝜂)*(𝜉 + 1)],
                  [(𝜂 + 1)*(𝜉 + 1)], [(1 - 𝜉)*(𝜂 + 1)]])/4
    return N4.T

def T4_V(v1, v2, v3, v4, absOn=True):
    """
    Computes the volume of a T4_hedron with 4 nodes (T4):v1,v2,v3,v4, 
    all in numpy array (3,) each containing 3 coordinates (x, y, z).
    Nodes arrangment: counterclockwise on the base triangle, and the
    thumb points to the tip node. 
    returns: the volume of T4, float.
    """
    # Form the moment matrix using 4 nodes and 1s: 
    vs = np.hstack((np.vstack([v1,v2,v3,v4]), np.ones(4).reshape(4,1))) 
    V = np.linalg.det(vs)/6.0
    if absOn: V = abs(V)  
        
    return V                                             # Volume of T4

def T4_V3(v1, v2, v3, v4, absOn=True):
    """
    Computes the volume of a tetrahedron with 4 nodes (T4):v1,v2,v3,v4, 
    all in numpy array (3,) each containing 3 coordinates (x, y, z).
    Nodes arrangment: counterclockwise on the base triangle, and the
    thumb points to the tip node. 
    returns: the volume of T4, float.
    """
    # Form the moment matrix using 4 nodes: 
    
    vs = np.vstack([v2-v1, v3-v1, v4-v1])
    det = np.linalg.det(vs)
    V = -np.linalg.det(vs)/6.0  # needs "-", because: i=1, j=4, i+j=odd 
    if absOn: V = abs(V)  
        
    return V                                             # Volume of T4

def H8_V(nodes, absOn=True):
    """
    Computes the volume of a hexahedron with 8 nodes (H8) by dividing
             H8 into six T4 sub-elements.
    imputs: 8 nodal (x, y, z) coordinates in NumPy arrays.
    Nodes arrangment: counterclockwise on both bottom and top surfaces;
      v1 (bottom) and v5 (top) on the same edge. 
    returns: The volume of the H8, float.
    """ 
    # Choose v0 and the three opposite faces on the T4 gives six T3s
    v0 = nodes[0] 
    T4_1_V = T4_V3(v0, nodes[5], nodes[2], nodes[1], absOn=absOn)
    T4_2_V = T4_V3(v0, nodes[5], nodes[6], nodes[2], absOn=absOn)
    T4_3_V = T4_V3(v0, nodes[2], nodes[6], nodes[3], absOn=absOn)
    T4_4_V = T4_V3(v0, nodes[6], nodes[7], nodes[3], absOn=absOn)
    T4_5_V = T4_V3(v0, nodes[7], nodes[6], nodes[5], absOn=absOn)
    T4_6_V = T4_V3(v0, nodes[4], nodes[7], nodes[5], absOn=absOn)

    # Return the 
    V = T4_1_V+T4_2_V+T4_3_V+T4_4_V+T4_5_V+T4_6_V     # volume of H8
    
    if absOn: V = abs(V)  
    return V  
 
# Example:
#nodes = [np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([1, 1, 0]),
#         np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([1, 0, 1]), 
#         np.array([1, 1, 1]), np.array([0, 1, 1])]

#volume = H8_V(nodes, absOn=False)

#print("Volume of the hexahedron:", volume) # result: 1.0

def volume_H8(nodeX, nodeY, nodeZ):
    '''
    Compute the volume of a general hexahedron, using its eight nodal
    coordinates (H8), with node-0 removed.
    It uses the vetor-matrix-vector (vMv) dot-product formula. 
    '''  
    z1, z2, z3, z4, z5, z6, z7 =  nodeZ               #  node-0 removed
    
    r=z3-z5; s=z1+z2; t=z2-z4; u=z2-z7; k=z5-z7    # to simplify matrix
    v=z2-z5; w=z1-z6; p=z3-z6; q=z1-z7; g=z4-z6; h=z4+z7
    
    Z = sp.Matrix([[0,   -r+z6,    z2,   -z5,-t-z6,   -v,    0],
                   [r-z6,    0, -w+z7,     0,    w,  q-r,   -p], 
                   [-z2,  w-z7,     0,    z7,    0,    u, t+z6],
                   [ z5,     0,   -z7,     0,-q+z6,   -k, r-z6],
                   [t+z6,   -w,     0,  q-z6,    0, -s+h,    g],
                   [v,    -q+r,    -u,     k,  s-h,    0, -t-r],
                   [0,       p, -t-z6, -r+z6,   -g,  t+r,    0]])
    
    V = (sp.Matrix(nodeX).T@Z@sp.Matrix(nodeY))[0]

    return V/12, Z

def polygon_area(polygon):
    '''Compute the area of a given polygon.
       It is the sum of the trapezoids formed by the polygon vertices
    '''
    x = polygon[:,0]
    y = polygon[:,1]
    
    # Roll array elements push up by 1 (with cyclic permutation)
    area = .5*np.abs(np.dot(x, np.roll(y,-1))-np.dot(y, np.roll(x,-1)))
    return area 

def primitive_gd(f_grad, x_init, eta, limit=1000, atol=1.e-6):
    ''' 
        A primitive solver to find the minimum of a 
        convex function f using the gradient descent algorithm. 
        Input: f_grad: gradient function of f that can be in nD 
               x_init--initial guess of x*
               eta--learning rate
               limit--maximum iterations
               atol--tolerance for the error of the minimum
    ''' 
    i = 0; x = x_init
    while lg.norm(f_grad(x)) > atol and i < limit:
        x = x - eta*f_grad(x)
        i += 1
        
    if lg.norm(f_grad(x)) > atol: print(f'Not yet converged:{atol}!!!')
    return x

def exam_DivTheorem0(V): 
    '''
       Examine a given vector field V with a constant divergence. 
       It prints out 
       * the total divergence of V in a spherical domain with radius 𝑎. 
       * The volume integral of the vector field over the sphere. 
       * The total flux of the vector field out of the sphere surface.
    '''
    x, y, z, 𝜃, 𝜑 = symbols('x, y, z, 𝜃, 𝜑', real=True)
    a = symbols('a', positive=True)              # radius of the sphere
    X = Matrix([x, y, z])

    V_ball = 4*sp.pi*a**3/3        # volume of the sphere with radius a
    div_V = (gr.div_vf(V, X)).simplify()    #  divergence of the vector
    div_VV = div_V*V_ball                  # integral of the divergence 
    print(f" Divergence of the vector field = {div_V}")
    print(f" Integral of the divergence of the vector field = {div_VV}")

    # Integral over an enclosed sphere: 
    x_ = a*sin(𝜃)*cos(𝜑)   # x_, y_, and z_ are x,y,z on sphere surface
    y_ = a*sin(𝜃)*sin(𝜑)
    z_ = a*cos(𝜃) 

    r_𝜃 = Matrix([x_.diff(𝜃), y_.diff(𝜃), z_.diff(𝜃)])       # Jacobian
    r_𝜑 = Matrix([x_.diff(𝜑), y_.diff(𝜑), z_.diff(𝜑)])
    n = gr.cross2vectors(r_𝜃, r_𝜑, sympy=True)     # normal vector on S

    dic = {x:x_, y:y_, z:z_}
    Vns = integrate(integrate(V.subs(dic).T@n, (𝜑,0,2*sp.pi)),(𝜃,0,sp.pi)) 
    print(f" Flux of the vector field = {Vns[0]}")
    print(f" Is the divergence theorem holds? {Vns[0]==div_VV}")

def plot_Lagrange_interp2d(f_true, nodes_cases, *p_name):
    '''Fit a 2D true function using a set of Lagrange shape functions.   
       Plot the fitted and the true function.
    '''
    x, y =symbols('x, y')                 # define 2D physic coordinates
    plt.rcParams.update({'font.size': 6})
    fig_s = plt.figure(figsize=(5,5))   
    plt.ioff() #plt.ion()
    
    X, Y = np.mgrid[-1:1:100j, -1:1:100j]              # grids for plots
    levels = np.linspace(0, 2, 21)
    ax = fig_s.add_subplot(2,2,1)
    ax.contour(X, Y, f_true(X,Y),cmap='jet',linewidths=.5,levels=levels) 
    ax.set_aspect(1.0)
    plt.title(f'True function')
    
    for k in range(len(nodes_cases)):        
        ax = fig_s.add_subplot(2,2,k+2)
        nodexi = np.linspace(-1., 1., nodes_cases[k][0]) # nodes along x
        nodeyj = np.linspace(-1., 1., nodes_cases[k][1])       # along y
        _, _, Nijs = N_Lagrange_fxy(nodexi, nodeyj)    # shape functions

        f_lagrange = 0
        for i in range(len(nodexi)):    # generate the Lagrange function
            for j in range(len(nodeyj)):
                f_lagrange += Nijs[i,j]*f_true(nodexi[i], nodeyj[j])
            
        np_f = lambdify([x, y],f_lagrange)   # convert to numpy function 
        
        plt.contour(X,Y,np_f(X,Y),cmap='jet',linewidths=.5,levels=levels) 
        ax.set_aspect(1.0)
        rmse = sqrt(np.mean((np_f(X,Y)-f_true(X,Y))**2)) 
        plt.title(f'Fitted with {nodes_cases[k]}, rmse={rmse:.4e}')
        
    plt.savefig('imagesDI/'+p_name[0]+'.png',dpi=500,bbox_inches='tight')
    #plt.show()

def volume_T4(nodeX, nodeY, nodeZ):
    '''
    Compute the volume of a tetrahedron, using the three nodal
    coordinates of an T4 (with node-0 removed).
    It uses the vetor-matrix-vector (vMv) dot-product formula. 
    Nodes arrangment: counterclockwise on the base triangle, and the
    thumb points to the tip node. 
    '''  
    #z1, z2, z3
    z1,  z3, z4 = nodeZ               # nodeZ has z1, z2, z3: z3⋅(z1×z2)  

    Z = sp.Matrix([[  0,  z4,  -z3],
                   [-z4,   0,   z1], 
                   [ z3, -z1,    0]])
    
    V = (sp.Matrix(nodeX).T@Z@sp.Matrix(nodeY))[0]

    return V/6, Z

def T3_nA3D(r0, r1, r2, module='sympy'):
    '''Compute the cross-product of two edge vectors of a T3 defined 
       by r0, r1, r2 in a 3D space. 
       return: nA, normal rector of the T3 element. 
    '''
    edge1 = r1 - r0                       # form two edge vectors of T3
    edge2 = r2 - r0
    if module == 'sympy':
        nA = sp.Rational(1,2)*edge1.cross(edge2)
    else:
        nA = np.cross(edge1, edge2)/2
    return nA

def A_3D_surface(nodes, faces, n_faces, p_area=True): 
    '''Compute the areas of the surfaces of a tessellated surface with 
       multiple T3 elements in 3D domain. Written with help of ChatGPT.
       input: nodes -   np array, nodes define the vertices of domain.
              faces -   list, faces that forms the pyramid. 
              n_faces - int, number of first surfaces to be computated. 
                        If n_faces = 4, only first 4 T3 is computed. 
              p_area -  print out the areas of the computated T3 faces.
       return: areas -  list, areas of all the computed faces.
               A -      float, the total area of all computed faces.
    ''' 
    # Compute the areas
    areas = []                    # list holds the areas of T3 elements
    for face in faces[:n_faces]:
        r0, r1, r2 = nodes[face[0]], nodes[face[1]], nodes[face[2]]
        area = T3_A3D(r0, r1, r2, module='numpy')    # compute the area
        areas.append(area) 

    if p_area: 
        for i, area in enumerate(areas):    # Print the area of each T3 
            print(f"Area of T3 {i+1}: {area:.6f}")

    A = sum(areas)                       # sum all the areas up
    print(f"The total area of the whole surface = {A:.6f}")  

    return A, areas 

def A_3D_surface(nodes, faces, n_faces, p_area=True): 
    '''Compute the areas of the surfaces of a tessellated surface with 
       multiple T3 elements in 3D domain. Written with help of ChatGPT.
       input: nodes -   np array, nodes define the vertices of domain.
              faces -   list, faces that forms the pyramid. 
              n_faces - int, number of first surfaces to be computated. 
                        If n_faces = 4, only first 4 T3 is computed. 
              p_area -  print out the areas of the computated T3 faces.
       return: areas -  list, areas of all the computed faces.
               A -      float, the total area of all computed faces.
    ''' 
    # Compute the areas
    areas = []                    # list holds the areas of T3 elements
    for face in faces[:n_faces]:
        r0, r1, r2 = nodes[face[0]], nodes[face[1]], nodes[face[2]]
        area = T3_A3D(r0, r1, r2, module='numpy')    # compute the area
        areas.append(area) 

    if p_area: 
        for i, area in enumerate(areas):    # Print the area of each T3 
            print(f"Area of T3 {i+1}: {area:.6f}")

    A = sum(areas)                       # sum all the areas up
    print(f"The total area of the whole surface = {A:.6f}")  

    return A, areas 

def Surf_Int_sf(s_f, nodes, faces, n_faces, areas, p_Ifs=True): 
    '''Compute the integrals of a scalar function over the surfaces of 
       a tessellated surface with multiple T3s (such as a pyramid). 
       input: s_f -     scalar function in numpy. 
              nodes -   np array, vertices of the tessellated surface.
              faces -   list, T3 faces forming the tessellated surface. 
              n_faces - number of first T3 surfaces to be computated. 
                        If n_faces = 4, the two base T3 is excluded. 
              areas -   list, areas of these T3s. 
              p_Ifs -   print integral (I) values for all computed T3s. 
       return: I_f - float, total integral of s_f over computed faces.
               I_fs  -  list, integrals of s_f over the computed T3s.
    ''' 
    # Computing the integral of the function s_f: 
    I_fs = []    # list holds integrals of the function for T3 elements
    for i, face in enumerate(faces[:n_faces]):
        r0, r1, r2 = nodes[face[0]], nodes[face[1]], nodes[face[2]]

        # Compute the function values at three nodes: 
        fi = [s_f(*xyz) for xyz in [r0, r1, r2]]
        fc = sum(fi)/len(fi)  
        if p_Ifs:  
            print(f" 𝑓 values at T3-{i+1} nodes: = {fi}, fc = {fc}")    
        
        I_f = fc*areas[i]         # integral of scalar function over T3
        I_fs.append(I_f) 
    
    if p_Ifs: 
        for i, I_f in enumerate(I_fs):       # Print the I_f of each T3 
            display(Math(f" \\text{{I_f over T3-}}{i+1}= {latex(I_f)}"))
        
    I_f = sum(I_fs)
    display(Math(f" \\text{{Total integral over the domain, I_f }}=\
                                                     {latex(I_f)}"))
    return I_f, I_fs

def normalVector3D_surface(nodes, faces, n_faces, p_nA=True): 
    '''Compute the normal vector of the surfaces of a tessellated 
       surface with multiple T3s (pyramid, etc.). With help of ChatGPT.
       input: nodes -   np array, define the vertices of the pyramid.
              faces -   list, faces that forms the pyramid. 
              n_faces - number of first surfaces used in computation. 
                        If n_faces = 4, the two base T3 is excluded. 
              p_nA -    print out the nAs for all computed T3s.
       return: nAs -    np array, normal vectors of all computed faces.
    '''  
    # Compute the normal vector of a T3: 
    nAs = []                          # list for normal vectors of a T3
    for face in faces[:n_faces]:
        r0, r1, r2 = nodes[face[0]], nodes[face[1]], nodes[face[2]]
        nA = T3_nA3D(r0, r1, r2, module='numpy')  
        nAs.append(nA) 

    if p_nA:
        for i, nA in enumerate(nAs):         # Print the nAs of each T3 
            display(Math(f" \\text{{Normal vector of T3-}}{i+1}=\
                                                   {latex(nA)}"))
    return nAs

def Surf_Int_Vf(V_f, nodes, faces, n_faces, nAs, p_Vfc=True): 
    '''Compute the flux of a vector function across the surfaces of 
       a tessellated surface with multiple T3s (such as a pyramid). 
       input: V_f -     vector function in numpy. 
              nodes -   np array, define the vertices of the pyramid.
              faces -   list, faces that forms the pyramid. 
              n_faces - number of first surfaces used in computation. 
                        If n_faces = 4, the two base T3 is excluded. 
              nAs -     np array, normal vectors of all T3s. 
              p_nA -    print out fluxes for all T3s
       return: flux_Vfs - list, fluxes (scalars) of V_f over T3 faces.
               flux_Vf  - total flux (scalars) over the whole surface.
    '''      
    x, y, z = symbols('x, y, z', real=True)

    # Computing the flux of the vector function
    flux_Vfs = []                            # list of integrals of V_f
    for i, face in enumerate(faces[:n_faces]):
        r0, r1, r2 = nodes[face[0]], nodes[face[1]], nodes[face[2]]
        rc = (r0 + r1 + r2)/3
        node_c = nodes.mean(0)

        # Compute the vector function values at the center of a T3: 
        V_f_c = V_f.subs({x:rc[0], y:rc[1], z:rc[2]})  
        if p_Vfc:
            display(Math(f" \\text{{V_f values at the center of \
                              T3-{i+1} = }}{latex(V_f_c.evalf(6).T)}"))

        flux_Vf = V_f_c.dot(nAs[i]) # flux of vector function over a T3
        flux_Vfs.append(flux_Vf) 

    for i, flux_Vf in enumerate(flux_Vfs):   # Print flux_Vf of each T3 
        display(Math(f" \\text{{Flux over T3-}}{i+1}=\
                              {latex(flux_Vf.evalf(5))}"))

    flux_Vf = sum(flux_Vfs)
    display(Math(f" \\text{{The total flux of Vf}} = \
                        {latex(flux_Vf.evalf(5))}"))

    return flux_Vf, flux_Vfs

def exam_DivThm_sphere(V, X, a): 
    '''
       Examine a given vector field V with a integrable divergence 
       over a spherical domain over r in [0, a]. The code prints out 
       1 The divergence of the vector field at any point in the domain.
       2 The total divergence of V over spherical domain with radius 𝑎. 
       3 The total flux of the vector field out of the sphere surface.
       4 Whether or not the divergence theorem holds. 
    '''
    r = sp.symbols('r', nonnegative=True)  # radial coordinate, spheres
    𝜃, 𝜑 = symbols('𝜃, 𝜑', real=True)
    x, y, z = X  

    #V_ball = 4*sp.pi*a**3/3 # volume of sphere of a; used for checking
    div_V = (div_vf(V, X)).simplify()    #  divergence of the vector
    display(Math(f" \\text{{Divergence of the vector field}}\
                                          = {latex(div_V)}"))

    # Integral over an enclosed sphere: 
    x_ = r*sin(𝜃)*cos(𝜑)   # x_, y_, and z_ are x,y,z on sphere surface
    y_ = r*sin(𝜃)*sin(𝜑)
    z_ = r*cos(𝜃) 
    dic = {x:x_, y:y_, z:z_}

    # Volume integration over the spherical ball with r=a: 
    fI = div_V.subs(dic)*(r**2 *sin(𝜃))           # detJ = r**2 *sin(𝜃)
    Int = sp.integrate; pi = sp.pi
    div_VV = Int(Int(Int(fI,(r,0,a)),(𝜃,0,pi)),(𝜑,0,2*pi)).simplify()
    #div_VV = div_V*V_ball     # total divergence, if div_V is constant
    
    # Surface integration over the surface of the sphere at r=a:  
    # compute the normal n: (method-1: general ×, 2 for sphere) 
    r_𝜃 = sp.Matrix([x_.diff(𝜃), y_.diff(𝜃), z_.diff(𝜃)])    # Jacobian
    r_𝜑 = sp.Matrix([x_.diff(𝜑), y_.diff(𝜑), z_.diff(𝜑)])
    
    #method-1:
    #n_ = cross2vectors(r_𝜃, r_𝜑, sympy=True).applyfunc(sp.simplify) 
    #norm_p = n_.norm().simplify().args[1].args[0]  # normal vector on S
    #n = n_/(a**2*norm_p) 
    
    #method-2:
    n = X.subs(dic)/a        # normal n on S: n = 1/r on sphere surface 
    #display(Math(f" \\text{{ unit normal n = }}{latex(n)}"))
    
    fSI = ((V.T@n)*(r**2 *sin(𝜃))).subs(dic).subs(r,a)[0].simplify()
    #display(Math(f" \\text{{ Surface integrand = }}{latex(fSI)}"))
    Vns=Int(Int(fSI,(𝜑,0,2*pi)),(𝜃,0,pi)).simplify() 
    
    # print out the results: 
    display(Math(f" \\text{{Total flux of the vector field across the \
                                    sphere surface}} = {latex(Vns)}"))
    display(Math(f" \\text{{Total divergence over the sphere}} \
                                                  = {latex(div_VV)}"))
    print(f" Is the divergence theorem holds? {Vns==div_VV}")

def dict_gen(v, v_):
    '''Generate a dictionary pairing v and v_'''
    dict_ = {}
    for key, value in zip(v, v_):                  # pairing the values
        dict_[key] = value
    return dict_

def workOnLine(V_f, r0, r1, x, y, z, t, t0, t1): 
    '''Compute the work done by force vector field V_f along a straight
       line path defined by position vectors r0 to r1, from t0 to t1.
       A line integral using the parameterization method (t).
       written based on a suggestion by ChatGPT. 
       input: V_f, list, components of the force vector field.
              r0,  list, coordinates of the starting point.
              r1,  list, coordinates of the ending point.
        return: work, float, work done from t0 to t1
    '''
    #x, y, z = sp.symbols('x, y, z')          # define the symbols
    # Parameterize the straight line path using t (linear):
    r_t = sp.Matrix(r0) + t * (sp.Matrix(r1) - sp.Matrix(r0)) 
    x_t, y_t, z_t = r_t

    # Compute the differential components of the path
    dx_dt = sp.diff(x_t, t)
    dy_dt = sp.diff(y_t, t)
    dz_dt = sp.diff(z_t, t)

    # Compute the differential vector d(r)
    dr_dt = sp.Matrix([dx_dt, dy_dt, dz_dt])

    # Substitute the parameterized path into the force field
    F_sub = sp.Matrix(V_f).subs({x: x_t, y: y_t, z: z_t})

    # Compute the dot product F ⋅ dr:
    F_dot_dr = F_sub.dot(dr_dt)

    # Integrate the dot product over the parameter t
    work = sp.integrate(F_dot_dr, (t, t0, t1))

    return work

def T3_A3D(r0, r1, r2, module='sympy'):
    '''Compute the area of T3 element in 3D. Using the cross-product of 
       two edge vectors of a T3 defined by position vectors r0, r1, r2 
       in a 3D space. 
       return: A, the area of the T3 element. 
    '''
    r10 = r1 - r0                         # form edge vector 1 of T3
    r20 = r2 - r0                         # form edge vector 2 of T3
    if module == 'sympy':
        cross_product = r10.cross(r20)
        A = sp.Rational(1, 2) * cross_product.norm() 
    else:
        cross_product = np.cross(r10, r20)
        A = np.linalg.norm(cross_product)/2
    return A

def find_potential_f2D(Vx, Vy, X):
    '''
       Find the potential function 𝜙 for a given two vector field
       components in 2D. The vector field must be a gradient field. 
       return: 𝜙, potetial fuction.
    '''
    x, y = X
    # Use condition: ∂𝜙/∂x = Vx 
    C = sp.Function('C')(y)        # define symbolic integral functions
    𝜙C = sp.integrate(Vx, (x)) + C         # 𝜙 with C(y); used V𝑥 = ∂𝜙/∂𝑥  

    # use condition: Vy = ∂𝜙/∂y
    eqn = 𝜙C.diff(y)- Vy                  # create an equation to solve 
    sln_C = sp.dsolve(eqn, C)                  # solve the eqn for C(y)

    𝜙C = 𝜙C.subs(C, sln_C.args[1])   # put solution of C(y) back to 𝜙C
    𝜙  = 𝜙C.subs(𝜙C.args[0],0).simplify() # set integral constant to 0
    printx('𝜙')

    # Check all conditions: 
    printx('(𝜙.diff(x)-Vx).simplify()') # check condition: Vx=∂𝜙/∂x
    printx('(𝜙.diff(y)-Vy).simplify()') # check condition: Vy=∂𝜙/∂y
    
    return 𝜙

def gsm_polygon(f, X, nodes, p_out=True):
    '''
    Computes the smoothed gradient of a given scalar function f (Sympy)
    over a general polygonal domain defined by its nodes. Example: 
    nodes = np.array([[-1,-1], [ 1,-1], [ 1, 1], [-1, 1]]). 
    return: s_gradf - smoothed gradient of f.
            gradf - exact gradient of f.
    '''
    x, y = X                      
    t = sp.symbols('t')                # parameter for line integration
    
    #A = gr.polygon_area(nodes)      # area of the polygon, alternative
    A = polygon_area_green(nodes)
    #gr.printx('A')
    gradf = gr.grad_f(f, X).subs(dict(zip(X, np.mean(nodes, axis=0))))
    
    # We roll the nodes back by 1, and use it to form the polygon edges.  
    nodes_r1 = np.roll(nodes, -1, axis=0)             
    r = nodes + t*(nodes_r1-nodes)   # parameterizaton with t for edges
    drdt = sp.diff(sp.Matrix(r), t)    # drdt[i,0]=dxdt; drdt[i,1]=dydt
    t_limits = (t, 0, 1)             # limits for t (0-1 for each edge)
    
    Int = sp.integrate
    s_dfdx = 0; s_dfdy = 0
    for i in range(len(nodes)): # line integrals over dt along each edge
        s_dfdx += Int((f*drdt[i,1]).subs({x:r[i,0], y:r[i,1]}),t_limits)
        s_dfdy += Int((f*drdt[i,0]).subs({x:r[i,0], y:r[i,1]}),t_limits)
        
    s_gradf = sp.simplify(sp.Matrix([s_dfdx, -s_dfdy])/A) 
 
    if p_out: 
        display(Math(f"\\text{{Exact gradient of f = }}{latex(gradf)}"))
        display(Math(f"\\text{{Smoothed gradient of f}}=\
                                       {latex(s_gradf)}")) 
    return s_gradf, gradf

def polygon_area_green(vertices):
    """
    Compute the area of a polygon using its vertices using 
    Green's theorem.
    inputs: vertices - List of tuples/lists, each contains x and y 
                     coordinates of a vertex of the polygon, eg: 
                     [(-1, -1), (1, -1), (3, -1), (1, 1), (-1, 1)]
    return: Area of the polygon.
    Code suggested by ChatGPT. 
    """
    n = len(vertices)

    area = 0.0

    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i+1)%n]  # Next vertex, and 
                                    # wrapping around to the 1st vertex
        area += (x1*y2 - y1*x2)
    
    return area/2

def nabla_f0(f, dfn = 2, dim = 2, smplfy=1):
    '''Compute the nabla operator ∇^2 (dfn = 2) or ∇^4 (dfn = 4) on 
       (sympy) function f  defined in 2D (dim = 2) or 3D (dim = 3). 
       smplfy=1: use sp.simplify(); =2 use sp.expand() for polynomials.
       return: nf - the nabla of f. 
    '''
    x, y, z = sp.symbols("x, y, z")
    smpfy = sp.simplify if smplfy == 1 else sp.expand 
    if dim == 2:                                                # 2D 
        if dfn == 2: 
            n2f=smpfy((f.diff(x,2)+f.diff(y,2)))                # ∇^2 
            nf = n2f
        elif dfn == 4:                                          # ∇^4
            n2f=smpfy((f.diff(x,2)+f.diff(y,2)))                # ∇^2 
            n4f=smpfy((n2f.diff(x,2)+n2f.diff(y,2))) 
            nf = n4f
        else: print(f'dfn must be set either 2 or 4')
    elif dim == 3:                                              # 3D
        if dfn == 2: 
            n2f=smpfy((f.diff(x,2)+f.diff(y,2)+f.diff(z,2)))    # ∇^2 
            nf = n2f
        elif dfn == 4:                                          # ∇^4
            n2f=smpfy((f.diff(x,2)+f.diff(y,2)+f.diff(z,2)))    # ∇^2 
            n4f=smpfy((n2f.diff(x,2)+n2f.diff(y,2)+n2f.diff(z,2))) 
            nf = n4f
        else: print(f'dfn must be set either 2 or 4')
    else: print(f'dim must be set either 2 or 3')
    return nf 

def polyBases0(order, dim=2):
    '''Create polynomial bases of the same given order for 
       2D (dim=2) or 3D (dim=3)
    ''' 
    x, y, z = sp.symbols("x, y, z")
    if dim == 2:
        basis = Matrix([x**i * y**(order-i) for i in range(order+1)])
    elif dim == 3: 
        basis = Matrix([x**i*y**j*z**k for i in range(order+1) for j in 
              range(order+1) for k in range(order+1) if i+j+k==order])
    else: print(f'dim must be set either 2 or 3')
    #print(f"Polynomials bases, complete order {order}:\n {basis.T}")
    return basis

def create_nbf0(order, X, dfn=4):
    '''Creates polynomial nabla basis functions that satisfy 
       ∇^2=0  (dfn=2, order>=2) or ∇^4=0 (dfn=4, order>=4) defined in 
       2D (dim=2) or 3D (dim=3). 
       return: polynomial basis functions, nbf (sympy Matrix).  
       '''
    dim = len(X)
    print(f'Dimension={dim}, order={order}, ∇^={dfn}')
    if dim == 3: 
        x, y, z = X
        𝜓 = sp.Function("𝜓")(x, y, z)        # a displacement-function
    if dim == 2:
        x, y = X  
        𝜓 = sp.Function("𝜓")(x, y)           # a displacement-function

    c = sp.symbols("c0:99")    # coeffs, define more than what we need
    
    # create polynomial bases of the same given order
    basis = polyBases(order, dim=dim)
    #print(f"Polynomials bases, complete order {order}:\n {basis.T}")
    
    cs = Matrix([c[i] for i in range(len(basis))])     # coeffs needed
    
    # compute total number of constants:
    tn_cs = (order+2)*(order+1)/2 if dim==3 else order+1 
    print(f'No. of constants={len(cs)},  should be {tn_cs}')
    
    𝜓 = (cs.T@basis)[0]    # create the function by linear combination
    #print(f"Initial 𝜓 with complete order {order} for 𝜓:\n {𝜓}")
    
    # form the order-reduced polynomial bases after applying ∇^2:
    basisR = polyBases(order-dfn, dim=dim)
    #print(f"Polynomials bases after ∇^2:\n {basisR.T} {len(basisR)}")
    
    nab𝜓 = nabla_f0(𝜓, dfn=dfn, dim=dim, smplfy=2)  #∇^4, 3D 
    
    # collect terms for order-reduced bases, to group the coeffs  
    for i in range(len(basisR)):       
        nab𝜓 = nab𝜓.collect(basisR[i]) #print(nab𝜓)
    
    # form equations of relations of the coeffs. 
    eqs = [nab𝜓.coeff(basisR[i]) for i in range(len(basisR))]  
    if len(eqs)==1: eqs = [nab𝜓]
        
    sln_cs=sp.solve(eqs, cs)  #print(sln_cs)
    keys_s = list(sln_cs.keys())                     # solution keys
    
    # get the remaining coefficients in the expression
    keys_r  = list(filter(lambda x: x not in keys_s, cs))
    
    ns = order-dfn+1   # number of solved coefficients
    nr = dfn           # number of remaining coefficients
    if dim == 3: 
        ns = (order-dfn+2)*(order-dfn+1)/2
        nr = tn_cs-ns  
    #print(len(keys_s)," should be",ns,"  ",len(keys_r)," should be",nr)
    
    𝜓_ci = 𝜓.subs(sln_cs).expand()           # get the 𝜓 with keys_r
    
    # collect terms based on keys_r, to group polynomials  
    for i in range(len(keys_r)):
        𝜓_ci = 𝜓_ci.collect(keys_r[i]) 
    
    # group polynomials, based on keys_r
    nbf=Matrix([(𝜓_ci.coeff(keys_r[i])).expand() 
                 for i in range(len(keys_r))]) 
    
    nab𝜓 = nabla_f0(nbf, dfn=dfn, dim=dim, smplfy=2)  #∇^4, 3D 
    
    print(f'Is ∇^{dfn} 𝜓=0? {nab𝜓.T@nab𝜓}! Number of bases={len(nbf)}')
    return nbf

def nabla_f(f, X, dfn = 2, smplfy=1):
    '''Compute the nabla operator ∇^2 (dfn = 2) or ∇^4 (dfn = 4) on 
       (sympy) function f  defined in 2D (dim = 2) or 3D (dim = 3). 
       smplfy=1: use sp.simplify(); =2 use sp.expand() for polynomials.
       return: nf - the nabla of f. 
    '''
    #x, y, z = sp.symbols("x, y, z")
    dim = len(X) 
    smpfy = sp.simplify if smplfy == 1 else sp.expand 
    if dim == 2:                                                # 2D
        x, y = X  
        if dfn == 2: 
            n2f=smpfy((f.diff(x,2)+f.diff(y,2)))                # ∇^2 
            nf = n2f
        elif dfn == 4:                                          # ∇^4
            n2f=smpfy((f.diff(x,2)+f.diff(y,2)))                # ∇^2 
            n4f=smpfy((n2f.diff(x,2)+n2f.diff(y,2))) 
            nf = n4f
        else: print(f'dfn must be set either 2 or 4')
    elif dim == 3:                                              # 3D
        x, y, z = X
        if dfn == 2: 
            n2f=smpfy((f.diff(x,2)+f.diff(y,2)+f.diff(z,2)))    # ∇^2 
            nf = n2f
        elif dfn == 4:                                          # ∇^4
            n2f=smpfy((f.diff(x,2)+f.diff(y,2)+f.diff(z,2)))    # ∇^2 
            n4f=smpfy((n2f.diff(x,2)+n2f.diff(y,2)+n2f.diff(z,2))) 
            nf = n4f
        else: print(f'dfn must be set either 2 or 4')
    else: print(f'dim must be set either 2 or 3')
    return nf 

def polyBases(order, X):
    '''Create polynomial bases of a given order for 2D (dim=2) 
       or 3D (dim=3)
    ''' 
    dim = len(X) 
    if dim == 2:
        x, y = X  
        basis = Matrix([x**i * y**(order-i) for i in range(order+1)])
    elif dim == 3: 
        x, y, z = X
        basis = Matrix([x**i*y**j*z**k for i in range(order+1) for j in 
               range(order+1) for k in range(order+1) if i+j+k==order])
    else: print(f'dim must be set either 2 or 3')
    #print(f"Polynomials bases, complete order {order}:\n {basis.T}")
    return basis

def create_nbf(order, X, dfn=4, prnt=True):
    '''Creates polynomial basis functions that satisfy 
       ∇^2=0  (dfn=2, order>=2) or ∇^4=0 (dfn=4, order>=4) defined in 
       2D (dim=2) or 3D (dim=3). 
       return: polynomial basis functions, nbf (sympy Matrix).  
       '''
    dim = len(X)
    if prnt: print(f'Dimension={dim}, order={order}, ∇^={dfn}')
    if dim == 3: 
        x, y, z = X
        𝜓 = sp.Function("𝜓")(x, y, z)        # a displacement-function
    if dim == 2:
        x, y = X  
        𝜓 = sp.Function("𝜓")(x, y)           # a displacement-function

    c = sp.symbols("c0:99")    # coeffs, define more than what we need
    
    # create polynomial bases of the same given order
    basis = polyBases(order, X)
    #print(f"Polynomials bases, complete order {order}:\n {basis.T}")
    
    cs = Matrix([c[i] for i in range(len(basis))])     # coeffs needed
    
    # compute total number of constants:
    tn_cs = (order+2)*(order+1)/2 if dim==3 else order+1 
    if prnt: print(f'No. of constants={len(cs)},  should be {tn_cs}')
    
    𝜓 = (cs.T@basis)[0]    # create the function by linear combination
    #print(f"Initial 𝜓 with complete order {order} for 𝜓:\n {𝜓}")
    
    # form the order-reduced polynomial bases after applying ∇^2:
    basisR = polyBases(order-dfn, X)
    #print(f"Polynomials bases after ∇^2:\n {basisR.T} {len(basisR)}")
    
    nab𝜓 = nabla_f(𝜓, X, dfn=dfn, smplfy=2)  #∇^4, 3D 
    
    # collect terms for order-reduced bases, to group the coeffs  
    for i in range(len(basisR)):       
        nab𝜓 = nab𝜓.collect(basisR[i]) #print(nab𝜓)
    
    # form equations of relations of the coeffs. 
    eqs = [nab𝜓.coeff(basisR[i]) for i in range(len(basisR))]  
    if len(eqs)==1: eqs = [nab𝜓]
        
    sln_cs=sp.solve(eqs, cs)  #print(sln_cs)
    keys_s = list(sln_cs.keys())                     # solution keys
    
    # get the remaining coefficients in the expression
    keys_r  = list(filter(lambda x: x not in keys_s, cs))
    
    ns = order-dfn+1   # number of solved coefficients
    nr = dfn           # number of remaining coefficients
    if dim == 3: 
        ns = (order-dfn+2)*(order-dfn+1)/2
        nr = tn_cs-ns  
    #print(len(keys_s)," should be",ns,"  ",len(keys_r)," should be",nr)
    
    𝜓_ci = 𝜓.subs(sln_cs).expand()           # get the 𝜓 with keys_r
    
    # collect terms based on keys_r, to group polynomials  
    for i in range(len(keys_r)):
        𝜓_ci = 𝜓_ci.collect(keys_r[i]) 
    
    # group polynomials, based on keys_r
    nbf=Matrix([(𝜓_ci.coeff(keys_r[i])).expand() 
                 for i in range(len(keys_r))]) 
    
    nab𝜓 = nabla_f(nbf, X, dfn=dfn, smplfy=2)  #∇^4, 3D 
    
    if prnt: print(f'Is ∇^{dfn} 𝜓=0? {nab𝜓.T@nab𝜓}! Number of bases={len(nbf)}')
    return nbf

def get_coeffs(monomials, expression, prnt=True):
    '''Collect the coefficients of each of the `monomials` in descending 
       order in the given `expression`. All in sympy expressions.
       return: coeff - the list of coefficients.
    ''' 
    coeff = [] 
    expr = expression 
    for i, m in enumerate(monomials): 
        coeff.append(expr.collect(m).coeff(m))        #get coefficients
        expr = (expr - coeff[i]*m).simplify().expand()
    coeff.append(expr)                     # The collected coefficients 
    
    if prnt: 
        for i in range(len(coeff)):          
            display(Math(f" \\text{{$c$}}{i}={latex(coeff[i])}")) 
    return coeff                             # the list of coefficients

def split_dict_n(long_dict, n):
    '''Split the given long dictionary into n smaller ones. It returns 
       a list of n smaller dictionaries. Suggested by ChatGPT.
    '''
    # Convert the dictionary items to an iterable
    iter_dict = iter(long_dict.items())
    
    # Calculate the approximate size of each split
    chunk_size = len(long_dict) // n
    
    # Create a list of dictionaries by slicing the iterator into n parts
    result = [dict(itertools.islice(iter_dict, chunk_size)) \
                                                 for _ in range(n - 1)]
    
    # Add the remaining items to the last dictionary
    result.append(dict(iter_dict))
    
    return result

def comptblty_check(X, E, 𝜈, U, 𝜀, 𝜎, given = 'U'): 
    '''Check compatibility conditions for given a displacement field.
       It returns the compatibility expressions. 
    '''
    x, y, z = X
    if given == "U":        # If the displacement expressions are given
        u, v, w = U
        𝜀xx = u.diff(x); 𝜀yy=v.diff(y); 𝜀zz= w.diff(z) # compute strains
        𝜀yz = ((w.diff(y)+v.diff(z))/2).simplify()
        𝜀xz = ((w.diff(x)+u.diff(z))/2).simplify()
        𝜀xy = ((v.diff(x)+u.diff(y))/2).simplify() 
    elif given == "S":            # If the stress expressions are given
        𝜎xx, 𝜎yy, 𝜎zz, 𝜎yz, 𝜎xz, 𝜎xy = 𝜎
        𝜀xx = (𝜎xx - 𝜈*𝜎yy - 𝜈*𝜎zz)/E
        𝜀yy = (𝜎yy - 𝜈*𝜎xx - 𝜈*𝜎zz)/E
        𝜀zz = (𝜎zz - 𝜈*𝜎xx - 𝜈*𝜎yy)/E
        𝜀yz = (1+𝜈)*𝜎yz/E; 𝜀xz = (1+𝜈)*𝜎xz/E; 𝜀xy = (1+𝜈)*𝜎xy/E 
    else: 
        𝜀xx, 𝜀yy, 𝜀zz, 𝜀yz, 𝜀xz, 𝜀xy = 𝜀
        
    exprs = []  
    exprs.append((𝜀xx.diff(y,2)+𝜀yy.diff(x,2) - 2*𝜀xy.diff(x).diff(y))\
                                                           .simplify()) 
    exprs.append((𝜀xx.diff(z,2)+𝜀zz.diff(x,2) - 2*𝜀xz.diff(x).diff(z))\
                                                           .simplify()) 
    exprs.append((𝜀yy.diff(z,2)+𝜀zz.diff(y,2) - 2*𝜀yz.diff(y).diff(z))\
                                                           .simplify()) 
    exprs.append((𝜀xy.diff(z,2) + 𝜀zz.diff(x).diff(y) \
                - 𝜀yz.diff(x).diff(z)- 𝜀xz.diff(y).diff(z)).simplify())
    exprs.append((𝜀xz.diff(y,2) + 𝜀yy.diff(x).diff(z) \
                - 𝜀xy.diff(y).diff(z)- 𝜀yz.diff(x).diff(y)).simplify()) 
    exprs.append((𝜀yz.diff(x,2) + 𝜀xx.diff(y).diff(z) \
                - 𝜀xz.diff(x).diff(y)- 𝜀xy.diff(x).diff(z)).simplify()) 
    return exprs

def stress2strain(E, 𝜈, 𝜎): 
    '''3D constitutive equation, from stresses to strains'''
    𝜎xx, 𝜎yy, 𝜎zz, 𝜎yz, 𝜎xz, 𝜎xy = 𝜎
    𝜀xx = ((𝜎xx - 𝜈*𝜎yy - 𝜈*𝜎zz)/E).simplify()
    𝜀yy = ((𝜎yy - 𝜈*𝜎xx - 𝜈*𝜎zz)/E).simplify()
    𝜀zz = ((𝜎zz - 𝜈*𝜎xx - 𝜈*𝜎yy)/E).simplify()
    𝜀yz = (1+𝜈)*𝜎yz/E; 𝜀xz = (1+𝜈)*𝜎xz/E; 𝜀xy = (1+𝜈)*𝜎xy/E 
    return [𝜀xx,𝜀yy,𝜀zz,𝜀yz,𝜀xz,𝜀xy]

def strain2stress(E, 𝜈, 𝜀): 
    '''3D constitutive equation, from strains to stresses'''
    E_ = E/((1+𝜈)*(1-2*𝜈)) 
    𝜀xx, 𝜀yy, 𝜀zz, 𝜀yz, 𝜀xz, 𝜀xy = 𝜀
    𝜎xx = E_*( (1-𝜈)*𝜀xx + 𝜈*(𝜀yy + 𝜀zz) ).simplify()    
    𝜎yy = E_*( (1-𝜈)*𝜀yy + 𝜈*(𝜀xx + 𝜀zz) ).simplify()   
    𝜎zz = E_*( (1-𝜈)*𝜀xx + 𝜈*(𝜀xx + 𝜀yy) ).simplify()  
    𝜎yz = E*𝜀yz/(1+𝜈); 𝜎xz = E*𝜀xz/(1+𝜈); 𝜎xy = E*𝜀xy/(1+𝜈) 
    return [𝜎xx,𝜎yy,𝜎zz,𝜎yz,𝜎xz,𝜎xy] 

def remove_zero_exprs(exprs):
    print(f" Total number of expressions = {len(exprs)}")
    expr0s = [i for i, expr in enumerate(exprs) if expr == 0]
    #print(f" The Index of the zero expressions: {expr0s[0]}...{expr0s[-1]}")
    exprs = [exprs[i] for i in range(len(exprs)) if i not in expr0s]
    print(f" Number of nonzero expressions = {len(exprs)}") 
    return exprs

def displacement2stress(X, G, 𝜈, U):
    '''Compute the stresses for given displacements 
       U: array like with u, v, w.
       G: shear modulus, 𝜈: Poisson's ratio.
       It returns 6 stress components in a list. 
    '''
    x, y, z = X
    u, v, w = U
    #G = E/(2*(1+𝜈))                  # Use G for more concise formulas 
    e = u.diff(x) + v.diff(y) + w.diff(z)   # volume strain (disp. div.)
    𝜎xx = (2*G*(𝜈/(1-2*𝜈)*e+u.diff(x)))    
    𝜎yy = (2*G*(𝜈/(1-2*𝜈)*e+v.diff(y)))    
    𝜎zz = (2*G*(𝜈/(1-2*𝜈)*e+w.diff(z)))  
    𝜎yz = (G*(w.diff(y)+v.diff(z)))
    𝜎xz = (G*(w.diff(x)+u.diff(z)))
    𝜎xy = (G*(v.diff(x)+u.diff(y)))
    
    return [𝜎xx, 𝜎yy, 𝜎zz, 𝜎yz, 𝜎xz, 𝜎xy] 

def equilibrium_check(X, 𝜎): 
    '''Check the differential equilibrium equations for a given set 
    of stresses.'''
    x, y, z = X
    𝜎xx, 𝜎yy, 𝜎zz, 𝜎yz, 𝜎xz, 𝜎xy = 𝜎
    eqn1 = (𝜎xx.diff(x) + 𝜎xy.diff(y) + 𝜎xz.diff(z)).simplify() 
    eqn2 = (𝜎xy.diff(x) + 𝜎yy.diff(y) + 𝜎yz.diff(z)).simplify() 
    eqn3 = (𝜎xz.diff(x) + 𝜎yz.diff(y) + 𝜎zz.diff(z)).simplify() 
    return [eqn1, eqn2, eqn3]

def find_stream_f2D(Vx, Vy, X):
    '''
       Find the stream function 𝜓 for a given set of vector field
       components. The vector field must be a divergence free field. 
       return: 𝜓 - the stream function found. 
    '''
    x, y = X
    # Use condition: ∂𝜓/∂x =-Vy 
    C = sp.Function('C')(y)        # define symbolic integral functions
    𝜓C = sp.integrate(-Vy, (x)) + C     # 𝜓 with C(y); used -Vy = ∂𝜓/∂𝑥  

    # use condition: Vx = ∂𝜓/∂y
    eqn = 𝜓C.diff(y)- Vx                  # create an equation to solve 
    sln_C = sp.dsolve(eqn, C)                  # solve the eqn for C(y)

    𝜓C = 𝜓C.subs(C, sln_C.args[1])   # put solution of C(y) back to 𝜓C
    𝜓  = 𝜓C.subs(𝜓C.args[0],0).simplify() # set integral constant to 0
    printx('𝜓')

    # Check all conditions: 
    printx('(𝜓.diff(x)+Vy).simplify()')            # check condition 
    printx('(𝜓.diff(y)-Vx).simplify()')            # check condition 
    
    return 𝜓

def displacement2StrainPolar(U,R):
    '''Compute strains for given displacement U=[ur, u𝜃]^T in the
    Polar coordinates (r, 𝜃). In sympy.'''
    r, 𝜃 = R
    ur, u𝜃 = U
    εrr = ur.diff(r)
    ε𝜃𝜃 = ur/r + u𝜃.diff(𝜃)/r
    εr𝜃 = (ur.diff(𝜃)/r + u𝜃.diff(r) - u𝜃/r)/2
    return (εrr, ε𝜃𝜃, εr𝜃)

def strain2stress2D(E, 𝜈, 𝜀, Ptype= 'stress'): 
    '''2D constitutive equation, from strains to stresses for both
       plane-stress and plane-strain problems.'''
    𝜀xx, 𝜀yy, 𝜀xy = 𝜀
    if Ptype== "stress": 
        E_ = E/(1-𝜈**2) 
        𝜎xx = E_*(𝜀xx + 𝜈*𝜀yy).simplify()    
        𝜎yy = E_*(𝜀yy + 𝜈*𝜀xx).simplify()   
    else: 
        E_ = E/((1+𝜈)*(1-2*𝜈)) 
        𝜎xx = E_*((1-𝜈)*𝜀xx + 𝜈*𝜀yy).simplify()    
        𝜎yy = E_*((1-𝜈)*𝜀yy + 𝜈*𝜀xx).simplify()   

    𝜎xy = (E*𝜀xy/(1+𝜈)).simplify()   
    return (𝜎xx,𝜎yy,𝜎xy) 

def constantUpdate(eqns, r, order, cs, prnt=False):
    '''Compute and update constants `cs` for a given 1D sympy `eqns`.
    return: sln_eqn - solution of cs in dictionary.
    '''
    exprs = [] 
    for eqn in eqns:
        for i in range(order): 
            exprs.append(eqn.coeff(r,i)) 

    exprs = remove_zero_exprs(exprs)  
    if prnt: display(exprs)
    sln_eqns = sp.solve(exprs, cs)       # solve for unknown constants 
    return sln_eqns                              # can be very lengthy


def stress2strain2D(E, 𝜈, 𝜎): 
    '''2D constitutive equation, for plane stress problems.'''
    𝜎xx, 𝜎yy, 𝜎xy = 𝜎
    𝜀xx = ((𝜎xx - 𝜈*𝜎yy)/E).simplify()
    𝜀yy = ((𝜎yy - 𝜈*𝜎xx)/E).simplify()
    𝜀zz = ((- 𝜈*𝜎xx - 𝜈*𝜎yy)/E).simplify()
    𝜀xy = (1+𝜈)*𝜎xy/E 
    return [𝜀xx,𝜀yy,𝜀xy,𝜀zz]

def splitFraction(expr):
    '''Separate numerator of expr with the common denominator''' 
    numerator, denominator = expr.as_numer_denom()
    numerator_terms = numerator.as_ordered_terms()
    separated_terms = [term / denominator for term in numerator_terms]
    return sum(separated_terms)


def factorFraction(expr, fraction):
    '''factor out the fraction from expr'''  
    return sp.factor_terms(expr / fraction) * fraction

def simplify_expr(expr):
    '''simplify both the numerator and the denominator of each term 
    in a given expr. Suggested by ChatGPT. '''
    terms = expr.as_ordered_terms()
    #terms = sum(sp.simplify(term) for term in terms)
    processed_terms = []
    for term in terms:
        num, den = sp.fraction(term.simplify())
        simplified_den = sp.factor(den)
        processed_terms.append(num / simplified_den)
    return sum(processed_terms)

def getExprs(eqns, x, order):
    '''Get the expressions for a given list of 1D sympy `eqns`.
    return: exprs - expressions of the coefficients of all orders of x. 
    '''
    exprs = [] 
    for eqn in eqns:
        for i in range(order+1): 
            exprs.append(eqn.expand().coeff(x,i)) 

    exprs = remove_zero_exprs(exprs)  
    return exprs

def expand_orthogonal_basis0(initial_vectors, n):
    """
    Expand a given set of orthogonal vectors to a complete orthogonal 
    basis in n-dimensional space using Gram-Schmidt orthogonalization.
    Parameters: initial_vectors: ndarray
                A set of orthogonal vectors (r x n, where r <= n).
             n: int, the dimension of the space.
    Returns: orthogonal_basis: ndarray
             An n x n array rows of orthonormal vectors.
    Slightly modified based on ChatGPT' suggestion. 
    """
    # Ensure initial vectors are orthonormal
    initial_vectors = np.array([v/lg.norm(v) for v in initial_vectors])
    
    # Generate n-r random vectors to complete the basis
    r = initial_vectors.shape[0]
    additional_vectors = np.random.randn(n - r, n)
    
    # Combine initial and additional vectors
    all_vectors = np.vstack((initial_vectors, additional_vectors))
    
    # Perform Gram-Schmidt orthogonalization
    orthogonal_basis = np.zeros((n, n))
    for i in range(n):
        vec = all_vectors[i]
        for j in range(i):
            # Subtract projections of v onto existing orthogonalized vs
            vec -= np.dot(vec, orthogonal_basis[j])*orthogonal_basis[j]
        orthogonal_basis[i] = vec / np.linalg.norm(vec)
    
    return orthogonal_basis

def indet_eqs_solver(A):
    '''Solve indeterminate system equations, using the rref form
       (reduced-row-echolen-from).  
    Inputs: A, np.array augmented with b. 
    Return: if no-solution: 0, 0, 0. Ortherwise:
            xp, particular solution
            N (matrix of null-space vectors), nd (nullity)
    '''
    np.set_printoptions(precision=4, suppress=True)      # print digits
    nc = np.shape(A)[1]-1
    nr = np.shape(A)[0]  
    M_rref, colns = Matrix(A).rref(iszerofunc=lambda x:abs(x)<1e-12)
    M_rref = np.array(M_rref).astype(float)
    
    for i in range(nr):       # Check whether the solution exist or not
        if lg.norm(M_rref[i,:-1]) == 0 and M_rref[i,-1] != 0:
            print(f"No solution: Ax!=b, "
                  f"at {i}th row, b'={M_rref[i,-1]}")
            print(f"The rref form:{M_rref}")
            return 0, 0, 0
    xp = np.zeros(nc)
    for i in range(len(colns)):
        xp[i] = xp[i] + M_rref[i,-1]              # particular solution

    # Basis for null-space (see the textbook by Gilbert Strong)
    r = np.size(colns)
    print(f"rref of A with pivot columns:\n{M_rref}; {colns}; "\
          f'  columns of A:',nc,'\nRank=',r,', or dimension of C(A)')
    nd = nc - r
    print(f"Dimension of the null-space N(A), or nullity = {nd}")
    
    i_old = [i for i in range(nc)]           # original variables order
    iFC = [i for i in i_old if i not in set(colns)]
    i_new = list(colns) + list(iFC)           # changed variables order
    print(f"Free columns = {iFC}, i_old= {i_old}, i_new= {i_new}")
    
    # Form the basis vectors for the Null space: N=[-F  I].T
    if nd == 0: 
        N = 0; print(f"Thus, the null-space has only the zero vector")
    else: 
        N =np.array(-M_rref[:r, iFC])     # free columns for pivot vars 
        N = np.append(N, np.eye(nd), axis=0) # add I for free variables
        print(' N shape:',np.shape(N), 'I(nd).shape:', np.eye(nd).shape)
        #print(f"Basis vectors for the null-space:\n{N}")
        
        # Swap rows, if variable order is changed 
        for i in range(nc):
            if i_new[i] != i_old[i]:
                N[[i_new[i], i_old[i]], :] = N[[i_old[i], i_new[i]], :]
                xp[i_new[i]], xp[i_old[i]] = xp[i_old[i]], xp[i_new[i]]
                i_new[i_new[i]], i_new[i_old[i]] = \
                i_new[i_old[i]], i_new[i_new[i]]
                print(f"Basis vectors for null-space after swap:\n{N}")
                
        print(f"Transformed basis vector of the null-space by A:\
              \n{A[:,:-1].dot(N).T}, N is orthogonal to all rows of A.")
        #print(f"N is also orthogonal to A.rref:\n{M_rref[:,:-1]@N}")

    return xp, N, nd

