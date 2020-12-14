# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 23:15:40 2020

@author: VanHa
"""

import numpy as np
import pandas as pd
import scipy.io
from scipy import sparse 
from numpy.linalg import norm
from scipy.sparse.linalg import spsolve
import time


def initial_point(a, b, c):
    n = a.shape[1]
    e = np.ones((n,))
    sA = sparse.csr_matrix(a) 
    lamda = spsolve(sA.dot(sA.T), sA.dot(c), use_umfpack=False)
    s = c.flatten() - sA.T.dot(lamda)
    
# solution for min norm(x) s.t. Ax = b
    x = a.T.dot(spsolve(sA.dot(sA.T), b, use_umfpack=False))

    # delta_x and delta_s
    delta_x = max(-1.5 * np.min(x), 0)
    delta_s = max(-1.5 * np.min(s), 0)

    # delta_x_c and delta_s_c
    pdct = 0.5 * (x + delta_x * e).dot(s + delta_s * e)
    delta_x_c = delta_x + pdct / (np.sum(s) + n * delta_s)
    delta_s_c = delta_s + pdct / (np.sum(x) + n * delta_x)

    # output
    x0 = x + delta_x_c * e
    s0 = s + delta_s_c * e
    lamda0 = lamda
    
    return x0, lamda0, s0

def step_length(x, s, dx, ds, rho=0.9995):
    alpha_x = -1 / min(min(dx / x), -1) ## to avoid the case of delta_x or delta_s doesn't have any element < 0, 
                                        # the system still get the value of 1
    alpha_x = min(1, rho * alpha_x)
    alpha_s = -1 / min(min(ds / s), -1)
    alpha_s = min(1, rho * alpha_s)
    return alpha_x, alpha_s


def newton_direction(r_b, r_c, r_x_s, a, m, n, x, s ):

    rhs = np.hstack(( -r_c , -r_b, -r_x_s ))
    rhs = rhs[:,np.newaxis]
    A = sparse.bmat( [[ None, a.T, sparse.eye(n) ],
		                  [ a  , None,   None ], 
                          [ sparse.diags(s, 0) , None,  sparse.diags(x, 0) ]]  )
    A = A.tocsc()

    delta = spsolve(A, rhs, permc_spec="MMD_ATA" )  ##Solve the sparse linear system Ax=b, 
                                                    ## with the permutation of minimum degree ordering.

    delta_x = delta[:n]
    delta_lamda = delta[n : n+m]
    delta_s = delta[m+n:]

    return delta_x, delta_lamda, delta_s



def optimization(A, B, C):
#### get number of column and row from the object functions and contraints
    m = A.shape[0] ##m rows.
    n = A.shape[1]  ##n columns ~ n variables
    eps_fea = 1.0e-8
    eps_opt = 1.0e-8
    iteration= []
    f_obj = []
    d_obj = []
    x_val = []
    mu_val = []
    check_rc = []
    check_rb = []
    
### checking A, B , C matrices
    if C.shape[1] != 1 or B.shape[1]!= 1:
        raise Exception("Error: c and b must be column vectors")
    if C.shape[0] != n or B.shape[0] != m:
        raise Exception("Error: inconsistent dimensions for c, b and A")
    if np.linalg.matrix_rank(A) != m: 
        raise Exception("Error: matrix A is not full row rank")

## Intial point   
    x, lamda, s = initial_point(A, B, C)
    # x= np.ones(n)*10
    # lamda= np.ones(m)
    # s= np.ones(n)*10
  
    mu = x.dot(s)/n
    r_b = A.dot(x) - B.flatten()
    r_c = A.T.dot(lamda)+ s - C.flatten()
    check_opt1 = norm(r_c)/(1+ norm(C))
    check_opt2 = norm(r_b)/(1+ norm(B))
    
    k = 0 
    obj_func = (C.T).dot(x)
    iteration.append(k)
    f_obj.append(np.ravel(obj_func)[0])
    d_obj.append(np.ravel(B.T.dot(lamda))[0])
    x_val.append(x)
    mu_val.append(mu)
    check_rc.append(check_opt1)
    check_rb.append(check_opt2)

    while not ((check_opt1 < eps_fea) & (check_opt2 < eps_fea) & (mu < eps_opt)):
        ###Predictor step:
        r_x_s = x*s
        dx_aff, dlamda_aff, ds_aff= newton_direction(
            r_b, r_c, r_x_s, A, m, n, x, s)

        ####### Compute step length in predictor
        alpha_aff_x, alpha_aff_s = step_length(x, s, dx_aff, ds_aff)
        mu_aff =  (x + alpha_aff_x * dx_aff).dot(s + alpha_aff_s * ds_aff) / n
        sigma = (mu_aff / mu) ** 3 
        
        ### Corrector step
        
        r_x_s_cor = dx_aff * ds_aff - sigma * mu * np.ones((n))
        
        ### Get corrector's direction
        dx_cc, dlamda_cc, ds_cc = newton_direction( 
            np.zeros((m)), np.zeros((n)) , r_x_s_cor, A, m, n, x, s)
        
        
        ### Compute search direction 
        dx = dx_aff + dx_cc
        dlamda = dlamda_aff + dlamda_cc
        ds = ds_aff + ds_cc

        ## Update step lenght after corrector:
        alpha_x, alpha_s = step_length(x, s, dx, ds)
        
        ## Update x, lamda, s
        x = x + alpha_x * dx
        lamda = lamda + alpha_s * dlamda
        s = s + alpha_s * ds
        
        ### Update right hand side and mu, and termination condition
        
        mu = x.dot(s)/n
        r_b = A.dot(x) - B.flatten()
        r_c = A.T.dot(lamda)+ s - C.flatten()
        check_opt1 = norm(r_c)/(1+ norm(C))
        check_opt2 = norm(r_b)/(1+ norm(B))
        
        obj_func = (C.T).dot(x)
        iteration.append(k+1)
        f_obj.append(np.ravel(obj_func)[0])
        d_obj.append(np.ravel(B.T.dot(lamda))[0])
        x_val.append(x)
        mu_val.append(mu)
        check_rc.append(check_opt1)
        check_rb.append(check_opt2)
        d = {'Iteration': iteration, 'Object function': f_obj, "Dual obj" :d_obj, "mu": mu_val, "rc": check_rc, "rb": check_rb }
        df = pd.DataFrame(d)
        k +=1  
        
    for i in range(1,n+1):
        print ('X%s: %01f' %(i, x[i-1]))
    print(df)
    
    return x, df
        

def step_length_w(x, s, dx, ds, rho=0.9995):
    np.seterr(divide='ignore', invalid='ignore')
    alpha_x = -1 / min(min(dx / x), -1) ## to avoid the case of delta_x or delta_s doesn't have any element < 0, 
                                        # the system still get the value of 1
    alpha_x = min(1, rho * alpha_x)
    alpha_s = -1 / min(min(ds / s), -1)
    alpha_s = min(1, rho * alpha_s)
    return alpha_x, alpha_s


def optimization_without(A, B, C):
#### get number of column and row from the object functions and contraints
    m = A.shape[0] ##m rows.
    n = A.shape[1]  ##n columns ~ n variables
    eps_fea = 1.0e-8
    eps_opt = 1.0e-8
    iteration= []
    f_obj = []
    d_obj = []
    x_val = []
    mu_val = []
    check_rc = []
    check_rb = []
    
### checking A, B , C matrices
    if C.shape[1] != 1 or B.shape[1]!= 1:
        raise Exception("Error: c and b must be column vectors")
    if C.shape[0] != n or B.shape[0] != m:
        raise Exception("Error: inconsistent dimensions for c, b and A")
    if np.linalg.matrix_rank(A) != m: 
        raise Exception("Error: matrix A is not full row rank")

## Intial point   
    x, lamda, s = initial_point(A, B, C)
    # x= np.ones(n)*10
    # lamda= np.ones(m)
    # s= np.ones(n)*10
  
    mu = x.dot(s)/n
    sigma = 0.1
    r_b = A.dot(x) - B.flatten()
    r_c = A.T.dot(lamda)+ s - C.flatten()
    r_x_s = x*s - sigma * mu * np.ones((n))
    check_opt1 = norm(r_c)/(1+ norm(C))
    check_opt2 = norm(r_b)/(1+ norm(B))
    
    k = 0 
    obj_func = (C.T).dot(x)
    iteration.append(k)
    f_obj.append(np.ravel(obj_func)[0])
    d_obj.append(np.ravel(B.T.dot(lamda))[0])
    x_val.append(x)
    mu_val.append(mu)
    check_rc.append(check_opt1)
    check_rb.append(check_opt2)
    
   
    while not ((check_opt1 < eps_fea) & (check_opt2 < eps_fea) & (mu < eps_opt)):
        ###Predictor step:
        dx, dlamda, ds = newton_direction(
            r_b, r_c, r_x_s, A, m, n, x, s)

     
        ## Update step lenght after corrector:
        alpha_x, alpha_s = step_length_w(x, s, dx, ds)
        
        ## Update x, lamda, s
        x = x + alpha_x * dx
        lamda = lamda + alpha_s * dlamda
        s = s + alpha_s * ds
        
        
        mu = x.dot(s)/n
        r_b = A.dot(x) - B.flatten()
        r_c = A.T.dot(lamda)+ s - C.flatten()
        r_x_s = x*s - sigma * mu * np.ones((n))
        check_opt1 = norm(r_c)/(1+ norm(C))
        check_opt2 = norm(r_b)/(1+ norm(B))
        
        obj_func = (C.T).dot(x)
        iteration.append(k+1)
        f_obj.append(np.ravel(obj_func)[0])
        d_obj.append(np.ravel(B.T.dot(lamda))[0])
        x_val.append(x)
        mu_val.append(mu)
        check_rc.append(check_opt1)
        check_rb.append(check_opt2)
        d = {'Iteration': iteration, 'Object function': f_obj, "Dual obj" :d_obj, "mu": mu_val, "rc": check_rc, "rb": check_rb }
        df = pd.DataFrame(d)
        k +=1  
        
    for i in range(1,n+1):
        print ('X%s: %01f' %(i, x[i-1]))
    print(df)
    
    return x, df




def mat_parse(file):
    mat_content = scipy.io.loadmat(file)
    mat_struct = mat_content['Problem']
    val = mat_struct[0, 0]
    A = sparse.csr_matrix(val['A']).todense()
    A = np.asarray(A)
    b = val['b']
    c = val['aux'][0][0][0]
    lb = val['aux'][0][0][1]
    ub = val['aux'][0][0][2]
    if (np.all(lb !=0)):
        raise Exception("Transform A, b, c before inputing")
    if (np.all(np.isinf(ub))==False):
        ##change A
        ub_stack = ub
        ub_stack[np.isinf(ub_stack) == False] = 1
        ub_stack[np.isinf(ub_stack) == True] = 0.0
        nnx =  np.diag(np.ravel(ub_stack))
        A_new = np.hstack((A,np.zeros((A.shape[0],A.shape[1])))) 
        A_stack = np.hstack((nnx, nnx))
        A_final = np.vstack((A_new,A_stack))
        
        ##change B
        ub_new = ub
        ub_new[np.isinf(ub_new) == True] = 0.0
        b_new = np.vstack((b,ub_new))
        
        ##change C
        c_new = np.vstack((c,np.zeros((A.shape[1],1))))
        
        ##update A, b, c
        A = A_final
        b= b_new
        c= c_new        
    return A, b, c

##problem 1

q,v,t = mat_parse("lp_share1b.mat")

t1 = time.perf_counter()
prob, df = optimization(q, v, t)
elapsed = (time.perf_counter() - t1)
print(" Time to excuse")
print(elapsed)


t2 = time.perf_counter()

prob, df = optimization_without(q, v, t)
elapsed = (time.perf_counter() - t2)
print(" Time to excuse")
print(elapsed)


##problem 2: 


a= np.array([[4,-2,1, 0, 0], [3,4,0, -1, 0],[1,1, 0, 0,1]])

c= np.array([-3,-2, 0,0, 0])
c= c[:, np.newaxis]

b= np.array([5,1, 2])
b= b[:, np.newaxis]

t1 = time.perf_counter()        
prob, df = optimization(a, b, c)
elapsed = (time.perf_counter() - t1)
print(" Time to excuse")
print(elapsed)


t2 = time.perf_counter()        
prob, df = optimization_without(a, b, c)
elapsed = (time.perf_counter() - t2)
print(" Time to excuse")
print(elapsed)