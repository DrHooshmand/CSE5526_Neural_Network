from __future__ import division
from itertools import cycle
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import random
import math
from svmutil import *


def whole_train(x_tr,y_tr,x_v,y_v,c,alpha):
    m = svm_train(y_tr, x_tr, '-s 0 -t 2 -c' + ' ' + str(c) + ' -g ' + str(alpha))
    p_label, p_acc, p_val = svm_predict(y_v, x_v, m)
    ACC, MSE, SCC = evaluations(y_v, p_label)
    print('The accuracy of training on the optimum C and Alpha is: %s' %ACC)
    return ACC





if __name__ == "__main__":

    Y, X = svm_read_problem('../../ncrna_s.train')
    IND=random.sample(range(0,len(X)-1) , int(len(X)/2) )

    ind1=random.sample(IND , 200)
    ind=[item for item in IND if item not in ind1]

    ind2 = random.sample(ind, 200)
    ind=[item for item in ind if item not in ind2]

    ind3 = random.sample(ind, 200)
    ind=[item for item in ind if item not in ind3]

    ind4 = random.sample(ind, 200)
    ind=[item for item in ind if item not in ind4]

    ind5 = random.sample(ind, 200)

    jj=ind1+ind2+ind3+ind4+ind5

    ind_rand=[ind1,ind2,ind3,ind4,ind5]

    xx=[]
    yy=[]
    for i in range(len(ind_rand)):
        xx.append( [ X[j] for j in ind_rand[i] ] )
        yy.append( [ Y[j] for j in ind_rand[i] ] )



    C=[]
    for i in range(-4,9):
        C.append(2**i)

    Alpha=C

    # C=[2**(-4),2**6]
    # Alpha=[2**6 , 2**9]

    result=[]
    for c in C:
        for alpha in Alpha:

            t=[]
            for n in range(5):
                x_v=xx[n]
                y_v=yy[n]

                x_t=[]
                y_t=[]
                for j in range(4):
                    if j != n:
                        x_t += xx[j]
                        y_t += yy[j]
                m = svm_train(y_t, x_t, '-s 0 -t 2 -c' + ' ' + str(c) +' -g '+ str(alpha) )
                p_label, p_acc, p_val = svm_predict(y_v, x_v, m)
                ACC, MSE, SCC = evaluations(y_v, p_label)
                t.append(ACC)
            result.append( [(c,alpha) , sum(t)/len(t) ] )


    k=[k for [(i,j),k] in result ]
    max_k=max(k)
    c_opt ,alpha_opt = [ (i,j) for [(i,j),k] in result if k==max_k ][0]

    print("Optimum C= %s and Optimum alpha = %s"  %(c_opt,alpha_opt))

    for i in range(len(Alpha)):
        for j in range(len(C)):
            print(k[i*(len(C))+j], end=' ')
        print()

    y_test, x_test = svm_read_problem('../../ncrna_s.test')
    acc_final=whole_train(X,Y,x_test,y_test,c_opt,alpha_opt)





print('done')