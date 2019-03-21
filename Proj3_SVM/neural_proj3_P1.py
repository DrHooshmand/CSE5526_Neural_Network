from __future__ import division
from itertools import cycle
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import random
import math
from svmutil import *






if __name__ == "__main__":
    C=[]
    acc=[]
    for i in range(-4,9):
        C.append(2**i)

    #C=[0.0625]

    for i in range(len(C)):
        y1, x1 = svm_read_problem('../../ncrna_s.train')
        y2, x2 = svm_read_problem('../../ncrna_s.test')
        m = svm_train(y1, x1, '-s 0 -c'+' '+str(C[i]))
        p_label, p_acc, p_val = svm_predict(y2, x2, m)
        ACC, MSE, SCC = evaluations(y2, p_label)
        acc.append(ACC)

plt.figure()
plt.plot(np.log2(C),acc, linewidth=3)
plt.xlabel('C parameter (log2 scale)')
plt.ylabel('Accuracy %')
plt.title('SVM:All training data performed on test data')
plt.savefig('all_train.pdf')
plt.close('all')





print('done')