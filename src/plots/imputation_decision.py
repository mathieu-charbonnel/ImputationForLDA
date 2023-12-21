import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import myLDA as ml
from sklearn.linear_model import LinearRegression
import datagen as dt
import accuracy as acc
import matplotlib.pyplot as plt
import random as rd

nb_tests=10
Imputation_technics=['g_mean','conditional_mean', 'nearest_neigbours','regression']
used_trs=[50,100,250,500,1000,2000]


'''d=dt.full_gen('normal',3,50,500,'MNAR',0.90)
X1=d[0]
Y1=d[1]
X2=d[2]
Y2=d[3]
print(acc.acc(X1,Y1,X2,Y2,'grand_mean'))'''

def one_graph(ncov,dim,type_missingness,prob_missingness):
    p=prob_missingness
    without_imp=np.zeros(6)
    grand_mean=np.zeros(6)
    conditional_mean=np.zeros(6)
    closest=np.zeros(6)
    regression=np.zeros(6)
    without_removing=np.zeros(6)
    optimal=np.zeros(6)
    training_set_size=['50','100','250','500','1000','2000']
    for i in range(nb_tests):
        data=dt.full_gen(ncov,dim,2000,1000,type_missingness,prob_missingness)
        fullX1=data[0]
        fullY1=data[1]
        X2=data[2]
        Y2=data[3]
        X1i=data[4]
        X2i=data[5]
        for j in range(6):
            op=0
            r1=acc.acc(fullX1[0:used_trs[j],:],fullY1[0:used_trs[j]],X2,Y2,'no_imputation')
            r2=acc.acc(fullX1[0:used_trs[j],:],fullY1[0:used_trs[j]],X2,Y2,'grand_mean')
            r3=acc.acc(fullX1[0:used_trs[j],:],fullY1[0:used_trs[j]],X2,Y2,'conditional_mean')
            r4=acc.acc(fullX1[0:used_trs[j],:],fullY1[0:used_trs[j]],X2,Y2,'closest')
            r5=acc.acc(fullX1[0:used_trs[j],:],fullY1[0:used_trs[j]],X2,Y2,'regression')

            without_removing[j]+=acc.acc(X1i[0:used_trs[j]][:],fullY1[0:used_trs[j]],X2i,Y2,'no_imputation')
            without_imp[j]+=r1
            grand_mean[j]+=r2
            conditional_mean[j]+=r3
            closest[j]+=r4
            regression[j]+=r5

            if acc.acc(fullX1[0:used_trs[j]][:],fullY1[0:used_trs[j]],fullX1[0:used_trs[j]][:],fullY1[0:used_trs[j]],'no_imputation')>op:
                op=r1
            if acc.acc(fullX1[0:used_trs[j]][:],fullY1[0:used_trs[j]],fullX1[0:used_trs[j]][:],fullY1[0:used_trs[j]],'grand_mean')>op:
                op=r2
            if acc.acc(fullX1[0:used_trs[j]][:],fullY1[0:used_trs[j]],fullX1[0:used_trs[j]][:],fullY1[0:used_trs[j]],'conditional_mean')>op:
                op=r3
            if acc.acc(fullX1[0:used_trs[j]][:],fullY1[0:used_trs[j]],fullX1[0:used_trs[j]][:],fullY1[0:used_trs[j]],'closest')>op:
                op=r4
            if acc.acc(fullX1[0:used_trs[j]][:],fullY1[0:used_trs[j]],fullX1[0:used_trs[j]][:],fullY1[0:used_trs[j]],'regression')>op:
                op=r5
            optimal[j]+=op

    without_removing/=nb_tests
    without_imp/=nb_tests
    grand_mean/=nb_tests
    conditional_mean/=nb_tests
    closest/=nb_tests
    regression/=nb_tests
    optimal/=nb_tests

    plt.figure(figsize=(10,5))
    plt.plot(training_set_size,optimal, color = 'yellow', linewidth = 2)
    plt.plot(training_set_size,without_removing, color = 'purple', linewidth = 2)
    plt.plot(training_set_size,without_imp, color = 'green', linewidth = 2)
    plt.plot(training_set_size,grand_mean, color = 'blue', linewidth = 2)
    plt.plot(training_set_size,conditional_mean, color = 'orange', linewidth = 2)
    plt.plot(training_set_size,closest, color = 'red', linewidth = 2)
    plt.plot(training_set_size,regression, color = 'black', linewidth = 2)

    mylabels = ['optimal','without_removing','No imputation', 'Grand Mean', 'Conditional Mean','Closest neighbour','Regression']

    plt.title('Covariance: '+ ncov+ '  Dim: '+ str(dim) +'  Type missingness: ' +type_missingness+ '  Prob_missingness: '+ str(p))
    plt.legend(labels = mylabels)
    plt.ylabel('Accuracy', fontsize = 10)
    plt.xlabel('Training size', fontsize = 10)
    plt.savefig('opt'+'Covariance: '+ ncov+ '  Dim: '+ str(dim) +'  Type missingness: ' +type_missingness+ '  Prob_missingness: '+ str(p)+'.jpg')
    plt.show()


''' cov_matrices : normal,str_correlation_higherIndex,str_correlation+high_diagonal '''
'''type of missingness : MCAR, MAR, MNAR'''


one_graph('random',5,'MNAR',0.2)
