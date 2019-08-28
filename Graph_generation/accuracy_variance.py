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
    without_imp=np.zeros((6,nb_tests))
    grand_mean=np.zeros((6,nb_tests))
    conditional_mean=np.zeros((6,nb_tests))
    closest=np.zeros((6,nb_tests))
    regression=np.zeros((6,nb_tests))
    without_removing=np.zeros((6,nb_tests))
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
            without_removing[j][i]+=acc.acc(X1i[0:used_trs[j]][:],fullY1[0:used_trs[j]],X2i,Y2,'no_imputation')
            without_imp[j][i]+=acc.acc(fullX1[0:used_trs[j],:],fullY1[0:used_trs[j]],X2,Y2,'no_imputation')
            grand_mean[j][i]+=acc.acc(fullX1[0:used_trs[j],:],fullY1[0:used_trs[j]],X2,Y2,'grand_mean')
            conditional_mean[j][i]+=acc.acc(fullX1[0:used_trs[j],:],fullY1[0:used_trs[j]],X2,Y2,'conditional_mean')
            closest[j][i]+=acc.acc(fullX1[0:used_trs[j],:],fullY1[0:used_trs[j]],X2,Y2,'closest')
            regression[j][i]+=acc.acc(fullX1[0:used_trs[j],:],fullY1[0:used_trs[j]],X2,Y2,'regression')
    v_without_removing=np.var(without_removing,axis=1)
    v_without_imp=np.var(without_imp,axis=1)
    v_grand_mean=np.var(grand_mean,axis=1)
    v_conditional_mean=np.var(conditional_mean,axis=1)
    v_closest=np.var(closest,axis=1)
    v_regression=np.var(regression,axis=1)

    plt.figure(figsize=(10,5))
    plt.plot(training_set_size,v_without_removing, color = 'purple', linewidth = 2)
    plt.plot(training_set_size,v_without_imp, color = 'green', linewidth = 2)
    plt.plot(training_set_size,v_grand_mean, color = 'blue', linewidth = 2)
    plt.plot(training_set_size,v_conditional_mean, color = 'orange', linewidth = 2)
    plt.plot(training_set_size,v_closest, color = 'red', linewidth = 2)
    plt.plot(training_set_size,v_regression, color = 'black', linewidth = 2)

    mylabels = ['without_removing','No imputation', 'Grand Mean', 'Conditional Mean','Closest neighbour','Regression']

    plt.title('Accuracy Variance'+'Covariance: '+ ncov+ '  Dim: '+ str(dim) +'  Type missingness: ' +type_missingness+ '  Prob_missingness: '+ str(p))
    plt.legend(labels = mylabels)
    plt.ylabel('Accuracy', fontsize = 10)
    plt.xlabel('Training size', fontsize = 10)
    plt.savefig('Variance '+'Covariance: '+ ncov+ '  Dim: '+ str(dim) +'  Type missingness: ' +type_missingness+ '  Prob_missingness: '+ str(p)+'.jpg')
    plt.show()


''' cov_matrices : normal,str_correlation_higherIndex,str_correlation+high_diagonal '''
'''type of missingness : MCAR, MAR, MNAR'''


one_graph('random',5,'MCAR',0.2)
