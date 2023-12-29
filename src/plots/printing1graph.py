import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from ..models import myLDA as ml
from sklearn.linear_model import LinearRegression
from ..loading import datagen as dt
from ..metrics import accuracy as acc
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
    #multiple_closest=np.zeros(6)
    #multiple_regression=np.zeros(6)

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
            without_removing[j]+=acc.acc(X1i[0:used_trs[j]][:],fullY1[0:used_trs[j]],X2i,Y2,'no_imputation')
            without_imp[j]+=acc.acc(fullX1[0:used_trs[j],:],fullY1[0:used_trs[j]],X2,Y2,'no_imputation')
            grand_mean[j]+=acc.acc(fullX1[0:used_trs[j],:],fullY1[0:used_trs[j]],X2,Y2,'grand_mean')
            conditional_mean[j]+=acc.acc(fullX1[0:used_trs[j],:],fullY1[0:used_trs[j]],X2,Y2,'conditional_mean')
            closest[j]+=acc.acc(fullX1[0:used_trs[j],:],fullY1[0:used_trs[j]],X2,Y2,'closest')
            regression[j]+=acc.acc(fullX1[0:used_trs[j],:],fullY1[0:used_trs[j]],X2,Y2,'regression')
            #multiple_closest[j]+=acc.acc(fullX1[0:used_trs[j],:],fullY1[0:used_trs[j]],X2,Y2,'multiple_closest')
            #multiple_regression[j]+=acc.acc(fullX1[0:used_trs[j],:],fullY1[0:used_trs[j]],X2,Y2,'multiple_regression')
    without_removing/=nb_tests
    without_imp/=nb_tests
    grand_mean/=nb_tests
    conditional_mean/=nb_tests
    closest/=nb_tests
    regression/=nb_tests
    #multiple_closest/=nb_tests
    #multiple_regression/=nb_tests

    plt.figure(figsize=(10,5))
    plt.plot(training_set_size,without_removing, color = 'purple', linewidth = 2)
    plt.plot(training_set_size,without_imp, color = 'green', linewidth = 2)
    plt.plot(training_set_size,grand_mean, color = 'blue', linewidth = 2)
    plt.plot(training_set_size,conditional_mean, color = 'orange', linewidth = 2)
    plt.plot(training_set_size,closest, color = 'red', linewidth = 2)
    plt.plot(training_set_size,regression, color = 'black', linewidth = 2)
    #plt.plot(training_set_size,multiple_closest, color = 'pink', linewidth = 2)
    #plt.plot(training_set_size,multiple_regression, color = 'grey', linewidth = 2)

    mylabels = ['without_removing','No imputation', 'Grand Mean', 'Conditional Mean','Closest neighbour','Regression']
    #,'multiple_closest','multiple_regression'

    plt.title('Covariance: '+ ncov+ '  Dim: '+ str(dim) +'  Type missingness: ' +type_missingness+ '  Prob_missingness: '+ str(p))
    plt.legend(labels = mylabels)
    plt.ylabel('Accuracy', fontsize = 10)
    plt.xlabel('Training size', fontsize = 10)
    plt.axis([ 0,6,0.7,0.95])
    plt.show()
    #plt.savefig('Covariance: '+ ncov+ '  Dim: '+ str(dim) +'  Type missingness: ' +type_missingness+ '  Prob_missingness: '+ str(p)+'.jpg')

