import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import myLDA as ml
from sklearn.linear_model import LinearRegression
import datagen as dt
import accuracy as acc
import matplotlib.pyplot as plt
import random as rd
import pandas

nb_tests=10
Imputation_technics=['g_mean','conditional_mean', 'nearest_neigbours','regression']
used_trs=[50,100,250,500,1000]

def one_graph(prob_missingness):
    p=prob_missingness
    without_imp=np.zeros(5)
    grand_mean=np.zeros(5)
    conditional_mean=np.zeros(5)
    closest=np.zeros(5)
    regression=np.zeros(5)
    without_removing=np.zeros(5)
    multiple_closest=np.zeros(5)
    multiple_regression=np.zeros(5)

    training_set_size=['50','100','250','500','1000']
    dini=(pandas.read_csv('data_banknote_authentication.csv').to_numpy())
    np.random.shuffle(dini)
    X1i=dini[0:1000,0:4]
    X2i=dini[1000:,0:4]
    Y1i=dini[0:1000,4]
    Y2i=dini[1000:,4]

    for i in range(nb_tests):
        df = (pandas.read_csv('data_banknote_authentication.csv').to_numpy())
        np.random.shuffle(df)
        for i in range(1371):
            for j in range(4):
                if ((np.random.uniform())<p*2) and (df[i,4]==1.0):
                    df[i,j]=0
        fullX1=df[0:1000,0:4]
        fullY1=df[0:1000,4]
        X2=df[1000:,0:4]
        Y2=df[1000:,4]
        for j in range(5):
            without_removing[j]+=acc.acc(X1i[0:used_trs[j]][:],Y1i[0:used_trs[j]],X2i,Y2i,'no_imputation')
            without_imp[j]+=acc.acc(fullX1[0:used_trs[j],:],fullY1[0:used_trs[j]],X2,Y2,'no_imputation')
            grand_mean[j]+=acc.acc(fullX1[0:used_trs[j],:],fullY1[0:used_trs[j]],X2,Y2,'grand_mean')
            conditional_mean[j]+=acc.acc(fullX1[0:used_trs[j],:],fullY1[0:used_trs[j]],X2,Y2,'conditional_mean')
            closest[j]+=acc.acc(fullX1[0:used_trs[j],:],fullY1[0:used_trs[j]],X2,Y2,'closest')
            regression[j]+=acc.acc(fullX1[0:used_trs[j],:],fullY1[0:used_trs[j]],X2,Y2,'regression')
            multiple_closest[j]+=acc.acc(fullX1[0:used_trs[j],:],fullY1[0:used_trs[j]],X2,Y2,'multiple_closest')
            multiple_regression[j]+=acc.acc(fullX1[0:used_trs[j],:],fullY1[0:used_trs[j]],X2,Y2,'multiple_regression')
    without_removing/=nb_tests
    without_imp/=nb_tests
    grand_mean/=nb_tests
    conditional_mean/=nb_tests
    closest/=nb_tests
    regression/=nb_tests
    multiple_closest/=nb_tests
    multiple_regression/=nb_tests

    plt.figure(figsize=(10,5))
    plt.plot(training_set_size,without_removing, color = 'purple', linewidth = 2)
    plt.plot(training_set_size,without_imp, color = 'green', linewidth = 2)
    plt.plot(training_set_size,grand_mean, color = 'blue', linewidth = 2)
    plt.plot(training_set_size,conditional_mean, color = 'orange', linewidth = 2)
    plt.plot(training_set_size,closest, color = 'red', linewidth = 2)
    plt.plot(training_set_size,regression, color = 'black', linewidth = 2)
    plt.plot(training_set_size,multiple_closest, color = 'pink', linewidth = 2)
    plt.plot(training_set_size,multiple_regression, color = 'grey', linewidth = 2)

    mylabels = ['without_removing','No imputation', 'Grand Mean', 'Conditional Mean','Closest neighbour','Regression','multiple_closest','multiple_regression']
    #,'multiple_closest','multiple_regression'

    plt.title('  Real_life_data MNAR '+ '  Prob_missingness: '+ str(p))
    plt.legend(labels = mylabels)
    plt.ylabel('Accuracy', fontsize = 10)
    plt.xlabel('Training size', fontsize = 10)
    plt.show()
    #plt.savefig('Covariance: '+ ncov+ '  Dim: '+ str(dim) +'  Type missingness: ' +type_missingness+ '  Prob_missingness: '+ str(p)+'.jpg')


''' cov_matrices : normal,str_correlation_higherIndex,str_correlation+high_diagonal '''
'''type of missingness : MCAR, MAR, MNAR'''


one_graph(0.2)
