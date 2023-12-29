import numpy as np
from sklearn.datasets import make_spd_matrix

ratio=0.5

def data_generation(ncov,dim,len_training,len_testing):
    E0=2*np.random.rand(dim)-0.5*np.ones(dim)
    E1=2*np.random.rand(dim)-0.5*np.ones(dim)
    cov=np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            if (i==j):
                cov[i][j]=1
            else:
                cov[i][j]=np.exp(-np.abs(i-j))
    if (ncov =='str_correlation_higherIndex'):
        for i in range(dim):
            for j in range(dim):
                cov[i][j]=cov[i][j]*max(i,j)
    if (ncov=='str_correlation+high_diagonal'):
        for i in range(dim):
            cov[i][i]*=5
    if (ncov=='random'):
        cov=make_spd_matrix(dim, random_state=None)
    # print(cov)
    #Training generation
    X_training=[]
    Y_training=[]
    for i in range(len_training):
        r=(np.random.uniform())
        if (r<ratio):
            X_training.append(np.random.multivariate_normal(E0,cov))
            Y_training.append(0)
        else:
            X_training.append(np.random.multivariate_normal(E1,cov))
            Y_training.append(1)
    #Testing generation
    X_testing=[]
    Y_testing=[]
    for i in range(len_testing):
        r=(np.random.uniform())
        if (r<ratio):
            X_testing.append(np.random.multivariate_normal(E0,cov))
            Y_testing.append(0)
        else:
            X_testing.append(np.random.multivariate_normal(E1,cov))
            Y_testing.append(1)
    return([X_training,Y_training,X_testing,Y_testing])


def removing_data(X1,X1class,X2,X2class, type_missingness, prob_missingness):
    p=prob_missingness
    dim=len(X1[0])
    len_train=len(X1)
    len_test=len(X2)
    Xtr=np.copy(X1)
    Xte=np.copy(X2)

    if (type_missingness=='MCAR'):
        for i in range(len_train):
            for j in range(1,dim):
                if ((np.random.uniform())<p):
                    Xtr[i][j]=0
        for i in range(len_test):
            for j in range(1,dim):
                if ((np.random.uniform())<p):
                    Xte[i][j]=0

    if (type_missingness=='MAR'):
        for j in range(1,dim):
            target_ratio=p
            ratiom=1
            limit=0
            while (ratiom>target_ratio):
                limit+=0.05
                ratiom=0
                for i in range(len_train):
                    if (np.abs(Xtr[i][j-1])>limit):
                        ratiom+=1/len_train

            for i in range(len_train):
                if (np.abs(Xtr[i][j-1])>limit):
                    Xtr[i][j]=0
            for i in range(len_test):
                if (np.abs(Xte[i][j-1])>limit):
                    Xte[i][j]=0

    if (type_missingness=='MNAR'):
        for i in range(len_train):
            if (X1class[i]==0):
                for j in range(1,dim):
                    if ((np.random.uniform())<p*(1/ratio)):
                        Xtr[i][j]=0
        for i in range(len_test):
            if (X2class[i]==0):
                for j in range(1,dim):
                    if ((np.random.uniform())<p*(1/ratio)):
                        Xte[i][j]=0

    return ([Xtr,X1class,Xte,X2class,X1,X2])

def full_gen(ncov,dim,len_training,len_testing,type,prob):
    l= data_generation(ncov,dim,len_training,len_testing)
    d= removing_data(l[0],l[1],l[2],l[3],type,prob)
    return (d)
