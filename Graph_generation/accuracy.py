import numpy as np
from sklearn.datasets import make_spd_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import myLDA as ml
from sklearn.linear_model import LinearRegression
import random as rd

nb_multiple_imputation=10

def distance(X1,X2,var):
    sum=0
    for i in range(len(X1)):
        if (X1[i]==0 or X2[i]==0):
            sum+=1
        else:
            sum+=((X1[i]-X2[i])**2)/(2*var[i])
    sum=np.sqrt(sum)
    return(sum)

def maj_vote(l):
    dim=len(l[0])
    T=np.zeros(dim)
    for i in range(dim):
        for j in range(len(l)):
            T[i]+=l[j][i]
        if (T[i]>len(l)/2):
            T[i]=1
        else:
            T[i]=0
    return(T)

def acc(X1,Y1,X2,Y2,imputation_method):
    dim=len(X1[0])
    len_training=len(X1)
    len_testing=len(X2)

    if (imputation_method=='multiple_closest'):
        mean=np.zeros(dim)
        for j in range(dim):
            n=0
            for i in range(len_training):
                val=X1[i][j]
                if (val!=0):
                    n+=1
                    mean[j]+=val
            mean[j]/=n

        #variance

        var=np.zeros(dim)
        for j in range(dim):
            n=0
            for i in range(len_training):
                val=X1[i][j]
                if (val!=0):
                    n+=1
                    var[j]+=(val-mean[j])**2
            var[j]/=n


        #Actual Imputation
        Xaf=np.copy(X1)
        Xal=np.copy(X1)
        for i in range(len_training):
            for j in range(dim):
                if (Xaf[i][j]==0):
                    min_dis1=dim+1
                    min_dis2=dim+1
                    closest_index1=0
                    closest_index2=0
                    for a in range (100):
                        int=rd.randint(0, len_training-1)
                        if (Xaf[int][j]!=0) and Y1[int]== Y1[i]:
                            dis=distance(Xaf[i],Xaf[int],var)
                            if (dis<min_dis1):
                                closest_index1=int
                                min_dis1=dis
                            elif (dis<min_dis2):
                                closest_index2=int
                                min_dis2=dis
                    Xaf[i][j]=Xaf[closest_index1][j]
                    Xal[i][j]=Xal[closest_index2][j]

        Xbf=np.copy(X2)
        Xbl=np.copy(X2)
        for i in range(len_testing):
            for j in range(dim):
                if (Xbf[i][j]==0):
                    min_dis1=dim+1
                    min_dis2=dim+1
                    closest_index1=0
                    closest_index2=0
                    for a in range (100):
                        int=rd.randint(0, len_testing-1)
                        if (Xbf[int][j]!=0):
                            dis=distance(Xbf[i],Xbf[int],var)
                            if (dis<min_dis1):
                                closest_index1=int
                                min_dis1=dis
                            elif (dis<min_dis2):
                                closest_index2=int
                                min_dis2=dis
                    Xbf[i][j]=Xbf[closest_index1][j]
                    Xbl[i][j]=Xbl[closest_index2][j]

        predictions=[]

        for i in range(nb_multiple_imputation+1):
            Xa=(i*Xaf+((nb_multiple_imputation)-i)*Xal)/nb_multiple_imputation
            Xb=(i*Xbf+((nb_multiple_imputation)-i)*Xbl)/nb_multiple_imputation
            lda = LinearDiscriminantAnalysis()
            lda.fit(Xa, Y1)
            predictions.append(lda.predict(Xb))

        sol=maj_vote(predictions)

        from sklearn.metrics import accuracy_score
        return accuracy_score(Y2, sol)




    if (imputation_method=='no_imputation'):
        lda = LinearDiscriminantAnalysis()
        lda.fit(X1, Y1)
        return(lda.score(X2,Y2))

    if (imputation_method=='grand_mean'):
        N=0
        Mean=np.zeros(dim)
        for i in range(len_training):
            for j in range(dim):
                val=X1[i][j]
                if (val!=0):
                    N+=1
                    Mean[j]+=val
        Mean/=N

        #Imputing
        Xa=np.copy(X1)
        for i in range(len_training):
            for j in range(dim):
                if Xa[i][j]==0:
                    Xa[i][j]=Mean[j]

        Xb=np.copy(X2)
        for i in range(len_testing):
            for j in range(dim):
                if Xb[i][j]==0:
                    Xb[i][j]=Mean[j]

        lda = LinearDiscriminantAnalysis()
        lda.fit(Xa, Y1)
        return(lda.score(Xb,Y2))

    if (imputation_method=='conditional_mean'):
        N0=0
        Mean0=np.zeros(dim)
        N1=0
        Mean1=np.zeros(dim)
        for i in range(len_training):
            if (Y1[i]==0):
                for j in range(dim):
                    val=X1[i][j]
                    if (val!=0):
                        N0+=1
                        Mean0[j]+=val
            else :
                for j in range(dim):
                    val=X1[i][j]
                    if (val!=0):
                        N1+=1
                        Mean1[j]+=val
        for j in range(dim):
            Mean0[j]=Mean0[j]/N0
        for j in range(dim):
            Mean1[j]=Mean1[j]/N1

        #Imputing the training set
        Xa=np.copy(X1)
        for i in range(len_training):
            for j in range(dim):
                if Xa[i][j]==0:
                    if Y1[i]==0:
                        Xa[i][j]=Mean0[j]
                    if Y1[i]==1:
                        Xa[i][j]=Mean1[j]

        #Imputing the testing sets

        Xb1=np.copy(X2)
        for i in range(len_testing):
            for j in range(dim):
                if Xb1[i][j]==0:
                    Xb1[i][j]=Mean0[j]
        Xb2=np.copy(X2)
        for i in range(len_testing):
            for j in range(dim):
                if Xb2[i][j]==0:
                    Xb2[i][j]=Mean1[j]

        lda = ml.LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                      solver='eigen', store_covariance=False, tol=0.0001)
        lda.fit(Xa, Y1)
        return(lda.score(Xb1,Xb2,Y2))

    if (imputation_method=='closest'):
        #Imputation of Training set
        #mean
        mean=np.zeros(dim)
        for j in range(dim):
            n=0
            for i in range(len_training):
                val=X1[i][j]
                if (val!=0):
                    n+=1
                    mean[j]+=val
            mean[j]/=n

        #variance

        var=np.zeros(dim)
        for j in range(dim):
            n=0
            for i in range(len_training):
                val=X1[i][j]
                if (val!=0):
                    n+=1
                    var[j]+=(val-mean[j])**2
            var[j]/=n


        #Actual Imputation
        Xa=np.copy(X1)
        for i in range(len_training):
            for j in range(dim):
                if (Xa[i][j]==0):
                    min_dis=dim+1
                    closest_index=0
                    for a in range (100):
                        int=rd.randint(0, len_training-1)
                        if (Xa[int][j]!=0) and Y1[int]== Y1[i]:
                            dis=distance(Xa[i],Xa[int],var)
                            if (dis<min_dis):
                                closest_index=int
                                min_dis=dis
                    Xa[i][j]=Xa[closest_index][j]

        Xb=np.copy(X2)
        for i in range(len_testing):
            for j in range(dim):
                if (Xb[i][j]==0):
                    min_dis=dim+1
                    closest_index=0
                    for a in range (100):
                        int=rd.randint(0, len_testing-1)
                        if (Xb[int][j]!=0):
                            dis=distance(Xb[i],Xb[int],var)
                            if (dis<min_dis):
                                closest_index=int
                                min_dis=dis
                    Xb[i][j]=Xb[closest_index][j]


        lda = LinearDiscriminantAnalysis()
        lda.fit(Xa, Y1)
        return(lda.score(Xb,Y2))

    if (imputation_method=='regression'):
        Xtraindet=np.copy(X1)
        Xtestdet=np.copy(X2)
        for j in range(1,dim):
            Xs=[]
            Xn=[]
            for h in range(len_training):
                if (Xtraindet[h][j]!=0):
                    Xs.append(Xtraindet[h][0:j])
                    Xn.append(Xtraindet[h][j])
            reg = LinearRegression().fit(Xs, Xn)

            for h in range(len_training):
                if (Xtraindet[h][j]==0):
                    Xtraindet[h][j]=(reg.predict([Xtraindet[h][0:j]]))[0]
            for h in range(len_testing):
                if (Xtestdet[h][j]==0):
                    Xtestdet[h][j]=(reg.predict([Xtestdet[h][0:j]]))[0]

        lda = LinearDiscriminantAnalysis()
        lda.fit(Xtraindet, Y1)
        return(lda.score(Xtestdet, Y2))

    if (imputation_method=='multiple_regression'):
        mean=np.zeros(dim)
        for j in range(dim):
            n=0
            for i in range(len_training):
                val=X1[i][j]
                if (val!=0):
                    n+=1
                    mean[j]+=val
            mean[j]/=n

        #variance

        var=np.zeros(dim)
        for j in range(dim):
            n=0
            for i in range(len_training):
                val=X1[i][j]
                if (val!=0):
                    n+=1
                    var[j]+=(val-mean[j])**2
            var[j]/=n

        results=[]
        for i in range(nb_multiple_imputation):
            Xtraindet=np.copy(X1)
            Xtestdet=np.copy(X2)
            for j in range(1,dim):
                Xs=[]
                Xn=[]
                for h in range(len_training):
                    if (Xtraindet[h][j]!=0):
                        Xs.append(Xtraindet[h][0:j])
                        Xn.append(Xtraindet[h][j])
                reg = LinearRegression().fit(Xs, Xn)

                for h in range(len_training):
                    if (Xtraindet[h][j]==0):
                        Xtraindet[h][j]=(reg.predict([Xtraindet[h][0:j]]))[0]+np.random.normal(0, np.sqrt(var[j]))
                for h in range(len_testing):
                    if (Xtestdet[h][j]==0):
                        Xtestdet[h][j]=(reg.predict([Xtestdet[h][0:j]]))[0]+np.random.normal(0, np.sqrt(var[j]))

            lda = LinearDiscriminantAnalysis()
            lda.fit(Xtraindet, Y1)
            results.append(lda.predict(Xtestdet))

        sol=maj_vote(results)

        from sklearn.metrics import accuracy_score
        return accuracy_score(Y2, sol)
