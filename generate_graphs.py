import src.plots.printing1graph as pr

#Parameters
dimensions=[5,10,25,50,100]
cov_matrice=['random','normal']
'''str_correlation_higherIndex','str_correlation+high_diagonal'''
probs_missingness=[0.05,0.1,0.3,0.5]
type_missingness=['MCAR','MAR','MNAR']

for c in cov_matrice:
    for d in dimensions:
        for p in probs_missingness:
            for t in type_missingness:
                pr.one_graph(c,d,t,p)
