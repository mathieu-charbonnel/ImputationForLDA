config_data=dict(
    dimensions=[5, 10, 25, 50, 100],
    cov_matrice=['random',
                 'normal', 
                 'str_correlation_higherIndex',
                 'str_correlation+high_diagonal'],
    probs_missingness=[0.05,0.1,0.3,0.5],
    type_missingness=['MCAR','MAR','MNAR'] 
)