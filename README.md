# Imputation before LDA classification
This repository aims at testing and comparing several imputation methods before performing LDA classification.
## Description
We run our experiments on 2 datasets. The first one is a data banknote authentification dataset containing four numerical features and a binary class.
The second is simply generated by sampling from 2 multivariate gaussian distributions.
The nature of the covariance matrices as well as the dimension (number of features) can be defined via command line arguments, as well as the missingness probability and the type of missingness.
The imputation methods that get tested are "Grand Mean", "Conditional Mean", "Nearest Neighbour", "Regression".
The concept of conditional mean is described in the pdf file at the root of the repository, along with the description of my experiments and conclusion.
## Getting Started
### Dependencies
matplotlib=3.8 \
numpy=1.26 \
pillow=10.0 \
python=3.12 \
scikit-learn=1.3.0
### Installing
cd ImputationForLDA \
conda create --name {env} --file requirements.txt
### Executing program
python main.py --dimensions 5 --cov_matrice normal --probs_missingness 0.1 --type_missingness MCAR
## Help
Please reach out.
## Authors
Mathieu Charbonnel
## Version History

* December 2023
    * Initial Release

## License
This project is not licensed.

## Acknowledgments
To Robert J. Durrant who supervised this scientific work.\
Lohweg,Volker. (2013). banknote authentication. UCI Machine Learning Repository.
