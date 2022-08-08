# MScProjects
A couple of projects from my MSc in Statistical Science at the University of Oxford.

## FakeNewsDetection
This project involved predicting whether a given news article is 'Fake News' or not. Given a dataset of 1000 articles and 12247 words, with 1 indicating the presence of a word in a document and 0 otherwise (see X_train.csv and y_train.csv), the goal of this project was to build a classifier with as high a generalisation accuracy as possible on the test dataset (X_test.csv). Many methods were used, but ultimately a LGBM model was chosen as the final classifier, with Bayesian Optimisation employed to tune its hyperparameters (see FakeNewsDetection.py). The final report is given in FakeNews_Report.pdf, while the predictions for the test dataset are given in Predictions.csv.

## PhDPublications
Given a dataset detailing various characteristics of PhD students, the goal of this project was to build a (Poisson) Generalised Linear Model which predicts the number of publications the student will produce in the final year of their degree. The final report is given in PhDPublications_Report.pdf, with the associated code in PhDPublications.R.
