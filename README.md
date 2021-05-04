# Capstone: Rumor Detection On Deep Learning Concatenating Hand-crafted Features and Context Embedding (One-page Summary)

## Abstract
With the rapid development and integration of social media into people's daily lives, the spread of false information and unverified rumors has been a rising problem. Many researchers have used machine learning and deep learning to address the rumor classification task to catch the propagation of rumors in real-time or after the fact, to understand the behaviors of spreading rumors, and label unverified rumorous tweets. This project performs the rumor detection task with SVM and Deep Learning classifiers using different sets of hand-crafted features and tweet context embeddings inspired by previous researches. The hand-crafted features include the source tweet's Twitter object fields and statistics of tweets derived by the source tweet. To test the generalization ability of proposed methods on unseen data and rumors, cross-validation is applied on PHEME and PHEME-R datasets, and an analysis of the results is presented. The final results achieved an accuracy of 70.16 and a recall score of 83.56, yet show a relative lower precision score, 56.74.

## I. INTRODUCTION
Rumor Detection is a binary classification task to classify whether a social media post is reporting rumors, which is not yet verified when it is spreading. This final year project tackles the rumor detection problem by training classification models with the tweet dataset that contains rumorous tweets and compares the results from various features representing the rumorousness of the tweets in different vector spaces. The rumors are assumed to be unseen by the model.

## II. DESIGN/METHODOLOGY/IMPLEMENTATION
This project approaches the rumor detection problem by utilizing two different aspects of the chosen dataset. The first is to exploit tweet object data stored in the files of JSON format to create hand-crafted features. The second is to perform word embedding algorithms to get dense vectors of tweet texts and then feed them into a deep neural network. The finalized list of features extracted for this experiment is presented in figure 1.

### A. Data 

The experiment used the PHEME dataset and PHEME-R dataset. PHEME dataset contains a total of 5,802 annotated rumors that consist of 3,830 tweets and 1,972 tweets that are deemed to be either rumors or not rumors, respectively.

### B. Cross Validation

Furthermore, to avoid the potential overfitting problems that could worsen the model's generalization ability on the small dataset of PHEME, k-fold Cross Validation is applied on PHEME and PHEME-R dataset, respectively, per event.

## III. EVALAUATION AND RESULTS
![image](https://user-images.githubusercontent.com/18901970/117030779-8bc75a80-ad32-11eb-88f0-ad1904b42d23.png)
TABLE I. THE RESULT OF BASELINE CLASSIFIERS INCLUDING CV-9 SVM, AND CRF ON PHEME DATASET

![image](https://user-images.githubusercontent.com/18901970/117030877-a4377500-ad32-11eb-947d-7c1e1ad43186.png)
THE RESULT OF CV-9 & NEURAL NETWORK WITH WEIGHTED SAMPLING

## IV. CONCLUSION
The analysis of the performances of each model and feature set, respectively, are presented in Table 1 and 2. The final model can classify the unseen events based on the feature sets extracted from the rumorous tweets from other events that have been discussed on Twitter, with an accuracy of 76.08 and an F1 score of 44.32 on the PHEME dataset, and an accuracy of 70.16 and an F1 score of 42.43 on the dataset combining the PHEME and the PHEME-R dataset.
