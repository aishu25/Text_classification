# coding: utf-8
# Loading datasets
from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

#printing categories
print twenty_train.target_names
# ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 
# 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 
# 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 
# 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 
# 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 
# 'talk.politics.misc', 'talk.religion.misc']

#prints first line of the first data file
print ("\n".join(twenty_train.data[0].split("\n")[:3]))

#Extracting features from text files
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape
print X_train_counts.shape


#TF - IDF
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print X_train_tfidf.shape

#Machine Learning
#Naive Bayes(NB) classifer

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
print "CLF : ", clf

# Building a pipeline: We can write less code and do all of the above, by building a pipeline as follows:
# The names ‘vect’ , ‘tfidf’ and ‘clf’ are arbitrary but will be used later.
# We will be using the 'text_clf' going forward.

from sklearn.pipeline import Pipeline


text_clf = Pipeline([('vect', CountVectorizer()),
					 ('tfidf', TfidfTransformer()),
					 ('clf', MultinomialNB())])

text_clf = text_clf.fit(twenty_train.data, twenty_train.target)
print "TEXT_CLF", text_clf

#Performance of NBClassifier
import numpy as np
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)

predicted = text_clf.predict(twenty_test.data)
print "PREDICTED : ", predicted
res = np.mean(predicted == twenty_test.target)
print "RESULT using NBClassifier : ", res
#RESULT using NBClassifier :  0.77389803505

#Performance of SGDClassifier
from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer()),
						 ('tfidf',TfidfTransformer()),
						 ('clf-svm', SGDClassifier(loss='hinge',penalty='l2',alpha=1e-3,n_iter=5,random_state=42))
						 ])
# penalty = ‘l2’ which is the standard regularizer for linear SVM models
# alpha = Constant that multiplies the regularization term. Defaults to 0.0001 Also used to compute learning_rate when set to ‘optimal’.
# loss = Defaults to ‘hinge’, which gives a linear SVM.
# random_state = The seed of the pseudo random number generator to use when shuffling the data.
# n_iter = The number of passes over the training data (aka epochs). Defaults to None. Deprecated, will be removed in 0.21.
 

text_clf_svm = text_clf_svm.fit(twenty_train.data, twenty_train.target)
predicted_svm = text_clf_svm.predict(twenty_test.data)
resl = np.mean(predicted_svm == twenty_test.target)
print  "RESULT using SGDClassifier : ", resl
# RESULT using SGDClassifier :  0.823818374934

# Grid Search
# Here, we are creating a list of parameters for which we would like to do performance tuning. 
# All the parameters name start with the classifier name (remember the arbitrary name we gave). 
# E.g. vect__ngram_range; here we are telling to use unigram and bigrams 
# and choose the one which is optimal.

from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1,1), (1,2)],
			  'tfidf__use_idf': (True, False),
			  'clf__alpha': (1e-2, 1e-3)}

# Creating instace of the GridSearch by passing the classifier, parameters and
# n_jobs=-1 which tells to use multiple cores from user machine


print "*****************OUT***************"
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)
print gs_clf.best_score_ # 0.906752695775
print gs_clf.best_params_ # {'vect__ngram_range': (1, 2), 'tfidf__use_idf': True, 'clf__alpha': 0.01}

# Grid Search for Support Vector Machine(SVM)

from sklearn.model_selection import GridSearchCV
parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),'clf-svm__alpha': (1e-2, 1e-3)}

gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(twenty_train.data, twenty_train.target)

gs_clf_svm.best_score_ 
gs_clf_svm.best_params_

print gs_clf_svm.best_score_ 
print gs_clf_svm.best_params_













