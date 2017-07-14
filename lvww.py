import sys
import random
import numpy as np 
import pandas as pd
from sklearn import metrics
from sklearn import cross_validation
from sklearn.svm import SVC



class LVW:
	def __init__(self):
		self.notify = 'find better feature subspace, iteration restart from 0'

	def lvw(self, X, y, clf, iteration, opti_obej='accuracy', score_base=0.5, num_folds=10):
		"""Las vegas wrapper
		Args:
			X(pandas.dataframe): input data 
			y(pandas.dataframe): labels
			clf: scikit-learn pre-trained classifier
			opti_obej(str): accuracy, precision, recall or f1, default accuracy
			score_base(float): baseline of optimize score, default .5
			num_folds(int): number of cv folds
		Returns:
			best_features(list): best feature subset
			score(float): optimized accuracy/precision/recall or f1 score
		"""
		X,y=self.check_X_y(X,y)
		score = score_base
		num_instances, num_features = X.shape
		#initialize best features and iteration
		best_features = list(range(num_features))
		t = 0
		while t < iteration:
			if t%50==0:
				print ('iteration: %d' % t)
			features_new = random.sample(range(0, num_features), random.randint(1,num_features))
			k_fold = cross_validation.StratifiedKFold(y, n_folds=num_folds)
			X_new = X[:,features_new]
			temp_score = []
			for fold_id, (train_idx, test_idx) in enumerate(k_fold):
				X_train, X_test = X_new[train_idx], X_new[test_idx]
			y_train, y_test = y[train_idx], y[test_idx]
			clf.fit(X_train,y_train)
			y_pred = clf.predict(X_test)
			if opti_obej=='accuracy':
				temp_score.append(metrics.accuracy_score(y_test,y_pred))
			elif opti_obej=='precision':
				temp_score.append(metrics.precision_score(y_test, y_pred))
			elif opti_obej=='recall':
				temp_score.append(metrics.recall_score(y_test,y_pred))
			elif opti_obej=='f1':
				temp_score.append(metrics.f1_score(y_test,y_pred))
			else:
				sys.exit()
			temp_score = sum(temp_score)/float(len(temp_score))
			if temp_score > score:
				print (self.notify)
				t = 0
				score = temp_score
				best_features = features_new
			else:
				t += 1
		return best_features, score
	def check_X_y(self, X, y):
		"""Check input, if pandas.dataframe, transform to numpy array
		Args:
			X(ndarray/pandas.dataframe): input instances 
			y(ndarray/pandas.series): input labels
		Returns:
			X(ndarray): input instances 
			y(ndarray): input labels
		"""
		if isinstance(X,pd.core.frame.DataFrame):
			X = X.as_matrix()
		if isinstance(y,pd.core.series.Series):
			y = y.values.flatten()
		return X,y







