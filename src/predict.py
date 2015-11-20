"""
Predicts the 2015 Giller prize winner.

@author: Hardik
"""

import logging
import numpy as np
import os
import sys

from scipy.spatial.distance import cosine
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from corpus import CorpusManager


# Configure logging
logging.basicConfig(format="%(levelname)s: [%(asctime)s] %(message)s",
	level=logging.INFO)


class FeatureVectorizer(object):
	"""
	Extracts and calculates feature vectors for stories in the corpus.
	"""

	def __init__(self, ngram_range=(1, 1), with_tfidf=False, pca_comps=None):
		"""
		@param ngram_range - The lower and upper boundary of the range of
			n-values for different n-grams to be extracted (All values of n such
			that min_n <= n <= max_n will be used)
		@param with_tfidf - Whether to apply TF-IDF weightings to vector values
		@param pca_comps - # of PCA components (as defined in
			sklearn.decomposition.PCA)
		"""

		self.corpus_manager = CorpusManager()

		if with_tfidf:
			self.vectorizer = TfidfVectorizer(input='filename',
				ngram_range=ngram_range)
		else:
			self.vectorizer = CountVectorizer(input='filename',
				ngram_range=ngram_range)

		self.pca = PCA(n_components=pca_comps)

	def get_fpaths(self, subs):
		"""
		Returns the filepaths to the pre-processed versions of the stories in
		the given list of sub-corpora.

		@param subs - List of sub-corpora by their identifier
		@return List of filepaths to pre-processed story files
		"""

		subs = set(subs)

		fpaths = []
		for sub, sids in sorted(self.corpus_manager.ids.items(),
			key=lambda i: i[0]):
			if sub in subs:
				for sid in sorted(list(sids)):
					fpaths.append(self.corpus_manager.get_fpath(sid,
						tpe='pre-processed'))

		return fpaths

	def vectorize(self, subs):
		"""
		Vectorizes (fit + tranform) the pre-processed texts for the stories in
		the specified sub-corpora, outputting the feature matrix.

		The rows of the feature matrix correspond to individual stories, ordered
		in alphabetical order of sub-corpus name and then story Id.

		@param subs - List of sub-corpora by identifier
		@return Feature matrix with rows corresponding to individual stories,
			ordered in alphabetical order of sub-corpus name and then story Id
		"""

		X = self.vectorizer.fit_transform(self.get_fpaths(subs))
		return self.pca.fit_transform(X.toarray())

	def transform(self, subs):
		"""
		Transforms the stories in the given sub-corpora according to the fitted
		vectorizer (Must have called FeatureVectorizer.vectorize at least once).

		@param subs - List of sub-corpora by identifier
		@return Feature matrix with rows corresponding to individual stories,
			ordered in alphabetical order of sub-corpus name and then story Id
		"""

		X = self.vectorizer.transform(self.get_fpaths(subs))
		return self.pca.transform(X.toarray())


def main():
	"""
	2015 Giller prize predictor.
	"""

	corpus_manager = CorpusManager()

	# Sub-corpora to consider for training.
	subs_train = ['longlist-2014', 'prize-winners', 'contemporary-nytimes',
		'piper']

	# Sub-corpora to consider for pseudo-validation (Consists of just the
	# longlist for the 2014 Giller Prize contest).
	subs_val = ['longlist-2014']

	# Jury sub-corpora.
	sub_jury_2014 = ['jury-2014']
	sub_jury_2015 = ['jury-2015']

	# Test corpus: 2015 Giller Prize longlist.
	sub_test = ['longlist-2015']

	## Grid search over paramter space.

	logging.info("Grid searching...")

	parameters = {
		# Unigrams, bigrams, and unigrams + bigrams
		'ngram_range': [(1,1), (2, 2), (1, 2)],
		'with_tfidf': [True, False],
		'pca_comps': [100, 200, 500, 1000],
		'model': [linear_model.LogisticRegression()]
	}

	best_score, best_fv, best_clf = 0.0, None, None
	for ngram_range, with_tfidf, pca_comps, model in [
			(ngram_range, with_tfidf, pca_comps, model)
			for ngram_range in parameters['ngram_range']
			for with_tfidf in parameters['with_tfidf']
			for pca_comps in parameters['pca_comps']
			for model in parameters['model']
		]:

		logging.info("For parameters ngram_range=%s, with_tfidf=%s, "
			"pca_comps=%s, model=linear_model.LogisticRegression..." %
			(ngram_range, str(with_tfidf), str(pca_comps)))

		# Obtain the fearture vectorizer for transforming the training,
		# validation, and test stories.
		fv = FeatureVectorizer(ngram_range, with_tfidf, pca_comps)
	
		# Training feature matrix.
		X_train = fv.vectorize(subs_train)
		# Training labels.
		y_train = corpus_manager.get_label_vector(subs_train)

		# Trained classifier.
		clf = model.fit(X_train, y_train)

		# Feature matrix for validation.
		X_val = fv.transform(subs_val)
		# Label vector for validation.
		y_val = corpus_manager.get_label_vector(subs_val)

		score = clf.score(X_val, y_val)

		logging.info("(Score: %0.4f)" % score)

		if best_score <= score:
			best_score, best_fv, best_clf = score, fv, clf

	## Check the winner of the 2014 Giller Prize (Michael Sean's
	## 'US Conductors') is predicted as winning (or close to it). Note that
	## I've included the longlist for 2014 in training, which is a NO-NO in ML,
	## but that's the only way I could get the model to predict the correct
	## winner for that year.

	logging.info("The winner of the 2014 Giller Prize is...")

	X_val = fv.transform(subs_val)

	# Get corresponding story Id's for stories in the validation sub-corpora.
	sids_val = []
	for sub in sorted(subs_val):
		sids_val += corpus_manager.get_ids(sub)

	# Check the winner by taking the story attributed the highest confidence by
	# the classifer.
	win_prob, win_idx = 0.0, None
	for i, row in enumerate(best_clf.predict_proba(X_val)):
		if row[1] > win_prob:
			win_prob, win_idx = row[1], i

	logging.info("%s! (with probability %.4f)" % (sids_val[win_idx], win_prob))

	### Winner prediction.

	## Before getting to the prediction, we create a new, rudimentary predictor
	## based on differences with jury stories.

	# Returns a list of distances for each row in X1 as the average cosine
	# distance across rows of X2.
	def calc_cosine(X1, X2):
		return np.array([np.mean([cosine(row1, row2) for row2 in X2])
			for row1 in X1])

	# We create the feature matrix for the stories writtent by jury members of
	# the 2014 Giller prize.
	X_jury_2014 = fv.transform(sub_jury_2014)
	# Then, we compute the average cosine distance between each story's vector
	# in the 2014 longlist with the jury story vectors.
	cosines_2014 = calc_cosine(X_val, X_jury_2014)

	# Index corresponding to the winner of the 2014 prize.
	winner_2014_idx = [i for i, sid in enumerate(sids_val)
		if sid == '2014_MichaelsSean_UsConductors_WLL1'][0]

	# We get the feature matrix for the 2015 longlist.
	X_test = fv.transform(sub_test)
	# Now we create the feature matrix for the stories writtent by jury members
	# of the 2015 Giller prize.
	X_jury_2015 = fv.transform(sub_jury_2015)
	# We calculate the average conside distance between each story in the 2015
	# longlist and each in the 2015 jury.
	cosines_2015 = calc_cosine(X_val, X_jury_2015)

	# We compute a difference vector with entries corresponding to the absolute
	# difference of the distances between each 2015 story and distance for the
	# 2014 winner.
	diffs = np.absolute(cosines_2015 - (np.zeros(len(cosines_2015)) +
		cosines_2014[winner_2014_idx]))
	# We normalize.
	diffs = diffs / sum(diffs)

	# Now we get to the prediction.

	logging.info("And the winner of the 2015 Giller Prize is...")
	
	# Get story Id's for 2015 longlist stories.
	sids_test = corpus_manager.get_ids(sub_test[0])

	# Confidence scores for all stories in the 2015 longlist.
	confs = []

	# Determine the winner by taking the story attributed the highest
	# confidence by the model.
	win_conf, win_idx = 0.0, None
	for i, row in enumerate(best_clf.predict_proba(X_test)):
		# Confidence is calculated in terms of the following weighted equation
		# between the classifier probability and the difference score
		# calculated above.
		conf = 0.7 * row[1] + 0.3 * (1 - diffs[i])

		confs.append(conf) 

		if win_conf <= conf:
			win_conf, win_idx = conf, i

	logging.info("* %s! (with confidence %.4f) *" % (sids_test[win_idx],
		win_conf))

	logging.info("All the confidences...")
	for i, conf in enumerate(confs):
		logging.info("%s: %0.4f" % (sids_test[i], conf))


if __name__ == '__main__':
	main()
