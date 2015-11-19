"""
Manages all corpus texts and information.

@author: Hardik
"""

import csv
import os
import sys

from collections import defaultdict


class CorpusManager(object):
	"""
	Manages all corpus texts and information.
	"""

	# Path to corpus data.
	DATA_DIRPATH = "/home/ndg/project/users/hvala/andrews-challenge/data"

	# Path to texts directory.
	TEXTS_DIRPATH = os.path.join(DATA_DIRPATH, 'texts')
	# Path to directory with CoreNLP .xml files.
	CORENLP_DIRPATH = os.path.join(DATA_DIRPATH, 'corenlp')
	# Path to pre-processed texts directory.
	PREPROCESSED_DIRPATH = os.path.join(DATA_DIRPATH, 'pre-processed')

	# Sub-corpora designations.
	SUBS = ['longlist-2014', 'longlist-2015', 'jury-2014', 'jury-2015',
		'prize-winners', 'contemporary-nytimes', 'piper']

	def __init__(self):
		# Retrieves all the story id's of the stories in the corpus and returns
		# a dictionary of them (as sets) organized by sub-corpora.
		def get_ids():
			ids = defaultdict(set)
			for root, subdirnames, fnames in os.walk(self.TEXTS_DIRPATH):
				for fname in fnames:
					if not fname.endswith('.txt'):
						continue

					sub = os.path.basename(root)
					ids[sub].add(fname[:-4])

			return ids

		self.ids = get_ids()

	def get_ids(self, sub):
		"""
		Returns the story Id's for the given sub-corpus sub.

		@param sub - Sub-corpus identifier (CorpusManager.SUBS gives all
			sub-corpus designations)
		@return List of Id's for stories in the sub-corpus (sorted
			alphabetically)
		"""

		return sorted(list(self.ids[sub]))

	def get_infopath(self, sub):
		"""
		Returns the filepath to the information .csv file for the given
		sub-corpus sub.

		@param sub - Sub-corpus identifier (CorpusManager.SUBS gives all
			sub-corpus designations)
		@return Filpath to information .csv file
		"""

		if sub == 'prize-winners':
			return os.path.join(self.DATA_DIRPATH, 'info-prize.csv')
		else:
			raise NotImplementedError

	def get_sub(self, sid):
		"""
		Returns the sub-corpora name for which the given story with id sid
		belongs to (Returns None if it doesn't belong to any).

		@param sid - Story id.
		@return Corresponding sub-corpus identifier
		"""

		for sub, sids in self.ids.iteritems():
			if sid in sids:
				return sub

		return None

	def get_fpath(self, sid, tpe):
		"""
		Returns the full filepath to a data file for the story with id sid.

		@param sid - Story Id
		@param tpe - Type of data (If it's 'text', 'corenlp', or
			'pre-processed', then the path to the original text, CoreNLP .xml,
			or pre-processed text, respectively, is returned)
		@return Filepath to data file
		"""

		if tpe == 'text':
			dirpath = self.TEXTS_DIRPATH
		elif tpe == 'corenlp':
			dirpath = self.CORENLP_DIRPATH
		elif tpe == 'pre-processed':
			dirpath = self.PREPROCESSED_DIRPATH
		else:
			raise ValueError("tpe must be 'text', 'corenlp', or "
				"'pre-processed'.")

		sub = self.get_sub(sid)
		if sub == None:
			return ValueError("Unrecognized Id %s." % sid)

		# File extension.
		ext = '.xml' if tpe == 'corenlp' else '.txt'

		return os.path.join(os.path.join(dirpath, sub), sid + ext)

	def get_labels(self):
		"""
		Returns a dictionary of dictionaries of the labels, first keyed by
		sub-corpus and then by story Id. The labels are 1 and 0 (integers),
		representing "winners" and "non-winners", respectively. The jury
		sub-corpora and the longlist for 2015 are excluded.

		@return Dictionary of dictionaries of labels
		"""

		labels = defaultdict(dict)
		for sub, sids in self.ids.iteritems():
			if sub == 'longlist-2014':
				labels[sub] = {sid: (1
						if sid == '2014_MichaelsSean_UsConductors_WLL1' else 0)
					for sid in sids}

			if sub == 'longlist-2015':
				continue

			if sub.startswith('jury'):
				continue

			if sub == 'prize-winners':
				with open(self.get_infopath(sub), 'rb') as f:
					reader = csv.reader(f, delimiter=',', quotechar='"')

					# Skip header.
					reader.next()

					for row in reader:
						sid = row[1][:-4]
						if row[-1].lower() == 'winner':
							labels[sub][sid] = 1
						else:
							labels[sub][sid] = 0
			if sub == 'contemporary-nytimes':
				labels[sub] = {sid: 1 for sid in sids}

			if sub == 'piper':
				labels[sub] = {sid: 0 for sid in sids}

		return labels

	def get_label_vector(self, subs):
		"""
		Returns the labels for each story in the given list of sub-corpora as
		a vector.

		@param subs - List of sub-corpora by identifier
		@return Label vector ordered first by sub-corpus name and then by story
			Id
		"""

		subs = set(subs)

		label_vector = []
		# Iterate over sub-corpora in alphabetical order.
		for sub, labels_subdict in sorted(self.get_labels().items(),
			key=lambda i: i[0]):
			if sub in subs:
				# Iterate over story Id's in alphabetical order.
				for sid, label in sorted(labels_subdict.items(),
					key=lambda i: i[0]):

					label_vector.append(label)

		# Returned as a list of values.
		return label_vector
