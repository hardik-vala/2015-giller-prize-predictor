"""
Pre-processes all texts in the corpus, tokenizing, lemmatizing, and replacing
all NE's by their CoreNLP NER label. The output texts are saved to a specified
folder mimicing the corpus texts directory structure.

@author: Hardik
"""

import logging
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], os.path.join('..', 'src')))

import xml.etree.ElementTree as et

from corpus import CorpusManager


# Configure logging
logging.basicConfig(format="%(levelname)s: [%(asctime)s] %(message)s",
	level=logging.INFO)


def preprocess(corenlp_path):
	"""
	Pre-processes the given text using it's CoreNLP output (stored in a .xml
	file located by the given path) by tokenizing, lemmatizing, and replacing
	all NE's by their CoreNLP NER label. Also, tokens are simply separated by a
	single space in the output, with all sentence structure lost.

	@param corenlp_path - Path to CoreNLP .xml file
	@return Space-seperated string of preprocessed tokens
	"""

	preprocessed_tokens = []
	for tok in et.parse(corenlp_path).getroot().iter('token'):
		# NER label.
		label = tok[5].text
		# If no label is assigned, then append the lemma of the word, otherwise
		# the label.
		preprocessed_tokens.append(tok[1].text if label == 'O' else label)

	return ' '.join(preprocessed_tokens)


def main():
	"""
	Pre-processes all texts in the corpus, tokenizing, lemmatizing, and
	replacing all NE's by their CoreNLP NER label. The output texts are saved to
	a specified folder mimicing the corpus texts directory structure.
	"""

	cm = CorpusManager()

	for root, subdirnames, fnames in os.walk(cm.CORENLP_DIRPATH):
		for fname in fnames:
			if fname.endswith('.xml'):
				in_path = os.path.join(root, fname)
				out_path = in_path.replace(cm.CORENLP_DIRPATH,
					cm.PREPROCESSED_DIRPATH)[:-4] + '.txt'

				if os.path.exists(out_path):
					logging.info("%s already exists. Skipping %s..." %
						(out_path, in_path))
					continue

				# If the parent directory doesn't exist, create it.
				par_dirpath = os.path.split(out_path)[0]
				if not os.path.exists(par_dirpath):
					os.makedirs(par_dirpath)

				# Temporary.
				if not par_dirpath.endswith('piper'):
					continue

				logging.info("Pre-processing %s..." % in_path)
				preprocessed_text = preprocess(in_path)

				logging.info("Outputting to %s..." % out_path)
				with open(out_path, 'w') as f:
					f.write(preprocessed_text.encode('utf-8'))


if __name__ == '__main__':
	main()
