import re
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk import word_tokenize


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z]')
STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

class CleanData(object):

	def get_wordnet_pos(word):
	    """Map POS tag to first character lemmatize() accepts"""
	    tag = nltk.pos_tag([word])[0][1][0].upper()
	    tag_dict = {"J": wordnet.ADJ,
	                "N": wordnet.NOUN,
	                "V": wordnet.VERB,
	                "R": wordnet.ADV}

	    return tag_dict.get(tag, wordnet.NOUN)

	def word_pos(text):
		'''1. Init Lemmatizer'''
		lemmatizer = WordNetLemmatizer()
		list_of_word = []
		for w in nltk.word_tokenize(text):
			list_of_word.append(lemmatizer.lemmatize(w, get_wordnet_pos(w)))
		txt_to_write = ' '.join(word for word in list_of_word)
		# print("txt_to_write  -----",txt_to_write)
		return txt_to_write



	def clean_text(text):
	    """
	        text: a string
	        
	        return: modified initial string
	    """
	    text = text.lower() # lowercase text
	    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
	    text = BAD_SYMBOLS_RE.sub(' ', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
	    # text = text.replace('x', ' ')
	#    text = re.sub(r'\W+', '', text)
	    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
	    # print("txt  -----",text)
	    return text