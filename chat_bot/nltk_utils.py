import nltk
import numpy as np
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):                         #Tokenization is the process of breaking down a piece of text into individual words
   return nltk.word_tokenize(sentence)

def stem(word):                                 #Stemming is the process of reducing inflected (or sometimes derived) words to their word stem, base or root form.
   return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
   tokenized_sentence = [stem(w) for w in tokenized_sentence]
   bag = np.zeros(len(all_words), dtype=np.float32)
   for index, w in enumerate(all_words):
      if w in tokenized_sentence:
         bag[index] = 1.0
   return bag