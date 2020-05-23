"""The objective is to create TFIDF Vectors of all the states.
   Each state has 2 Articles.
   The States are created and the Articles are merged
   Then TFIDF is calculated and stored in a CSV File"""
# Import Libraries
import numpy as np
import itertools as it
import pandas as pd
import re, unicodedata
import nltk
import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

file=pd.read_csv(r'C:\Users\visanand\Downloads\Atul\DQN - Final\TFIDF-States\Articles.csv',header = None)

Col1 =file[1]
Col2 = file[2]

Col1 = Col1.values.tolist()
Col2 = Col2.values.tolist()

Data = []

for i in range(len(Col1)):
    file = Col1[i] + " " + Col2[i] 
    Data.append(file)
    
#result = pd.DataFrame(Data)

#result.to_csv(r'C:\Users\atukumar2\Downloads\Data\Combined.csv',header = None, index = False)

State_Space = []

# Creating states with unique combinations of all the Articles such that each state is set of 3 Articles
State_Space=list(it.combinations(Data, 2))

# Download Packages
nltk.download('punkt')
nltk.download('wordnet')

# Functoin for Text Preprocessing
def remove_non_ascii(words):
    """ 
    Removes non alphabetical characters.  
  
    Parameters: 
    arg1 (list): Takes in a list of words
  
    Returns: 
    list: Returns a list after removing the non alphabetical characters. 
  
    """
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """ 
    Converts everything to lowercase characters.  
  
    Parameters: 
    arg1 (list): Takes in a list of words
  
    Returns: 
    list: Returns a list of lowercase characters.
  
    """
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """ 
    Removes punctuation.  
  
    Parameters: 
    arg1 (list): Takes in a list of words
  
    Returns: 
    list: Returns a list after removing puntuation. 
  
    """
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """ 
    Runs inflect engine  
  
    Parameters: 
    arg1 (list): Takes in a list of words
  
    Returns: 
    list: Returns a list words. 
  
    """
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """ 
    Removes stopwords.  
  
    Parameters: 
    arg1 (list): Takes in a list of words
  
    Returns: 
    list: Returns a list of characters after removing stopwords. 
  
    """
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def lemmatize_verbs(words):
    """ 
    Lemmatization is done here.  
  
    Parameters: 
    arg1 (list): Takes in a list of words
  
    Returns: 
    list: Returns a list of lemmatized characters 
  
    """
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    """ 
    Normalization is done here.  
  
    Parameters: 
    arg1 (list): Takes in a list of words
  
    Returns: 
    list: Returns a list of normalized characters 
  
    """
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    return words
def lemmatize(words):
    lemmas = lemmatize_verbs(words)
    return lemmas

States = []

for i in range(len(State_Space)):
    S = State_Space[i][0] + " " + State_Space[i][1] 
    States.append(S)
    

preprocessed = []

# Preprocessing the text in the states
for i in range(0,len(States)):
    words = nltk.word_tokenize(States[i])
    words = normalize(words)
    lemmas = lemmatize(words)
    preprocessed.append(lemmas)

# Creating a corpus of unique words from all the states
corpus = set().union(*preprocessed)

wordDict = []

for i in range(len(States)):
    wordDict.append(dict.fromkeys(corpus, 0))
    
for j in range(len(States)):
    for i in preprocessed[j]:
        wordDict[j][i] = wordDict[j][i] + 1

# Function to compute TF
def computeTF(wordDict, bow):
    """ 
    Computes TF.  
  
    Parameters: 
    arg1 (list): Takes in a list of word dictionary having unique words
    arg2 (list): Takes in a list of bag of words
  
    Returns: 
    list: Returns the computed TF 
  
    """
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count/float(bowCount)
    return tfDict

TF = []

for i in range(len(States)):
    TF.append(computeTF(wordDict[i],preprocessed[i]))

# Function to compute IDF
def computeIDF(docList):
    """ 
    Computes IDF.  
  
    Parameters: 
    arg1 (list): Takes in a list of word dictionary having unique words
    arg2 (list): Takes in a list of bag of words
  
    Returns: 
    list: Returns the computed IDF. 
  
    """
    import math
    idfDict = {}
    N = len(docList)
    
    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1
    
    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / float(val))
        
    return idfDict

IDF = computeIDF(wordDict)

# Function to compute TFIDF
def computeTFIDF(tfBow, idfs):
    """ 
    Computes TF-IDF.  
  
    Parameters: 
    arg1 (list): Takes in a list of TFs
    arg2 (list): Takes in a list of IDFs
  
    Returns: 
    list: Returns the computed TF-IDF. 
  
    """
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return tfidf

TFIDF = []

for i in range(len(States)):
    TFIDF.append(computeTFIDF(TF[i],IDF))

result = pd.DataFrame(TFIDF)

result.to_csv(r'C:\Users\visanand\Downloads\Atul\DQN - Final\TFIDF-States\TFIDF-States.csv',index = False)

