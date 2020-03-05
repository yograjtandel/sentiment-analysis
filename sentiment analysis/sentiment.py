import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk import word_tokenize
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_frame = pd.read_csv('dataset_2lc_with_lemetisation1.csv',encoding = "ISO-8859-1")

''' graph of data '''
# df=data_frame['label'].value_counts().sort_values(ascending=False)
# print(df)
# x = list(data_frame['label'].unique()) 
# plt.bar(x,df)
# plt.show()
# print(data_frame.info())


data_frame = data_frame.reset_index(drop=True)
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z]')
STOPWORDS = set(stopwords.words('english'))

# The maximum number of words to be used. (most frequent)
vocab_size = 50000
# Max number of words in each complaint. 
max_len = 250
# This is fixed.
embedding_dim = 100
oov_tok = '<oov>'
trunc_type = 'post'
padding_type = 'post'


# '''1. Init Lemmatizer'''
# lemmatizer = WordNetLemmatizer()


# def get_wordnet_pos(word):
#     """Map POS tag to first character lemmatize() accepts"""
#     tag = nltk.pos_tag([word])[0][1][0].upper()
#     tag_dict = {"J": wordnet.ADJ,
#                 "N": wordnet.NOUN,
#                 "V": wordnet.VERB,
#                 "R": wordnet.ADV}

#     return tag_dict.get(tag, wordnet.NOUN)

# def word_pos(text):
#     list_of_word = []
#     for w in nltk.word_tokenize(text):
#       list_of_word.append(lemmatizer.lemmatize(w, get_wordnet_pos(w)))
#     txt_to_write = ' '.join(word for word in list_of_word)
#     # print("txt_to_write  -----",txt_to_write)
#     return txt_to_write



def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub(' ', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text

def prepare_text_for_inpute():
    tokenizer = Tokenizer(num_words=vocab_size, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True,oov_token = oov_tok)
    tokenizer.fit_on_texts(data_frame['review'].values)
    word_index = tokenizer.word_index

    print(word_index)
    
    with open('word_index.pickle', 'wb') as handle:
        pickle.dump(word_index,handle, protocol = pickle.HIGHEST_PROTOCOL)

    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer,handle, protocol = pickle.HIGHEST_PROTOCOL)

    global X
    global Y
    global X_train,X_test,Y_train,Y_test
    X = tokenizer.texts_to_sequences(data_frame['review'].values)
    print('data tensor before:', X[1])
    X = pad_sequences(X, maxlen=max_len,padding = padding_type,truncating = trunc_type)
    print(X.shape,'Shape of data tensor:', type(X))
    print('data tensor:', X[1])
    Y = pd.get_dummies(data_frame['label']).values
    print('Shape of label tensor:', Y.shape)
    print('dummy sentence',X[1],'y----',Y[0])


    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.05, random_state = 42)
    print(X_train.shape,Y_train.shape)
    print(X_test.shape,Y_test.shape)
        

def create_model():
    global model
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, embedding_dim, input_length = X.shape[1]))
    model.add(keras.layers.SpatialDropout1D(0.2))
    model.add(keras.layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(keras.layers.Dense(3, activation=tf.nn.softmax))
    model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])
    model.summary()

def train_model():
    epochs = 5
    batch_size = 64
    global history
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.05,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
    model.save('model.h5')

def evaluate_model():
    accr = model.evaluate(X_test,Y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show();

    plt.title('Accuracy')
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.show();


# data_frame['review'] = data_frame['review'].apply(word_pos)
# print("after lemmitisation -- >",data_frame['review'][0])
data_frame['review'] = data_frame['review'].apply(clean_text)
# print("after clean text -- >",data_frame['review'][0])
data_frame['review'] = data_frame['review'].str.replace('\d+', '')
# print("after remove digit -- >",data_frame['review'][0])


prepare_text_for_inpute()
create_model()
train_model()
evaluate_model()










