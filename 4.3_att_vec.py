import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from gensim.models import Word2Vec
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, LSTM, Bidirectional
from keras.models import Sequential
from sklearn.metrics import accuracy_score
import time
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# cve, vul_type, att_type, att_vec, root_cau
labelled_df = pd.read_csv('data/labelled.csv')
# labelled_df = pd.read_csv('data/labelled_half.csv')

X_labelled = labelled_df['cve']
y_labelled = labelled_df['att_vec']

att_vecs = ['Via field argument or parameter', 'Via crafted data or file', 'By executing script',
            'Via direct request', 'Via crafted device', 'Via protocol-based attack']
n_classes = len(att_vecs)

y_labelled_codes = pd.factorize(y_labelled, sort=True)[0]

X_train_labelled, X_test_labelled, y_train_labelled, y_test_labelled = train_test_split(
    X_labelled, y_labelled_codes, test_size=0.2, stratify=y_labelled, random_state=6
)

model_w2v = Word2Vec.load('model/cve_w2v')

def word2vec_transform(text, model, size):
    text = text.apply(lambda x: [word.lower() for word in re.findall(r'\w+', x) if word.lower() in model.wv.key_to_index])
    text = text.apply(lambda x: np.mean([model.wv.get_vector(word) for word in x], axis=0) if len(x) > 0 else np.zeros(size))
    text = np.vstack(text)
    text = text / np.linalg.norm(text, axis=1)[:, np.newaxis]
    return text

y_train_labelled_codes = pd.factorize(y_train_labelled, sort=True)[0]

X_train_labelled_trans = word2vec_transform(X_train_labelled, model_w2v, 100)
X_test_labelled_trans = word2vec_transform(X_test_labelled, model_w2v, 100)

# gnb
start_time = time.time()
gnb = GaussianNB()
gnb.fit(X_train_labelled_trans, y_train_labelled_codes)

gnb_pred = gnb.predict(X_test_labelled_trans)
gnb_score = accuracy_score(y_test_labelled, gnb_pred)
print(gnb_score)
print(time.time()-start_time)

# svm
start_time = time.time()
model = SVC(kernel='linear')
model.fit(X_train_labelled_trans, y_train_labelled_codes)

svm_pred = model.predict(X_test_labelled_trans)
svm_score = accuracy_score(y_test_labelled, svm_pred)
print(svm_score)
print(time.time()-start_time)

# cnn
start_time = time.time()
train_y = keras.utils.to_categorical(y_train_labelled, num_classes=n_classes) # one-hot

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train_labelled)

X_train_cnn = tokenizer.texts_to_sequences(X_train_labelled)
X_test_cnn = tokenizer.texts_to_sequences(X_test_labelled)

maxlen = 100
X_train_cnn = pad_sequences(X_train_cnn, maxlen=maxlen)
X_test_cnn = pad_sequences(X_test_cnn, maxlen=maxlen)

embedding_dim = 50
embedding_index = {}
embedding_file = 'model/CVE_specific_vectors.txt'

with open(embedding_file, encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

embedding_matrix = np.zeros((5000, embedding_dim))

for word, i in tokenizer.word_index.items():
    if i < 5000:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

cnn = Sequential()
cnn.add(Embedding(5000, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False)) # 嵌入层
cnn.add(Conv1D(32, 5, activation='relu')) # 卷积层
cnn.add(GlobalMaxPooling1D()) # 全局最大池化层
cnn.add(Dense(n_classes, activation='softmax')) # 全连接层
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

cnn.fit(X_train_cnn, train_y, epochs=50, batch_size=32, verbose=0)

cnn_pred = cnn.predict(X_test_cnn)
cnn_pred = np.argmax(cnn_pred, axis=1)
cnn_score = accuracy_score(y_test_labelled, cnn_pred)
print(cnn_score)
print(time.time()-start_time)