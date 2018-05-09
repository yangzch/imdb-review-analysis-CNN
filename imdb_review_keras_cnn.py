import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

BASE_DIR = '/home/zcyang/test/keras/case_data'
TRAIN_DATA_DIR = os.path.join(BASE_DIR, 'train')
#MAX_NB_WORDS = 56131
MAX_NB_WORDS = 20000

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = [] # list of label ids

for k in range(0, 10000):
    fpath = os.path.join(TRAIN_DATA_DIR, 'reviews'+str(k)+'.txt')
    f = open(fpath)
    a = f.readlines() 
    score = int(a[0])-1
    review = a[1]
    if len(a) < 6:
        for line in range (2,len(a)): 
            review=review+' '+a[line]
    else:
        review=review+' '+a[-5]+' '+a[-4]+' '+a[-3]+' '+a[-2]+' '+a[-1]        
        for line in range (2,len(a)-5):
            review=review+' '+a[line]

    texts.append(review)
    labels.append(score)
    f.close()

print('Found %s reviews.' % len(texts))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
#sequence_length = max(len(x) for x in sequences)
sequence_length = 500

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=sequence_length)

#labels = np.asarray(labels)
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

batch_size = 32
embedding_dims = 500
filters = 500
kernel_size = 5
hidden_dims = 500
epochs = 2
num_labels = 10

nfold = 5
for i in range(nfold):
    print "Training on fold " + str(i+1) + "/" + str(nfold) + "..."
    x_test  = data[2000*i:2000*(i+1), :]
    x_train = np.concatenate((data[:2000*i, :], data[2000*(i+1):, :]))
    y_test  = labels[2000*i:2000*(i+1), :]
    y_train = np.concatenate((labels[:2000*i, :], labels[2000*(i+1):, :]))
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Build model...')
    model = Sequential()

    # start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(MAX_NB_WORDS,
                    embedding_dims,
                    input_length=sequence_length))
    #model.add(Dropout(0.1))

    # add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
    # use max pooling:
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.1))

    # add a hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.1))
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))
 
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=10,
                        validation_data=(x_test, y_test))

    # Plot Model Loss and Model accuracy
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])  # RAISE ERROR
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy.png')
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss']) #RAISE ERROR
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png')

    y_pred = model.predict(x_test)

    y_pred = y_pred.argmax(axis=-1)
    y_test = y_test.argmax(axis=-1)

    print(y_pred)
    print(type(y_pred[1]))

    y_diff = np.absolute(y_pred-y_test)
    print(np.average(y_diff))

