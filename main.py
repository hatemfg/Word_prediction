from nltk.corpus import gutenberg
import numpy
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, GRU, SimpleRNN
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.cross_validation import train_test_split
import numpy as np
import climate

climate.enable_default_logging()

corpuses = [gutenberg.words()[:20000]]

data = []
for corpus in corpuses:
    for word in corpus:
        for character in word:
            if character.isalpha():
                data.append(character.lower())
        data.append(" ")

# create mapping of unique chars to integers
chars = sorted(set(data))
char_to_int = dict()

for i in range(0, len(chars)):
    first_part = np.array([0] * i)
    end_part = np.array([0] * (len(chars) - i - 1))
    char_to_int[chars[i]] = np.hstack((first_part, [1], end_part))

# summarize the loaded data
n_chars = len(data)
n_vocab = len(chars)
print "Total Characters: ", n_chars
print "Total Vocab: ", n_vocab

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []


def gen_sequences():
    for i in range(0, int((n_chars - seq_length)), 1):
        seq_in = data[i: i + seq_length]
        seq_out = data[i + seq_length]
        yield [char_to_int[char] for char in seq_in], char_to_int[seq_out]


cpt = 1
for gen in gen_sequences():
    dataX.append(gen[0])
    dataY.append(gen[1])

n_patterns = (n_chars - seq_length) * 1

print "Total Patterns: ", n_patterns
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, len(chars)))

# convert vector to numpy array
dataY = np.asarray(dataY)

# define the checkpoint
filepath = "./model-{epoch:02d}-{loss:.4f}-{acc:0.4f}.pkl"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# splitting
X_train, X_test, y_train, y_test = train_test_split(X, dataY, test_size=0.2)

# models
model = Sequential()
model.add(SimpleRNN(256, input_shape=(seq_length, len(chars))))
model.add(Dropout(0.2))
model.add(Dense(len(chars), activation='softmax'))
# adam adapts the learning rate dynamically
# categorical_crossentropy: best for 1 of m encoding outputs
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model_big = Sequential()
model_big.add(GRU(256, input_shape=(seq_length, len(chars)), return_sequences=True))
model_big.add(Dropout(0.2))
model_big.add(GRU(256))
model_big.add(Dropout(0.2))
model_big.add(Dense(len(chars), activation='softmax'))

model_big.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, nb_epoch=10, batch_size=32, validation_split=0.1, callbacks=callbacks_list)
print accuracy_score(y_test, model.predict(X_test))

model_big.fit(X_train, y_train, nb_epoch=10, batch_size=32, validation_split=0.1, callbacks=callbacks_list)
print accuracy_score(y_test, model_big.predict(X_test))
#
