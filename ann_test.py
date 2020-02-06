import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.models import load_model

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse


def parse_args():
    ''' Parse command line arguments '''
    p = argparse.ArgumentParser()
    p.add_argument("--numLayers", type=int, default=3, help="Number of hidden layers in the NN.")
    p.add_argument("--saveModel", default='save_ann.h5', help='Filepath to save the ANN in.')
    p.add_argument("--loadModel", default='NULL', help='Filepath to load tree from. By default, no tree will be loaded.')
    p.add_argument("--numRows", type=int, default=1000, help="Number of rows to create in the test data.")
    args = p.parse_args()
    return args




def ann(x, y, test_x, test_y, batch_size, epochs, num_layers, input_node_num, loadANN):
    # x is the training data array (shape should be ()# of examples, input_node_num))
    # y is the traing label array

    if loadANN != 'NULL':
        model = load_model(loadANN)
    else:
        model = Sequential()
        model.add(Dense(input_node_num, activation='sigmoid', input_shape=(input_node_num,)))
        for i in range(num_layers):
            model.add(Dense(input_node_num, activation='sigmoid'))
            model.add(Dropout(0.2))
        model.add(Dense(2, activation='sigmoid'))

        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])

        history = model.fit(x, y,
                            batch_size= batch_size,
                            epochs = epochs,
                            verbose = 1,
                            validation_data=(test_x, test_y))

    score = model.evaluate(test_x, test_y, verbose=1)
    return score, model


def main():
    '''Add a comment later'''
    args = parse_args()
    training = []
    training_labels = []
    val_data = []
    val_labels = []
    defaults = [[0, 0], [1, 0], [0, 1], [1, 1], [0, 1]]
    labs = [0, 1, 0, 1, 0]
    for i in range(args.numRows):
        training.append(defaults[i % 5])
        training_labels.append(labs[i % 5])
        val_data.append(defaults[i % 5])
        val_labels.append(labs[i % 5])


    x = np.asarray(training)
    y = np.asarray(training_labels)

    x_test =  np.asarray(val_data)
    y_test = np.asarray(val_labels)

    y = keras.utils.to_categorical(y, 2)
    y_test = keras.utils.to_categorical(y_test, 2)


    batch_size = 64
    epochs = 15
    scores, model = ann(x, y, x_test, y_test,
                batch_size, epochs, args.numLayers - 1, 2, args.loadModel)
    model.save(args.saveModel)
    print("\n\n", "-----"*30)
    print("With", args.numLayers - 1, "hidden layers, accuracy was:", scores[1])
    predictions = model.predict(x_test)
    isZero = False
    print(predictions[:10])

    if isZero:
        print("Oh good")
    else:
        print("Uh oh")


if __name__ == "__main__":
    main()
