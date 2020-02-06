import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.models import load_model


import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse

NUMERIC_FEATURES = [2, 3, 4, 5, 6]

def parse_args():
    ''' Parse command line arguments '''
    p = argparse.ArgumentParser()
    p.add_argument("trainingData", help="Path to csv containing training data set.")
    p.add_argument("valData", help="Path to csv file containing validation data set.")
    p.add_argument("--numLayers", type=int, default=3, help="Number of hidden layers in the NN.")
    p.add_argument("--saveModel", default='save_ann.h5', help='Filepath to save the ANN in.')
    p.add_argument("--loadModel", default='NULL', help='Filepath to load tree from. By default, no tree will be loaded.')
    args = p.parse_args()
    return args

def get_data(filepath, train_test_split):
    '''Returns a list of all data points and all data labels, split into training
        and testing sets.'''
    with open(filepath) as file:
        training = []
        training_labels = []
        testing = []
        testing_labels = []
        i = 0
        for line in file:
            split_line = line.split(",")
            if i < train_test_split:
                training.append([int(el) for el in split_line[:-1]])
                training_labels.append(int(split_line[-1]))
            else:
                testing.append([int(el) for el in split_line[:-1]])
                testing_labels.append(int(split_line[-1]))
            i += 1
    return np.asarray(training), np.asarray(training_labels), np.asarray(testing), np.asarray(testing_labels)


def ann(x, y, test_x, test_y, batch_size, epochs, num_layers, input_node_num, loadANN):
    # x is the training data array (shape should be ()# of examples, input_node_num))
    # y is the traing label array
    if loadANN != 'NULL':
        model = load_model(loadANN)
    else:
        model = Sequential()
        model.add(Dense(input_node_num, activation='relu', input_shape=(input_node_num,)))
        for i in range(num_layers):
            model.add(Dense(100, activation='relu'))
        model.add(Dense(2, activation='softmax'))

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
    training, training_labels, val_data, val_labels = get_data(args.trainingData, 2000)
    training_labels = keras.utils.to_categorical(training_labels, 2)
    val_labels = keras.utils.to_categorical(val_labels, 2)
    training = training.astype('float32')
    val_data = val_data.astype('float32')
    batch_size = 64
    epochs = 20
    scores, model = ann(training, training_labels, val_data, val_labels,
                batch_size, epochs, args.numLayers - 1, 7, args.loadModel)
    model.save(args.saveModel)
    print("\n\n", "-----"*30)
    print("With", args.numLayers - 1, "hidden layers, accuracy was:", scores[1])
    predictions = np.ndarray.tolist(model.predict(training))
    tr_label_list = np.ndarray.tolist(training_labels)

    string = ''
    for i in range(len(training)):
        for el in training[i]:
            string += str(el)
            string += ","
        string += str(predictions[i].index(max(predictions[i])))
        string += ","
        string += str(tr_label_list[i].index(1.0))
        string += '\n'
    string = string[:-1]

    with open("ANNTrainingPred.csv", 'w') as file:
        file.write(string)












if __name__ == "__main__":
    main()
