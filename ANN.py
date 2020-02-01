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

def get_data(trainingData, valData):
    '''Returns a list of all data points and all data labels, split into training
        and testing sets.'''

    features = []
    features_with_values = {}
    data = []
    training_labels = []
    with open(trainingData) as file:
        firstLine = True
        new_index = 0
        for line in file:
            split = line.split(",")
            if firstLine:
                index_to_features = split[1:]
                for feature in index_to_features:
                    if index_to_features.index(feature) in NUMERIC_FEATURES:
                        features_with_values[index_to_features.index(feature)] = new_index
                        new_index += 1
                    else:
                        features_with_values[index_to_features.index(feature)] = {}
                firstLine = False
            else:
                person = split[1:-1]
                for index, value in enumerate(person):
                    if not index in NUMERIC_FEATURES:
                        if value not in features_with_values[index]:
                            features_with_values[index][value] = new_index
                            new_index += 1
                label = int(split[-1])
                data.append(person)
                training_labels.append(label)

    val_data = []
    val_labels = []
    #initially load validation data
    with open(valData) as file:
        firstLine = True
        for line in file:
            if firstLine:
                firstLine = False
                continue
            else:
                split = line.split(",")
                person = split[1:-1]
                for index, value in enumerate(person):
                    if not index in NUMERIC_FEATURES:
                        if value not in features_with_values[index]:
                            features_with_values[index][value] = new_index
                            new_index += 1
                label = int(split[-1])
                val_data.append(person)
                val_labels.append(label)


    training = []
    for index, person in enumerate(data):
        example = [0]*(new_index)
        for index2, value in enumerate(person):
            if index2 in NUMERIC_FEATURES:
                example[features_with_values[index2]] = int(value)
            else:
                example[features_with_values[index2][value]] = 1
        training.append(example)


    validation = []
    for person in val_data:
        example = [0]*(new_index)
        for index4, value in enumerate(person):
            if index4 in NUMERIC_FEATURES:
                example[features_with_values[index4]] = int(value)
            else:
                example[features_with_values[index4][value]] = 1
        validation.append(example)



    return np.asarray(training), np.asarray(training_labels), np.asarray(validation), np.asarray(val_labels), index_to_features, features_with_values, new_index


def ann(x, y, test_x, test_y, batch_size, epochs, num_layers, input_node_num, loadANN):
    # x is the training data array (shape should be ()# of examples, input_node_num))
    # y is the traing label array
    if loadANN != 'NULL':
        model = load_model(loadANN)
    else:
        model = Sequential()
        model.add(Dense(input_node_num, activation='relu', input_shape=(input_node_num,)))
        for i in range(num_layers):
            model.add(Dense(input_node_num, activation='relu'))
        model.add(Dense(1, activation='softmax'))

        model.summary()

        model.compile(loss='mean_squared_error',
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
    training, training_labels, val_data, val_labels, index_to_features, features_with_values, input_node_num = get_data(args.trainingData, args.valData)
    batch_size = 64
    epochs = 5
    scores, model = ann(training, training_labels, val_data, val_labels,
                batch_size, epochs, args.numLayers - 1, input_node_num, args.loadModel)
    model.save(args.saveModel)
    print("\n\n", "-----"*30)
    print("With", args.numLayers - 1, "hidden layers, accuracy was:", scores[1])
    predictions = model.predict(val_data)

    isZero = False
    for pred in predictions:
        if 0.0 in pred:
            isZero = True

    if isZero:
        print("Oh good")
    else:
        print("Uh oh")













if __name__ == "__main__":
    main()
