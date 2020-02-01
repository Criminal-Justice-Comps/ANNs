'''import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop'''

from collections import defaultdict
import matplotlib.pyplot as plt
import argparse

NUMERIC_FEATURES = [1]

def parse_args():
    ''' Parse command line arguments '''
    p = argparse.ArgumentParser()
    p.add_argument("trainingData", help="Path to csv containing training data set.")
    p.add_argument("valData", help="Path to csv file containing validation data set.")
    #p.add_argument("--hideTree", action="store_true", help="Turn off the display of all trees.")
    #p.add_argument("--saveTree", default='save_tree.txt', help='Filepath to save the tree in.')
    #p.add_argument("--loadTree", default='NULL', help='Filepath to load tree from. By default, no tree will be loaded.')
    #p.add_argument("--trainCount", type=int, default=100, help='Number of data points to use as the training set.')
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

    training = []
    for index, person in enumerate(data):
        example = [0]*(new_index)
        for index2, value in enumerate(person):
            if index2 in NUMERIC_FEATURES:
                example[features_with_values[index2]] = int(value)
            else:
                example[features_with_values[index2][value]] = 1
        training.append(example)

    val_data = []
    val_labels = []
    with open(valData) as file:
        firstLine = True
        for line in file:
            if firstLine:
                firstLine = False
                continue
            else:
                split = line.split(",")
                person = split[1:-1]
                example = [0]*(new_index)
                for index, value in enumerate(person):
                    if index in NUMERIC_FEATURES:
                        example[features_with_values[index]] = int(value)
                    else:
                        example[features_with_values[index][value]] = 1
                label = int(split[-1])
                val_data.append(example)
                val_labels.append(label)


    return training, training_labels, val_data, val_labels, index_to_features, features_with_values, new_index


def main():
    '''Add a comment later'''
    args = parse_args()
    training, training_labels, index_to_features, val_data, val_labels, features_with_values, input_node_num = get_data(args.trainingData, args.valData)
    batch_size = 128
    num_classes = 10
    epochs = 2















if __name__ == "__main__":
    main()
