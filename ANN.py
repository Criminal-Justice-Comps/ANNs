"""
ANN.py
Kellen Dorchen and Cameron Kline-Sharpe
Jan/Feb 2020

This file creates or loads a feed forward neural network with
a variable number of hidden layers

Usage:
    python3 ANN.py filename [optional arguments]
Where "filename" is the path to a .csv file (use TFV.csv for now)
The file should be a csv with the following properties:
    No column or row headers
    The last column (and only that column) contains the ground truth
    All entries are numbers.
        For numeric features, this is easy.
        For categorical features, simply assign each category to a positive integer
The [optional arguments] are described in parse_args.
Note that the default values are not optimized for model correctness, but rather they
offer an idea of what the model looks like without taking a lot of time to run.


The --loadModel command allows the user to load a pretrained network. Possible uses include:
    * loading a model that took a long time to train
    * testing a pre-optimized model
    * etc.

The predictions of the final model on the validation data are
saved in the file specified by the --saveOutput argument. This data is stored in
a comma seperated file as follows:
    prediction,truth                    ## headers
    pred1, truth1                       ## prediction for person 1, truth for person 1
    pred2, truth2
      .      .
      .      .
      .      .
And so on.

The confusion matrix is also printed out.


TODO:
    Potential problem: always predict negative for violent Recidivism
"""



import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.models import load_model

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse
import csv
import json


def parse_args():
    ''' Parse command line arguments '''
    p = argparse.ArgumentParser()
    p.add_argument("trainingData", help="Path to csv containing training and validation data sets.")
    p.add_argument("--numLayers", type=int, default=3, help="Number of hidden layers in the NN. This cannot be 0.")
    p.add_argument("--saveModel", default='save_ann.h5', help='Filepath to save the ANN in.')
    p.add_argument("--loadModel", default='NULL', help='Filepath to load tree from. By default, no tree will be loaded.')
    p.add_argument("--saveOutput", default='ANNPred.csv', help='Filepath to save the ANN in.')
    p.add_argument("--trainSize", type=int, default=8000, help="Number of training examples to use. Validation set will have 9861-trainSize examples")
    p.add_argument("--inputNodes", type=int, default=14, help="The number of nodes in the input layer. For TVF.csv, this is 7.")
    p.add_argument("--epochs", type=int, default=10, help="The number of times to iterate through the data when training the network.")
    p.add_argument("--batchSize", type=int, default=64, help="The batch size to use when training the network.")
    p.add_argument("--layerSize", type=int, default=14, help="The number of nodes each hidden layer should have.")
    p.add_argument("--guessViolent", action="store_true", default=False, help="Predict Violent Recidivism rather than Recidivism.")
    args = p.parse_args()
    return args

def get_data(filepath, train_test_split, check_violent):
    '''Returns a list of all data points and all data labels, split into training
        and testing sets.'''
    training = []
    training_labels = []
    val_data = []
    val_labels = []

    ground_truth = -1
    to_use = []
    with open(filepath) as file:
        i = 0
        for line in file:
            split_line = line.split(",")
            if i == 0 :
                features = split_line

                # get the correct index of the ground truth to guess
                ground_truth = features.index('is_recid')
                if check_violent:
                    ground_truth = features.index('is_violent_recid\n')

            elif i < train_test_split:
                person = []
                for element in split_line[:-2]:
                    person.append(int(element))
                training.append(person)
                training_labels.append(int(split_line[ground_truth]))
            else:
                person = []
                for element in split_line[:-2]:
                    person.append(int(element))

                val_data.append(person)
                val_labels.append(int(split_line[ground_truth]))
            i += 1
    return np.asarray(training), np.asarray(training_labels), np.asarray(val_data), np.asarray(val_labels)
#numLayers17layerSize16epochs50

def ann(x, y, test_x, test_y, batch_size, epochs, num_layers, input_node_num, layer_size, loadANN):
    # x is the training data array (shape should be ()# of examples, input_node_num))
    # y is the traing label array
    model = None
    score = None
    if loadANN != 'NULL':
        model = load_model(loadANN)
        if not len(x) == 0 and not len(test_x) == 0:
            x = np.concatenate((x, test_x), axis=0)
        if not len(y) == 0 and not len(test_y) == 0:
            y = np.concatenate((y, test_y), axis=0)
        score = model.evaluate(x, y, verbose=1)
    else:
        model = Sequential()
        model.add(Dense(input_node_num, activation='relu', input_shape=(input_node_num,)))
        for i in range(num_layers):
            model.add(Dense(layer_size, activation='relu'))
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

def create_json(file, saveOutput):
    data_dictionary = {}
    people = []
    with open(saveOutput, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if row["sex"] == "0.0":
                data_dictionary["sex"] = "Female"
            elif row["sex"] == "1.0":
                data_dictionary["sex"] = "Male"

            if row["race"] == "0.0":
                data_dictionary["race"] = "Other"
            elif row["race"] == "1.0":
                data_dictionary["race"] = "African-American"
            elif row["race"] == "2.0":
                data_dictionary["race"] = "Caucasian"
            elif row["race"] == "3.0":
                data_dictionary["race"] = "Hispanic"
            elif row["race"] == "4.0":
                data_dictionary["race"] = "Native American"
            elif row["race"] == "5.0":
                data_dictionary["race"] = "Asian"

            data_dictionary["age"] = row["age"]
            data_dictionary["prediction"] = row["prediction"]
            data_dictionary["truth"] = row["truth"]
            people.append(data_dictionary)
            data_dictionary = {}
        people_alg_dict = {"people":people, "ANN":[]}
    #
    # with open("../Fairness/ANNPredictions.json", 'w') as file:
    #         json.dump(people_alg_dict, file)


def main():
    '''Add a comment later'''
    args = parse_args()
    training, training_labels, val_data, val_labels = get_data(args.trainingData, args.trainSize, args.guessViolent)
    training_labels = keras.utils.to_categorical(training_labels, 2)
    val_labels = keras.utils.to_categorical(val_labels, 2)
    training = training.astype('float32')
    val_data = val_data.astype('float32')
    scores, model = ann(training, training_labels, val_data, val_labels,
                args.batchSize, args.epochs, args.numLayers - 1,
                args.inputNodes, args.layerSize, args.loadModel)
    model.save(args.saveModel)


    print("\n\n", "-----"*30)
    print("With", args.numLayers, "hidden layers, accuracy was:", scores[1])

    if args.loadModel == "NULL":
        predictions = model.predict(val_data)
        predictions = np.ndarray.tolist(predictions)
        y = np.ndarray.tolist(val_labels)
        x = np.ndarray.tolist(val_data)
    else:
        x = training
        y = training_labels
        if not len(training) == 0 and not len(val_data) == 0:
            x = np.concatenate((x, val_data), axis=0)
        if not len(training_labels) == 0 and not len(val_labels) == 0:
            y = np.concatenate((y, val_labels), axis=0)
        predictions = model.predict(x)
        predictions = np.ndarray.tolist(predictions)
        y = np.ndarray.tolist(y)
        x = np.ndarray.tolist(x)


    confusion = confusion_matrix(predictions, y)
    print("---"*30)
    print("""                                       Truth
                                1                   0

                          1     {}               {}
        Prediction
                          0     {}               {}
        """.format(confusion[0],confusion[1],confusion[2],confusion[3]))
    string = 'person_id,Male,Female,African-American,Other,Caucasian,Hispanic,Native American,Asian,age,juv_fel_count,juv_misd_count,juv_other_count,priors_count,prediction,truth\n'
    for i in range(len(predictions)):
        for el in x[i]:
            string += str(el)
            string += ","
        string += str(predictions[i].index(max(predictions[i])))
        string += ","
        string += str(y[i].index(1.0))
        string += '\n'
    string = string[:-1]

    with open(args.saveOutput, 'w') as file:
        file.write(string)

    #create_json(file, args.saveOutput)

def confusion_matrix(predictions, truths):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(predictions)):
        pred = predictions[i].index(max(predictions[i]))
        truth = truths[i].index(max(truths[i]))
        #print(predictions[i], truths[i])
        if pred == 1:
            if pred == truth:
                tp += 1
            else:
                fp += 1
        else:
            if pred == truth:
                tn += 1
            else:
                fn += 1
    return [tp, fp, fn, tn]


if __name__ == "__main__":
    main()
