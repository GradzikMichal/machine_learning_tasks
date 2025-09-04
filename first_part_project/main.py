import numpy
import pandas as pd
import numpy as np
from pathlib import Path
from time import time
from ActivationFunctions import ReLu, LogSoftMax, Sigmoid
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Input, Model
import matplotlib.pyplot as plt
from OurNeuralNetwork import OurNeuralNetwork


def transactions_csv():
    transactions = pd.read_csv('../resources/reduced_transactions.csv',
                               usecols=['transaction_date', 'customer_id', 'product_code'])
    n_rows = transactions['customer_id'].unique().size
    n_cols = transactions['product_code'].unique().size
    article_indices = pd.DataFrame(index=np.arange(n_cols), data=transactions['product_code'].unique())
    customer_indices = pd.DataFrame(index=transactions['customer_id'].unique(), data=np.arange(n_rows))
    with open('transaction_matrix.npy', 'wb') as all_items, open('transaction_matrix_train.npy', 'wb') as train_items:
        for customer_id in customer_indices.index:
            print(customer_id)
            testData = np.zeros(n_cols, dtype=numpy.int8)
            trainData = np.zeros(n_cols, dtype=numpy.int8)
            bought_items_data = transactions[transactions['customer_id'] == customer_id]
            bought_sorted = bought_items_data.sort_values(by='transaction_date')['product_code'].values
            recently_bought = bought_sorted[-3:]
            for article_id, article_code in article_indices.iterrows():
                if article_code[0] in bought_sorted:
                    testData[int(article_id)] = 1
                    if article_code[0] not in recently_bought:
                        trainData[int(article_id)] = 1
            np.save(all_items, testData)
            np.save(train_items, trainData)
        all_items.close()
        train_items.close()


if __name__ == "__main__":
    X = np.load('X.npy')
    x_maxs = np.amax(X, axis=1)
    X = X.T/x_maxs.T
    X = X.T
    Y = np.load('Y.npy')
    #print(list(Y[0]))
    print(len(X[0]))
    print(len(Y[0]))
    myNN = OurNeuralNetwork(
        input_layer_size=len(X[0]),
        output_layer_size=len(Y[0]),
        hidden_layers=np.array([264]),
        active_function=[ReLu(), LogSoftMax()],
        learning_rate=9e-1,
        verbose=False
    )

    x_train = X[:5000]
    y_train = Y[:5000]
    x_test = X[5000:]
    y_test = Y[5000:]
    print(len(x_train[0]))
    print(len(y_train[0]))

    myNN2 = OurNeuralNetwork(
        input_layer_size=len(X[0]),
        output_layer_size=len(Y[0]),
        hidden_layers=np.array([528, 1000, 132]),
        active_function=[ReLu(), ReLu(), ReLu(), LogSoftMax()],
        learning_rate=7e-1,
        verbose=False
    )
    history2 = myNN.gradientDescent(
        X=x_train,
        Y=y_train,
        iterations=10000
    )

    plt.plot(history2)
    score, mean, predictions = myNN.test(x_test, y_test)
    print(score)
    print(mean)
