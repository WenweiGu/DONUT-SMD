import warnings
import numpy as np
import pandas as pd
from donut import complete_timestamp, standardize_kpi
import os

warnings.filterwarnings("ignore")

data_dir_path = 'C:/Users/Administrator/Downloads/research/donut-master/SMD/data_concat/'
csvs = os.listdir(data_dir_path)

csv_path = []

for i in csvs:
    csv_path.append(data_dir_path + i)

numbers = []

for j in csvs:
    name_temp = os.path.split(j)[1]
    numbers.append(name_temp[5:-4])


# print(numbers)


def generate_score(number):
    # Read the raw data.
    data_dir_path = 'C:/Users/Administrator/Downloads/research/donut-master/SMD/data_concat/data-' + number + '.csv'
    data = np.array(pd.read_csv(data_dir_path, header=None), dtype=np.float64)
    tag_dir_path = './SMD/test_label/machine-' + number + '.csv'
    tag = np.array(pd.read_csv(tag_dir_path, header=None), dtype=np.int)
    labels = np.append(np.zeros(int(len(data) / 2)), tag)
    # pick one colume
    values = data[:, 1]
    timestamp = np.arange(len(data)) + 1

    # If there is no label, simply use all zeros.
    # labels = np.zeros_like(values, dtype=np.int32)

    # Complete the timestamp, and obtain the missing point indicators.
    timestamp, (values, labels) = \
        complete_timestamp(timestamp, (values, labels))

    # Split the training and testing data.
    test_portion = 0.5
    test_n = int(len(values) * test_portion)
    train_values = values[:-test_n]
    test_values = values[-len(train_values):]
    train_labels, test_labels = labels[:-test_n], labels[-test_n:]
    # print(len(test_values), len(test_labels))

    # Standardize the training and testing data.
    train_values, mean, std = standardize_kpi(
        train_values, excludes=train_labels)
    test_values, _, _ = standardize_kpi(test_values, mean=mean, std=std)

    import tensorflow as tf
    from donut import Donut
    from tensorflow import keras as K
    from tfsnippet.modules import Sequential

    # We build the entire model within the scope of `model_vs`,
    # it should hold exactly all the variables of `model`, including
    # the variables created by Keras layers.
    with tf.variable_scope('model') as model_vs:
        model = Donut(
            h_for_p_x=Sequential([
                K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                               activation=tf.nn.relu),
                K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                               activation=tf.nn.relu),
            ]),
            h_for_q_z=Sequential([
                K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                               activation=tf.nn.relu),
                K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                               activation=tf.nn.relu),
            ]),
            x_dims=120,
            z_dims=5,
        )

    from donut import DonutTrainer, DonutPredictor

    trainer = DonutTrainer(model=model, model_vs=model_vs)
    predictor = DonutPredictor(model)

    with tf.Session().as_default():
        trainer.fit(train_values, train_labels, mean, std)
        test_score = predictor.get_score(test_values)

    if not os.path.exists('./score'):
        os.makedirs('./score')

    np.save('./score/' + number + '.npy', test_score)

    # print(len(test_score))


for j in numbers:
    generate_score(j)
    print('Finish generating' + j)
