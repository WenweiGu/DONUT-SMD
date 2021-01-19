import numpy as np
import pandas as pd
import os
from sklearn.metrics import f1_score

data_dir_path = 'C:/Users/Administrator/Downloads/research/donut-master/SMD/data_concat/'
csvs = os.listdir(data_dir_path)

csv_path = []

for i in csvs:
    csv_path.append(data_dir_path + i)

numbers = []

for j in csvs:
    name_temp = os.path.split(j)[1]
    numbers.append(name_temp[5:-4])

def adjust_predicts(score, label, percent=None,
                    pred=None,
                    threshold=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is higher than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
        :param pred:
        :param label:
        :param score:
        :param calc_latency:
        :param threshold:
        :param percent:
    """
    if score is not None:
        if len(score) != len(label):
            raise ValueError("score and label must have the same length")
        score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    predict = []
    if pred is None:
        if percent is not None:
            threshold = np.percentile(score, percent)
            predict = score > threshold
        elif threshold is not None:
            predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for k in range(len(predict)):
        if actual[k] and predict[k] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(k, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[k]:
            anomaly_state = False
        if anomaly_state:
            predict[k] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def __iter_thresholds_without_adjust(score, label):
    best_f1 = -float("inf")
    best_theta = None
    for anomaly_ratio in np.linspace(1e-4, 10, 500):
        pred = anomaly_ratio < score

        f1 = f1_score(pred, label)
        if f1 > best_f1:
            best_f1 = f1
            best_theta = anomaly_ratio
    return best_f1, best_theta


def __iter_thresholds(score, label):
    best_f1 = -float("inf")
    best_theta = None
    best_adjust = None
    for anomaly_ratio in np.linspace(1e-3, 0.3, 50):
        adjusted_anomaly = adjust_predicts(
            score, label, percent=100 * (1 - anomaly_ratio)
        )
        f1 = f1_score(adjusted_anomaly, label)
        if f1 > best_f1:
            best_f1 = f1
            best_adjust = adjusted_anomaly
            best_theta = anomaly_ratio
    return best_f1, best_theta, best_adjust


def evaluate(number):
    raw_score = np.load('./score/' + number + '.npy')
    #print(score.size)
    #score = raw_score
    score = (raw_score - min(raw_score)) / (max(raw_score) - min(raw_score))
    score = 1 - score
    tag_dir_path = './SMD/test_label/machine-' + number + '.csv'
    tag = np.array(pd.read_csv(tag_dir_path, header=None), dtype=np.int)
    labels = tag[-len(score):]
    #print(len(labels))
    print('adjust', __iter_thresholds(score, labels)[0], 'without adjust', __iter_thresholds_without_adjust(score, labels)[0])
    return __iter_thresholds(score, labels)[0], __iter_thresholds_without_adjust(score, labels)[0]


index = []
f1_with_adjust = []
f1_without_adjust = []

for i in numbers:
    index.append(i)
    f1_with_adjust.append(round(evaluate(i)[0], 2))
    f1_without_adjust.append(round(evaluate(i)[1], 2))
    print('Evaluate %s finished' % i)

csvFile = open('C:/Users/Administrator/Downloads/research/donut-master/performance-normalize.csv', 'w+', newline='')
f1_data = pd.DataFrame({'Series number': index, 'F1 with adjust': f1_with_adjust,
                        'F1 without adjust': f1_without_adjust})
f1_data.set_index('Series number')
f1_data.to_csv(csvFile)
