import struct

import random
from math import log, sqrt
import numpy as np
import numba as nb
import matplotlib.pyplot as plt

train_image_path = './train-images-idx3-ubyte'
train_label_path = './train-labels-idx1-ubyte'

def read_file(path, type):
    if type == 'image':
        with open(path, 'rb') as image:
            magic_num, data_num, rows, cols = struct.unpack('>IIII', image.read(16))
            images = np.fromfile(image, dtype =np.uint8).reshape(data_num, 784)
        return images
    elif type == 'label':
        with open(path, 'rb') as label:
            magic, data_num = struct.unpack('>II', label.read(8))
            labels = np.fromfile(label, dtype=np.uint8)
        return labels

def plot_mnist(Result):
    for label in range(10):
        print(f'{label}:')
        row = ''
        for pixel in range(784):
            if pixel % 28 == 0:
                row = ''
            grey = "1 " if Result[pixel][label] >= 0.5 else "0 "
            row += grey
            if pixel % 28 == 27:
                print(row)
        print()

# do the EM Algorithm
def EM_algo(Data, lamb, P):
    Weight = E_step(Data, lamb, P)
    lamb, P = M_step(Data, lamb, P, Weight)
    plot_mnist(P)

    return np.reshape(lamb, (1, 10)), P

@nb.jit()
def E_step(Data, lamb, P):
    weight = np.full((60000, 10), 1, dtype=np.float64)  # 60000 * 10
    temp = 0
    for num in range(60000):
        #temp = 0
        for label in range(10):
            for index, pixel in enumerate(Data[num]):
                if Data[num][index] == 1:
                    weight[num][label] *= P[index][label]
                else:
                    weight[num][label] *= (1 - P[index][label])
            weight[num][label] *= lamb[0][label]
            temp += weight[num][label]
        if temp == 0:
            temp = 1
        weight[num] = weight[num] / temp

    return weight

@nb.jit()
def M_step(Data, lamb, P, W):
    lamb = np.sum(W, axis=0) / len(W)
    probability = np.zeros((28 * 28, 10)).astype(np.float64)

    for label in range(10):
        for pixel in range(28*28):
            probability[pixel][label] = np.dot(np.transpose(Data)[pixel], np.transpose(W)[label])
            temp = np.sum(W, axis=0)[label]
            if temp == 0:
                temp = 1
            probability[pixel][label] /= temp
    return lamb, probability

@nb.jit
def decide_label(Data, Labels, lamb, P):
    mapping = np.zeros(shape=(10, 10), dtype=np.int)
    relation = np.full((1, 10), -1, dtype=np.int)
    predict = np.full((1, 60000), 0, dtype=np.int)

    for num in range(60000):
        temp = np.zeros(shape=10, dtype=np.float64)
        for label in range(10):
            accu = float(1)
            for pixel in range(28 * 28):
                if Data[num][pixel] == 1:
                    accu *= P[pixel][label]
                else:
                    accu *= (1 - P[pixel][label])
            temp[label] = lamb[0][label] * accu
        mapping[Labels[num]][np.argmax(temp)] += 1
        predict[num] = np.argmax(temp)

    for i in range(1, 11):
        ind = np.unravel_index(np.argmax(mapping, axis=None), mapping.shape)
        relation[0][ind[0]] = ind[1]
        for j in range(0, 10):
            mapping[ind[0]][j] = -1 * i
            mapping[j][ind[1]] = -1 * i

    for num in range(60000):
        for i in range(0, 10):
            if relation[0][i] == predict[num]:
                predict[num] = i
                break

    return relation, predict

def plot_label(Result, relation):
    for label in range(10):
        label_related = relation[0][label]
        print(f'labeled class {label}:')
        row = ''
        for pixel in range(784):
            if pixel % 28 == 0:
                row = ''
            grey = "1 " if Result[pixel][label_related] >= 0.5 else "0 "
            row += grey
            if pixel % 28 == 27:
                print(row)
        print()


def confusion_matrix(Labels, Predicts):
    error = 0

    for label in range(10):
        tp, fp, fn, tn = 0, 0, 0, 0
        for num in range(60000):
            if Labels[num] == label:
                if Predicts[num] == label:
                    tp += 1
                else:
                    fp += 1
                    error += 1
            else:
                if Predicts[num] == label:
                    fn += 1
                    error += 1
                else:
                    tn += 1

        print("\n---------------------------------------------------------------\n")
        print(f"Confusion Matrix {label}:")
        print(f"                Predict number {label}      Predict not number {label}")
        print(f"Is number {label}                {tp:5d}                    {fp:5d}")
        print(f"Isn't number {label}             {fn:5d}                    {tn:5d}")
        sensitivity = tp / (tp + fp)
        specificity = tn / (fn + tn)
        print(f"Sensitivity (Successfully predict number {label}): ", sensitivity)
        print(f"Sepcificity (Successfully not predict number {label}): ", specificity)

    return error

if __name__ == "__main__":
    train_images = read_file(train_image_path, 'image') # 60000 * 784
    train_labels = read_file(train_label_path, 'label') # 60000 * 1
    train_images = train_images // 128

    lamb = np.full((1, 10), 0.1, dtype=np.float64)  # 1 * 10
    probability = np.random.rand(28 * 28, 10).astype(np.float64)    # 784 * 10
    pre_probability = np.copy(probability)
    iteration = 0

    lamb, probability = EM_algo(train_images, lamb, probability)
    while True:
        difference = np.sum(np.sum(np.abs(probability - pre_probability), axis=0))
        iteration += 1
        print(f"No. of Iteration: {iteration}, Difference: {difference}\n")
        print('------------------------------\n')

        if np.allclose(probability, pre_probability):
            break
        pre_probability = np.copy(probability)
        lamb, probability = EM_algo(train_images, lamb, probability)

    relation, predict = decide_label(train_images, train_labels, lamb, probability)
    plot_label(probability, relation)

    error = confusion_matrix(train_labels, predict)
    print(f"Total iteration to converge: {iteration}")
    print(f"Total error rate: {error / 600000}")