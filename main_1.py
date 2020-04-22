import argparse

import random
from math import log, sqrt
import numpy as np
import matplotlib.pyplot as plt

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', dest='num', type=int, help="Number of Data Points")
    parser.add_argument('-t', dest='theta', type=int, nargs="+", help="Means and Variances")
    args = parser.parse_args()

    return args

# use Marsaglia polar method
# https://en.wikipedia.org/wiki/Normal_distribution#Generating_values_from_normal_distribution
def MP_method(mean, std):
    while True:
        u = random.uniform(-1, 1)
        v = random.uniform(-1, 1)
        if (u**2) + (v**2) < 1:
            break;

    s = (u**2) + (v**2)
    x = u * sqrt((-2) * log(s) / s)
    y = v * sqrt((-2) * log(s) / s)

    return mean + (std * x)

def create_data(mean_x, var_x, mean_y, var_y, num):
    Data = []
    for _ in range(num):
        x = MP_method(mean_x, var_x)
        y = MP_method(mean_y, var_y)
        Data.append([1, x, y])

    return Data

def gradient_descent(X, Y, Weight):
    return np.dot(np.transpose(X), (Y - 1 / (1 + np.exp(-np.dot(X, Weight)))))

def Hessian_matrix(Weight, X, num):
    Hessian = np.identity(num)
    for i in range(num):
        X_Phi = np.copy(X[i])
        exponential = np.exp(-np.dot(X_Phi , Weight))
        if np.isinf(exponential):
            exponential = np.exp(700)

        Hessian[i, i] = exponential / ((1 + exponential) ** 2)

    return np.dot(np.dot(np.transpose(X), Hessian), X)

def print_result(method, Weight, Predict, Label, num):
    print(f'{method}:\n')
    print('w:')
    print(f'{Weight}\n')
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(2*num):
        if Label[i] == 0 and Predict[i] == 0:
            tn += 1
        elif Label[i] == 0 and Predict[i] == 1:
            fp += 1
        elif Label[i] == 1 and Predict[i] == 0:
            fn += 1
        else:
            tp += 1

    print("confusion matrix:")
    print("                Predict cluster1      Predict cluster2")
    print(f"  Is cluster 1                {tn:2d}                    {fp:2d}")
    print(f"  Is cluster 2                {fn:2d}                    {tp:2d}")
    sensitivity = tn / (tn + fp)
    specificity = tp / (fn + tp)
    print("Sensitivity (Successfully predict cluster 1): ", sensitivity)
    print("Sepcificity (Successfully predict cluster 2): ", specificity)

def plot_result(Data, Gradient, Newton, num):
    plt.subplot(1,3,1)
    plt.title('Ground Truth')
    plt.scatter(Data[:num, 1], Data[:num, 2], s=15, c='r')
    plt.scatter(Data[num:, 1], Data[num:, 2], s=15, c='b')

    cluster1, cluster2 = list(), list()
    for i in range(2*num):
        if Gradient[i] == 0:
            cluster1.append(X[i])
        else:
            cluster2.append(X[i])

    cluster1 = np.array(cluster1)
    cluster2 = np.array(cluster2)

    plt.subplot(1, 3, 2)
    plt.title('Gradient descent')
    plt.scatter(cluster1[:, 1], cluster1[:, 2], s=15, c='r')
    plt.scatter(cluster2[:, 1], cluster2[:, 2], s=15, c='b')

    cluster1, cluster2 = list(), list()
    for i in range(2 * num):
        if Newton[i] == 0:
            cluster1.append(X[i])
        else:
            cluster2.append(X[i])

    cluster1 = np.array(cluster1)
    cluster2 = np.array(cluster2)

    plt.subplot(1, 3, 3)
    plt.title("Newton's Method")
    plt.scatter(cluster1[:, 1], cluster1[:, 2], s=15, c='r')
    plt.scatter(cluster2[:, 1], cluster2[:, 2], s=15, c='b')

if __name__ == "__main__":
    args = set_args()

    num = args.num
    mx1, vx1 = args.theta[0], args.theta[1]
    my1, vy1 = args.theta[2], args.theta[3]
    mx2, vx2 = args.theta[4], args.theta[5]
    my2, vy2 = args.theta[6], args.theta[7]

    Data1 = np.array(create_data(mx1, vx1, my1, vy1, num))
    Data2 = np.array(create_data(mx2, vx2, my2, vy2, num))
    X = np.concatenate((Data1, Data2))
    Y = np.array([[0] for _ in range(num)] + [[1] for _ in range(num)])

    Weight = np.zeros([3, 1])
    counter = 0
    while True:
        Gradient = gradient_descent(X, Y, Weight)
        new_Weight = Weight + Gradient

        if np.allclose(Weight, new_Weight) or counter>=10000:
            break

        counter += 1
        Weight = new_Weight

    Weight_Gradient = Weight

    Weight = np.zeros([3, 1])
    counter = 0
    while True:
        Hessian = Hessian_matrix(Weight, X, 2*num)
        Gradient = gradient_descent(X, Y, Weight)
        try:
            Hessian_inverse = np.linalg.inv(Hessian)
            new_Weight = Weight + np.dot(Hessian_inverse, Gradient)
        except np.linalg.LinAlgError as err:
            new_Weight = Weight + Gradient

        if np.allclose(Weight, new_Weight) or counter>=10000:
            break

        counter += 1
        Weight = new_Weight

    Weight_Newton = Weight

    Res_Gradient = 1 / (1 + np.exp(-np.dot(X, Weight_Gradient)))
    Predict_Gradient = [0 if predict<0.5 else 1 for predict in Res_Gradient]

    print_result('Gradient descent:', Weight_Gradient, Predict_Gradient, Y, num)
    print('\n----------------------------------------\n')

    Res_Newton = 1 / (1 + np.exp(-np.dot(X, Weight_Newton)))
    Predict_Newton = [0 if predict < 0.5 else 1 for predict in Res_Newton]

    print_result("Newton's method", Weight_Newton, Predict_Newton, Y, num)

    plot_result(X, Predict_Gradient, Predict_Newton, num)
    plt.show()

