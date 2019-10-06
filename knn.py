import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#train = pd.read_csv('Data_csv.csv').values
train = pd.read_csv('train.txt', sep=',', header=None, dtype='Int64').values
train_len = train[:, 0].size

class_1 = []
class_2 = []
for i in range(train_len):
    if train[i, 2] == 1:
        class_1.append(train[i, :])
    else:
        class_2.append(train[i, :])

class_1 = np.array(class_1)
class_2 = np.array(class_2)

x_1 = class_1[:, 0]
y_1 = class_1[:, 1]

x_2 = class_2[:, 0]
y_2 = class_2[:, 1]

test = pd.read_csv('test.txt', sep=',', header=None, dtype='Int64').values
#test = pd.read_csv('test_csv.txt', sep=',', header=None, dtype='float64').values

test_len = test[:, 0].size

file1 = open('prediction1.txt', 'w')

k = int(input("the value of k: "))
predicted_1 = []
predicted_2 = []

tp = 0
fp = 0
tn = 0
fn = 0

for i in range(test_len):
    classify = []
    x1 = test[i][0]
    y1 = test[i][1]

    file1.write(f"the test point is: {x1}, {y1} \n")

    for j in range(train_len):
        x2 = train[j][0]
        y2 = train[j][1]
        classify.append([(x2-x1)**2 + (y2-y1)**2, train[j][2]])
    classify = sorted(classify, key=lambda x: x[0])
    classify = np.array(classify)

    count_1 = 0
    count_2 = 0

    for l in range(k):
        file1.write(f"Distance {l + 1}:{classify[l, 0]}\tclass: {classify[l, 1]}\n")
        if classify[l][1] == 1:
            count_1 += 1
        else:
            count_2 += 1

    if count_1 > count_2:
        predicted_1.append(test[i, :])
        # if test[i][2] == 1:
        #     tp += 1
        # else:
        #     fp += 1
        file1.write(f"Predicted Class: 1\n\n")
    else:
        predicted_2.append(test[i, :])
        # if test[i][2] == 2:
        #     tn += 1
        # else:
        #     fn += 1
        file1.write(f"Predicted Class: 2\n\n")

file1.close()

predicted_1 = np.array(predicted_1)
predicted_2 = np.array(predicted_2)

nx_1 = predicted_1[:, 0]
ny_1 = predicted_1[:, 1]
nx_2 = predicted_2[:, 0]
ny_2 = predicted_2[:, 1]

plt.scatter(x_1, y_1, c='b', marker='*', label='class 1')
plt.scatter(x_2, y_2, c='r', marker='+', label='class 2')

plt.scatter(nx_1, ny_1, c='y', marker='o', label='predicted class 1')
plt.scatter(nx_2, ny_2, c='g', marker='^', label='predicted class 2')
plt.legend()
plt.show()

# recall = tp/(tp+fn)
# precision = tp/(tp+fp)
# acc = ((tp+tn)/(tp+tn+fp+fn))*100
# f = (2*precision*recall)/(precision+recall)
#
# print(f"Recall {recall}")
# print(f"precision {precision}")
# print(f"accuracy {acc}%")
# print(f"f measure {f}")

