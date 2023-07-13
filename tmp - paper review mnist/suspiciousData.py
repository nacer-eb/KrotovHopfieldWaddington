import numpy as np
import matplotlib.pyplot as plt

data_1 = np.loadtxt("LearningTimes_1.csv", delimiter=",")
data_2 = np.loadtxt("LearningRates_1.csv", delimiter=",")

print(np.shape(data_1))
print(np.shape(data_2))


plt.plot(data_1[:-2, 0], (data_1[:-2, 1]*data_2[:, 1]), "-.")
plt.show()
