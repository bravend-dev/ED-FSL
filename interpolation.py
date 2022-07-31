import numpy as np

N = 5
epsilon = 0.2
k = np.log(epsilon * (N - 1) + 1) - np.log(1 - epsilon)
print(k)
for x in range(100):
    a = np.random.uniform(0, k, (1, N))
    # print(a)
    expa = np.exp(a)
    prop = expa / expa.sum()
    # print(prop)

    print(prop.max() - prop.min())