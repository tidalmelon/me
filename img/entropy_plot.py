# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 100)
x2 = 1-x
tt = np.linspace(2, 2, 100)
y = -x*np.log(x)/np.log(tt) -x2*np.log(x2)/np.log(tt)

plt.figure(figsize=(6, 6))
plt.plot(x, y, color="red", linewidth=2, label='')
plt.xlabel("p")
plt.ylabel("-plogp")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend()
plt.show()
