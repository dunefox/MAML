import matplotlib.pyplot as plt
import numpy as np

wc = 70
eta = .16

def L(w):
    return 0.001*w**4 - 5*(w-5)**2 + 10000

def dw_L(w):
    return 4*0.001*w**3 - 2*5*(w-5)

def update(w):
    return w - eta * dw_L(w)

ws = [ wc ]
for _ in range(50):
    wn = update(wc)
    ws.append(wn)
    wc = wn

ws = np.array(ws)

plt.subplot(1, 2, 1)

w = np.linspace(-80, 80, 1000)
plt.plot(w, L(w))

x = ws
y = L(ws)
plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1, width=0.005)
plt.xlabel('w')
plt.ylabel('L(w)')

plt.subplot(1, 2, 2)
plt.plot(ws)
plt.xlabel('iteration')
plt.ylabel('w')

plt.show()

