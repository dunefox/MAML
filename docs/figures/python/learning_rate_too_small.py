import matplotlib.pyplot as plt
import numpy as np

wc = 80
eta = .01

def L(w):
    return 0.001*w**4 - 5*(w-5)**2

def dw_L(w):
    return 4*0.001*w**3 - 2*5*(w-5)

def update(w):
    return w - eta * dw_L(w)

ws = [ wc ]
for _ in range(4):
    wn = update(wc)
    ws.append(wn)
    wc = wn

ws = np.array(ws)

w = np.linspace(-100, 100, 1000)
plt.plot(w, L(w))

x = ws
y = L(ws)
plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1, width=0.005)
plt.xlabel('w')
plt.ylabel('L(w)')

plt.show()
