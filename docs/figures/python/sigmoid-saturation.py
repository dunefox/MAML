import numpy as np
import matplotlib.pyplot as plt

def alpha(z):
    return z

def dz_alpha(z):
    return 1

def L(w):
    return 1/2 * (yi - alpha(w * xi))**2

def dw_L(w):
    return -(yi - alpha(w * xi)) * dz_alpha(w * xi) * xi

def update(w):
    return w - eta * dw_L(w)


xi = 1
yi = 1
wc = -3
eta = .1

ws = [ wc ]
Ls = [ L(wc) ]
for _ in range(400):
    wn = update(wc)
    Ln = L(wc)
    ws.append(wn)
    Ls.append(Ln)
    wc = wn

ws = np.array(ws)
Ls = np.array(Ls)

plt.subplot(1, 2, 1)

plt.plot(Ls)
plt.xlabel(r'iteration $i$')
plt.ylabel(r'$L\left(w^{(i)}\right)$')


def alpha(z):
    return np.tanh(z)

def dz_alpha(z):
    return 1 / np.square( np.cosh(z) )


xi = 1
yi = 1
wc = -3
eta = .1

ws = [ wc ]
Ls = [ L(wc) ]
for _ in range(400):
    wn = update(wc)
    Ln = L(wc)
    ws.append(wn)
    Ls.append(Ln)
    wc = wn

ws = np.array(ws)
Ls = np.array(Ls)

plt.subplot(1, 2, 2)
plt.plot(Ls)
plt.xlabel(r'iteration $i$')
plt.ylabel(r'$L\left(w^{(i)}\right)$')

plt.subplots_adjust(wspace=.5)

plt.show()
