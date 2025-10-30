import numpy as np
from numpy import sin, cos, log, pi, random
import matplotlib.pyplot as plt

class Senal():
  def __init__(self):
    self._start = None
    self._stop = None
    self._step = None
    self._N = None
    self._dominio = None
    self.dominio_conv = None
    self._x = None

  def getDominio(self):
    return self._dominio

  def setDominio(self, s0, s1, st):
    if s0 == 0:
        s0 = st

    self._start = s0
    self._stop = s1
    self._step = st
    self._dominio = np.arange(s0, s1, st)
    self._N = len(self._dominio)

  def setX(self, A, a, B, b, C, c, noise=True):
    if noise:
        epsilon = random.normal(0, 0.2, self._N)
    else:
        epsilon = 0

    self._x = (A * sin(a * self._dominio)) + (B * cos(b * self._dominio)) + (C * log(c * self._dominio)) + epsilon

  def convolucion(self, h):
    #y = np.convolve(self._x, h, mode="same")
    y = np.convolve(self._x, h, mode="full")
    dom = np.linspace(self._start, self._stop, len(y), endpoint=False)
    self.dominio_conv = dom
    return dom, y

  def DFT(self):
    X = np.zeros(self._N, dtype=complex)
    for k in range(self._N):
      for n in range(self._N):
        X[k] += self._x[n] * np.exp(-2j * np.pi * k * n / self._N)
    return X

  def FFT(self):
    X = np.fft.rfft(self._x)
    return X

  def grafica(self, log_magnitud=False):
    """ Grafica la señal y su FFT 
    \nmagnitud: "log" o "normal" """

    X = np.abs(self.FFT())
    freq_norm = np.linspace(0, 0.5, len(X))
    label_mag = "Mag" if not log_magnitud else "log(Mag)"

    if log_magnitud:
      X = 1 * np.log10(X)

    plt.figure(figsize=(12,10))
    plt.subplot(2, 1, 1)
    plt.stem(self._dominio, self._x, label="Senal")
    plt.title("Grafico de la Señal")
    plt.ylabel("Amplitud")
    plt.xlabel("Dominio")
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.stem(freq_norm, X, linefmt='b-', markerfmt='bo', basefmt='k', label=label_mag)
    plt.stem(freq_norm, np.angle(X), linefmt='r-', markerfmt='ro', basefmt='k', label="Phase")

    plt.title("Grafico de FFT de la Señal")
    plt.ylabel("Amplitud")
    plt.xlabel("frec")
    plt.grid()
    plt.legend()
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])

    plt.tight_layout()
    plt.show()



def grafica_conv_tmp(y, x):
  Y = np.fft.rfft(y)
  freq_norm = np.linspace(0, 0.5, len(Y))

  plt.figure(figsize=(12,10))
  plt.subplot(2, 1, 1)
  plt.stem(x, y, label="Senal")
  plt.title("Grafico de la Señal")
  plt.ylabel("Amplitud")
  plt.xlabel("Dominio")
  plt.grid()
  plt.legend()

  plt.subplot(2, 1, 2)
  plt.stem(freq_norm, np.abs(Y), linefmt='b-', markerfmt='bo', basefmt='k', label="Mag")
  plt.stem(freq_norm, np.angle(Y), linefmt='r-', markerfmt='ro', basefmt='k', label="Phase")

  plt.title("Grafico de FFT de la Señal")
  plt.ylabel("Amplitud")
  plt.xlabel("frec")
  plt.grid()
  plt.legend()
  plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])

  plt.tight_layout()
  plt.show()

def grafica_conv_fft(y):
  freq_norm = np.linspace(0, 0.5, len(y))

  plt.figure(figsize=(12,5))
  plt.stem(freq_norm, np.abs(y), linefmt='b-', markerfmt='bo', basefmt='k', label="Mag")
  plt.stem(freq_norm, np.angle(y), linefmt='r-', markerfmt='ro', basefmt='k', label="Phase")

  plt.title("Grafico de FFT de la Señal")
  plt.ylabel("Amplitud")
  plt.xlabel("frec")
  plt.grid()
  plt.legend()
  plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])

  plt.tight_layout()
  plt.show()

def freq_conv(f_x, f_h):
    n_zeros = len(f_x) - len(f_h)
    zeros = np.zeros(abs(n_zeros))
    if n_zeros >= 0:
        f_h = np.concatenate((f_h, zeros))
    else:
        f_x = np.concatenate((f_x, zeros))
    
    return f_x * f_h

if __name__ == "__main__":
  """
      Creación de la señal de entrada:
          x(wt) = A*sin(awt) + B*cos(bwt) + C*log(cwt) + epsilon
  """

  fs = 100

  start, stop, step = 0, 6*pi, 2*pi/fs

  senal1 = Senal()

  senal1.setDominio(start, stop, step)

  senal1.setX(5, 1, 1, 30, 5, 10)

  dom = senal1.getDominio()

  senal1.grafica(log_magnitud=True)