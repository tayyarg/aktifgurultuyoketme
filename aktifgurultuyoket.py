import numpy as np
from scipy.signal import lfilter
from scipy.io import wavfile
import matplotlib.pyplot as plt

# iterasyon sayisi
N=10000

# Jet içi gürültü kaydını yükleyelim
fs, signal = wavfile.read('insidejet.wav')
y = np.copy(signal)

# giriş sinyal kaydı 8 bit olduğu için 256'ya normalize edelim
x = np.true_divide(y, 256)

# filtre katsayılarını oluşturalım
ind = np.arange(0,2.0,0.2)
p1 = np.array(np.zeros(50)).transpose()
p2 = np.array([np.exp(-(x**2)) for x in ind]).transpose()
p = np.append(p1,p2)
p_normalized = [x/np.sum(p) for x in p]
p_len = len(p_normalized)

# x giriş sinyali üzerinde FIR filtreleme yapalım
d = lfilter(p, [1.0], x)

# uyarlanır filtre katsayılarını ilklendirelim
w_len = p_len
w = np.zeros(w_len)

# sinyal gücü ve iterasyon sayısına bakarak adım aralığını bulalım (1/(N*E[x^2])'nin iki katı makul)
mu = 2/(N*np.var(x))

error_array = []
# uyarlanır filtre algoritmasını çalıştıralım
for i in range(w_len, N):
  x_ = x[i:i-w_len:-1]
  e = d[i] + np.array(w.T).dot(x_)
  w = w - mu * 2 * x_ * e
  error_array.append(e) 

f1 = plt.figure()
f2 = plt.figure()

ax1 = f1.add_subplot(111)
ax1.plot(p)
ax1.set_title('Birincil yol (primary path) filtre katsayıları')
ax2 = f2.add_subplot(111)
ax2.plot(error_array)
ax2.set_title('Jet içi gürültüsü - Uyarlanır Filtre hata eğrisi')
ax2.set(xlabel='iterasyon', ylabel='e')
plt.show()