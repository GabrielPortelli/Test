# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 18:49:46 2020

@author: gabri
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as  plt# utile pour les graphiques

image = cv.imread("lena.png")
b,v,r = cv.split(image)         # récupère 3 matrices d'octets séparées pour R et V et B
y = 0.299*r + 0.587*v + 0.114*b # opération matricielle
y = y.astype(np.uint8)          # convertit les réels en octets

cv.imshow("Luminance Y", y)
#cv.waitKey(0)
#cv.destroyAllWindows()


hist = np.zeros(256, int)       # prépare un vecteur de 256 zéros (pour chaque gris)
for i in range(0, len(y)):
    for j in range(0,y.shape[1]):
        hist[y[i,j]] += 1

#print(hist)
plt.plot(hist)
plt.show()

# Calcule l'histogramme cumulé hc
hc = np.zeros(256, int)         # prépare un vecteur de 256 zéros
hc[0] = hist[0]
for i in range(1,256):
    hc[i] = hist[i] + hc[i-1]


# Normalise l'histogramme cumulé
nbpixels = y.size
hc = hc / nbpixels * 255
print(hc)
plt.plot(hc)
plt.show()

# Utilise hc comme table de conversion des niveaux de gris
for i in range(0,y.shape[0]):       # énumère les lignes
    for j in range(0,y.shape[1]):   # énumère les colonnes
        y[i,j] = hc[y[i,j]]
cv.imshow("Luminance Y après égalisation", y)
cv.waitKey(0)
cv.destroyAllWindows()