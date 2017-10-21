import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

img1 = cv2.imread('hero1.jpg')
img2 = cv2.imread('hero2.jpg')
color = ('b','g','r')

hist1_1 = cv2.calcHist([img1],[0],None,[256],[0,256])
hist2_1 = cv2.calcHist([img2],[0],None,[256],[0,256])
result = cv2.compareHist(hist1_1, hist2_1, method=cv2.HISTCMP_CORREL)
print("Single Channel Similarity: " + str(result))

hist1_2 = cv2.calcHist([img1],[1],None,[256],[0,256])
hist1_3 = cv2.calcHist([img1],[2],None,[256],[0,256])

hist2_2 = cv2.calcHist([img2],[1],None,[256],[0,256])
hist2_3 = cv2.calcHist([img2],[2],None,[256],[0,256])


hist1= np.column_stack([hist1_1, hist1_2, hist1_3]) 
hist2= np.column_stack([hist2_1, hist2_2, hist2_3]) 
result = cv2.compareHist(hist1, hist2, method=cv2.HISTCMP_CORREL)
print("Combined Channels Similarity: " + str(result))