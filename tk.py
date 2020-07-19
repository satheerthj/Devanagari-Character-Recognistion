# importing tkinter and tkinter.ttk 
# and all their functions and classes 
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter.ttk import *
import cv2
from PIL import Image
import pytesseract
import argparse
import os
# load and evaluate a saved model
from numpy import loadtxt
import numpy as np 
import pandas as pd 
import os
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
# importing askopenfile function 
# from class filedialog 
from tkinter.filedialog import askopenfile 
root = Tk()
root.geometry('1270x720')
root.title("Devanagari Script Recognistion")
# This function will be used to open 
# file in read mode and only Python files 
# will be opened 
print('loading dataset.....')
dataset = pd.read_csv('labels.csv')
print('load complete')
y=dataset['character'].to_numpy()
labelencoder = LabelEncoder()
YP = labelencoder.fit_transform(y)
model = keras.models.load_model('devnagari_model.h5')
print('load complete')
# summarize model.
model.summary()

def open_file(): 		
    	filename = filedialog.askopenfilename()
    	print(filename)
    	image = cv2.imread(filename)
    	print("converting to grayscale..")
    	gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    	x = gray.reshape(1,32, 32, 1)
    	ypred = model(x) 	
    	pred_y = labelencoder.inverse_transform(np.argmax(ypred, axis=-1))
    	print(pred_y)
    	label2 = tk.Label(text = "the recognised script")
    	label = tk.Label(text = pred_y)
    	label2.pack()
    	label.pack()
    	plt.axis('off')
    	plt.imshow(gray)
    	plt.show()


    	 
btn = Button(root, text ='Open', command = lambda:open_file()) 
btn.pack(side = TOP, pady = 10) 
pathlabel = Label(root)
pathlabel.pack()
mainloop() 
