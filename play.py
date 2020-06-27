#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 00:28:59 2020

@author: berk
"""


#------------------------------------------------------
#This section for reduce some errors related to GPU allocation on my system.
#it may not neccesary for yours. If it is, removing this part may increase the performance.
from tensorflow import Session,ConfigProto
from keras.backend.tensorflow_backend import set_session
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(Session(config=config))
#--------------------------------------------------------



from pynput.keyboard import Key, Controller
import os
from mss import mss
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array


keyboard = Controller()

def grabscreen():
    im = mss().grab({'top': 350, 'left': 625, 'width': 750, 'height': 130})
    im = Image.frombytes('RGB', im.size, im.rgb)
    im = img_to_array(im)
    return im

model = load_model('model.h5')   


while True:
    im = grabscreen()

    result = model.predict(im[None])
    if np.argmax(result)== 1:
        os.system('clear')
        print("up")
        keyboard.press(Key.space)
        keyboard.release(Key.space)            
    else:                       
        os.system('clear')
        print("straight")
        continue
    







