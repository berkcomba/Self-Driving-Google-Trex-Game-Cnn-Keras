#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 18:12:26 2 020

@author: berk
"""



from pynput.keyboard import Key, Listener
import os
import time
from mss import mss
from PIL import Image

#key is a class(straight or up)
def grabscreen(key):
    time1=time.time()
    path="data/"+key+"/"
    name = len(os.listdir(os.path.join(os.getcwd(),path)))  #filename is current lenght of class's directory.
    im = mss().grab({'top': 350, 'left': 625, 'width': 750, 'height': 130}) #it is for 1920x1080 screens. It definitly different for smaller or bigger screens..
    im = Image.frombytes('RGB', im.size, im.rgb)
    im.save("data/"+key+"/"+str(name)+'.png')
    time2=time.time()
    print("it took "+str(1000*(time2-time1))+" ms")      #how long did screenshot takes.        
    name+=1
    print('{0} pressed ScreenShot taken'.format(key))


def on_press(key): 
    if key == Key.space: 
        grabscreen("up")      
    else: 
        grabscreen("straight")
        print(key)

def on_release(key):
    print('{0} release'.format(key))
    if key == Key.esc:
        # Stop listener
        return False



# Collect events until released
with Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()

