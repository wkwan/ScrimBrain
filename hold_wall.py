import atexit
from time import sleep

import pyautogui
pyautogui.PAUSE = 0

def close():
    pyautogui.keyUp('o')

sleep(3) #get ready
pyautogui.keyDown('o')

while True:
    print("hodor")

atexit.register(close)