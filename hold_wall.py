import atexit

import pyautogui
pyautogui.PAUSE = 0

pyautogui.keyDown['o']

def close(self):
    pyautogui.keyUp['o']

atexit.register(close)