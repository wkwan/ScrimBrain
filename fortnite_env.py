from enum import Enum
import gymnasium as gym
import numpy as np
import dxcam
from PIL import Image
import easyocr
import math
from difflib import SequenceMatcher

import pyautogui
pyautogui.PAUSE = 0

import vgamepad as vg
import time

# To make the player face another direction, use a simulated right joystick instead of moving the mouse
# because controlling the mouse with pyautogui moves the cursor to a new position instantaneously,
# which causes the screen to jump in Fortnite.

# Though the simulated right joystick also moves it instantaneously, the movement in Fortnite is 
# still relatively smooth.
gamepad = vg.VX360Gamepad()

N_CHANNELS = 3
HEIGHT = 1080
WIDTH = 1920

RESIZE_FACTOR = 8

class DetectionState(Enum):
    DETECTED_TARGET = 1
    DETECTED_OTHER = 2
    DETECTED_NOTHING = 3

class FortniteEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(3)
        
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                            shape=(HEIGHT//RESIZE_FACTOR, WIDTH//RESIZE_FACTOR, N_CHANNELS), dtype=np.uint8)
        
        # for grabbing screencaptures
        self.cam = dxcam.create(output_idx=0)
        # for detecting the SCORE text to give a reward
        self.reader = easyocr.Reader(['en'])
        
        self.cur_step = 0
        self.step_since_last_reset = 0
        self.score = 0
        self.score_detected_cooldown_period = False

    # feed a downscaled screencapture to the neural net for efficiency
    def quarter_sized_screencap_np(self, screencap_img):
        return np.array(Image.fromarray(screencap_img).resize((WIDTH//RESIZE_FACTOR, HEIGHT//RESIZE_FACTOR), Image.Resampling.LANCZOS))
    
    # detect the SCORE text when the player reaches a trigger
    def score_detected(self, full_img):
        elim_ocr = self.reader.readtext(full_img[415:455, 975:1115], detail=0)
        if (len(elim_ocr) > 0):
            for i in range(len(elim_ocr)):
                elim_match_ratio = SequenceMatcher(None, elim_ocr[i], "SCORE").ratio()
                if (elim_match_ratio > 0.7):
                    return DetectionState.DETECTED_TARGET
            return DetectionState.DETECTED_OTHER
        return DetectionState.DETECTED_NOTHING

    def step(self, action):
        # if the server automatically restarted, attempt to queue back into the game
        if (self.cur_step % 10000 == 0):
            print("check if we need to reenter the game")
            try:
                back_button_location = pyautogui.locateOnScreen('media/level-up.png')
                pyautogui.press('esc')
                print(f"level up screen detected, pressed esc, now sleep for 15s")
                time.sleep(15)
                print("done sleeping after pressing esc")
            except Exception as e:
                print("level up screen not found")

            try:
                play_button_location = pyautogui.locateOnScreen('media/play-button.png')
                pyautogui.click(pyautogui.center(play_button_location))
                print(f"clicked play button at {play_button_location}, now sleep for 60s")
                time.sleep(60)
                print("done sleeping after clicking play button")
            except Exception as e:
                print("play button not found")

        reward = 0
        right_thumb_x = 0
        right_thumb_y = 0            

        # always move forward
        pyautogui.keyDown('w')            

        if action == 2: 
            right_thumb_x = -1 # look to the left
        elif action == 1:
            right_thumb_x = 0 # look straight
        elif action == 0:
            right_thumb_x = 1 # look to the right

        gamepad.right_joystick_float(x_value_float=right_thumb_x, y_value_float=right_thumb_y)
        gamepad.update()

        player_obs = None

        try:
            player_full_img = self.cam.grab()
            player_obs = self.quarter_sized_screencap_np(player_full_img)
        except Exception as e:
            print(f"step {self.cur_step} screencap fail {e}")  

        terminated = False
        if player_full_img is not None:
            try:
                score_detected = self.score_detected(player_full_img)
                if score_detected == DetectionState.DETECTED_TARGET:
                    if not self.score_detected_cooldown_period:
                        reward = 1
                        self.score_detected_cooldown_period = True
                        print(f"step {self.cur_step} score detected reward {reward}")

                 # Wait for the SCORE text to fully disappear before allowing another reward.
                 # Consecutive SCORE's without the text disappearing won't be registered.
                elif score_detected == DetectionState.DETECTED_NOTHING:
                    self.score_detected_cooldown_period = False  
            except Exception as e:
                print(f"step {self.cur_step} score detect detect failed {e}")

        truncated = False
        info = {}

        if player_obs is None:
            player_obs = np.zeros((HEIGHT//RESIZE_FACTOR, WIDTH//RESIZE_FACTOR, N_CHANNELS), dtype=np.uint8)

        if self.step_since_last_reset >= 5000: # fixed episode length to incentivize AI to reach targets as fast as possible
            terminated = True
            return player_obs, reward, True, False, {}

        self.cur_step += 1
        self.step_since_last_reset += 1
        return player_obs, reward, terminated, truncated, info

    def render(self):
        pass

    def reset(self, seed=None, options=None):
        # always move forward
        pyautogui.keyDown('w')            
        player_obs = None
        try:
            player_obs = self.quarter_sized_screencap_np(self.cam.grab())
        except:
            player_obs = np.zeros((HEIGHT//RESIZE_FACTOR, WIDTH//RESIZE_FACTOR, N_CHANNELS), dtype=np.uint8)
        gamepad.reset()
        gamepad.update()
        self.step_since_last_reset = 0
        return player_obs, {}

    def close(self):
        gamepad.reset()
        gamepad.update()
        pyautogui.keyUp('w')
