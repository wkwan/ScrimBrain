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
gamepad = vg.VX360Gamepad()

# from ultralytics import YOLO
# yolo_model = YOLO('yolov9e.pt')

N_CHANNELS = 3
HEIGHT = 1080
WIDTH = 1920
# MAX_Y_DIST_TO_CROSSHAIR = HEIGHT/2
MAX_DISTANCE_TO_CROSSHAIR = math.sqrt((HEIGHT/2) ** 2 + (WIDTH/2) **2)

class DetectionState(Enum):
    DETECTED_TARGET = 1
    DETECTED_OTHER = 2
    DETECTED_NOTHING = 3

holdable_vertical_move_keys = [
    'w', #forward
    's' #backward
]

holdable_horizontal_move_keys = [
    'a', #left
    'd' #right
]

# pressable_mode_keys = [
#     'e', #assault rifle"
#     # 'k', #wall
#     'f', #floor
#     'l', #stairs
#     '1', #roof
#     'g' #edit
# ]

holdable_keys = [
    'o', #fire, place building, select building edit
    'p', #target, reset building edit
]

pressable_keys = [
    'space', #jump
    'shiftleft', #sprint
    'ctrlleft', #crouch/slide
    # 'u', #reload
    # 'v', #repair/upgrade
    # 'r', #rotate
    # '9', #change building material
]

# has_at_least_one_nonzero_reward_during_learn_phase = False

class FortniteEnv(gym.Env):
    def __init__(self, use_yolo_reward=False):
        super().__init__()
        possible_actions = [3, 3] + [2]*len(holdable_keys) + [2]*len(pressable_keys) + [5, 5]
        # possible_actions = [3, 3] + [2]*len(holdable_keys) + [3, 3]
        # possible_actions = [2]*len(holdable_keys) + [3, 3]
        # possible_actions = [2]*len(holdable_keys) + [5, 5]
        self.action_space = gym.spaces.MultiDiscrete(possible_actions)
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                            shape=(HEIGHT//4, WIDTH//4, N_CHANNELS), dtype=np.uint8)
        self.cam = dxcam.create(output_idx=0)
        self.reader = easyocr.Reader(['en'])
        self.cur_step = 0
        self.player_killed_opponent_cooldown_period = False
        self.opponent_killed_player_cooldown_period = False
        self.use_yolo_reward = use_yolo_reward
        self.step_since_last_reset = 0

        self.prev_right_thumb_x = 0
        self.prev_right_thumb_y = 0

        self.score = 0
        self.last_step_killcount = 0
        self.score_detected_cooldown_period = False

    def quarter_sized_screencap_np(self, screencap_img):
        # Image.fromarray(screencap_img).resize((WIDTH//4, HEIGHT//4), Image.Resampling.LANCZOS).show()
        return np.array(Image.fromarray(screencap_img).resize((WIDTH//4, HEIGHT//4), Image.Resampling.LANCZOS))
    
    def elim_detected(self, full_img):
        elim_ocr = self.reader.readtext(full_img[655:685, 650:800], detail=0)
        # print("elim ocr: ", elim_ocr)
        if (len(elim_ocr) > 0):
            for i in range(len(elim_ocr)):
                elim_match_ratio = SequenceMatcher(None, elim_ocr[i], "ELIMINATED").ratio()
                if (elim_match_ratio > 0.7):
                    return DetectionState.DETECTED_TARGET
            return DetectionState.DETECTED_OTHER
        return DetectionState.DETECTED_NOTHING
    

    def score_detected(self, full_img):
        elim_ocr = self.reader.readtext(full_img[415:455, 975:1115], detail=0)
        # print("score detected ocr: ", elim_ocr)
        if (len(elim_ocr) > 0):
            for i in range(len(elim_ocr)):
                elim_match_ratio = SequenceMatcher(None, elim_ocr[i], "SCORE").ratio()
                if (elim_match_ratio > 0.7):
                    return DetectionState.DETECTED_TARGET
            return DetectionState.DETECTED_OTHER
        return DetectionState.DETECTED_NOTHING
    
    # hardcoded to the player name and the width of of the player name
    def got_killed_detected(self, full_img):
        elim_ocr = self.reader.readtext(full_img[610:640, 25:115], detail=0)
        # print("got killed detected ocr: ", elim_ocr)
        if (len(elim_ocr) > 0):
            for i in range(len(elim_ocr)):
                elim_match_ratio = SequenceMatcher(None, elim_ocr[i], "fefe3532").ratio()
                if (elim_match_ratio > 0.7):
                    return DetectionState.DETECTED_TARGET
            return DetectionState.DETECTED_OTHER
        return DetectionState.DETECTED_NOTHING
    
    def got_killed_by_guard_detected(self, full_img):
        elim_ocr = self.reader.readtext(full_img[615:645, 30:115], detail=0)
        # print("got killed detected ocr: ", elim_ocr)
        if (len(elim_ocr) > 0):
            for i in range(len(elim_ocr)):
                elim_match_ratio = SequenceMatcher(None, elim_ocr[i], "Guard").ratio()
                if (elim_match_ratio > 0.7):
                    return DetectionState.DETECTED_TARGET
            return DetectionState.DETECTED_OTHER
        return DetectionState.DETECTED_NOTHING
    
    def killcount(self, full_img):
        killcount_ocr = self.reader.readtext(full_img[80:115, 790:890], detail=0)
        # print("got killed detected ocr: ", elim_ocr)
        if (len(killcount_ocr) > 0):
            # print("killcount: ", elim_ocr)
            for i in range(len(killcount_ocr)):
                if (killcount_ocr[i].isdigit()):
                    return int(killcount_ocr[i])

    def step(self, action):

        if (self.cur_step % 10000 == 0):
            print("check if we need to reenter the game")
            try:
                back_button_location = pyautogui.locateOnScreen('media/level-up.png')
                pyautogui.press('esc')
                print(f"level up screen detected, pressed esc, now sleep for 15s")
                time.sleep(15)
                print("done sleeping after pressing esc")
            except Exception as e:
                print("level up not found")

            try:
                play_button_location = pyautogui.locateOnScreen('media/play-button.png')
                pyautogui.click(pyautogui.center(play_button_location))
                print(f"clicked play butto at {play_button_location}, now sleep for 60s")
                time.sleep(30)
                print("done sleeping after clicking play button")
            except Exception as e:
                print("play button not found")

                # print(action)
        reward = 0

        # if self.cur_step % 3 == 0:
        for i in range(len(holdable_vertical_move_keys)):
            if action[0] == i:
                pyautogui.keyDown(holdable_vertical_move_keys[i])
                # print("holdable_vertical_move_keys down: ", holdable_vertical_move_keys[i])
            else:
                pyautogui.keyUp(holdable_vertical_move_keys[i])

        for i in range(len(holdable_horizontal_move_keys)):
            if action[1] == i:
                pyautogui.keyDown(holdable_horizontal_move_keys[i])
                # print("holdable_horizontal_move_keys down: ", holdable_horizontal_move_keys[i])
            else:
                pyautogui.keyUp(holdable_horizontal_move_keys[i])

        # for i in range(len(pressable_mode_keys)):
        #     if action[2] == i:
        #         pyautogui.press(pressable_mode_keys[i])
        #         # print("pressable_mode_keys press: ", pressable_mode_keys[i])

        for i in range(len(holdable_keys)):
            if action[2+i] == 1:
                pyautogui.keyDown(holdable_keys[i])
                # print("holdable_keys down: ", holdable_keys[i])
            else:
                pyautogui.keyUp(holdable_keys[i])

        for i in range(len(pressable_keys)):
            if action[4+i] == 1:
                pyautogui.press(pressable_keys[i])
                # print("pressable_keys press: ", pressable_keys[i])

        # right_thumb_x = min(max(((action[-2]) - 1) * 0.25 + self.prev_right_thumb_x, -1), 1)
        # right_thumb_y = min(max(((action[-1]) - 1) * 0.25 + self.prev_right_thumb_y, -1), 1)
        # print(f"thumbs {right_thumb_x} {right_thumb_y}")
        right_thumb_x = (action[-2] - 2)/2
        right_thumb_y = (action[-1] - 2)/2
        self.prev_right_thumb_x = right_thumb_x
        self.prev_right_thumb_y = right_thumb_y

        # gamepad.right_joystick_float(x_value_float=right_thumb_x, y_value_float=right_thumb_y)
        gamepad.right_joystick_float(x_value_float=right_thumb_x, y_value_float=0)
        gamepad.update()

        
        player_obs = None

        try:
            player_full_img = self.cam.grab()
            player_obs = self.quarter_sized_screencap_np(player_full_img)
        except Exception as e:
            # print(f"step {self.cur_step} player health ocr failed {e}")  
            print(f"step {self.cur_step} screencap fail {e}")  

        # print("potential reward: ", -math.log10((self.step_since_last_reset+1)/1000))
        terminated = False
        if player_full_img is not None:
            # try:
            #     player_killed_opponent_detected = self.elim_detected(player_full_img)
            #     if player_killed_opponent_detected == DetectionState.DETECTED_TARGET:
            #         if not self.player_killed_opponent_cooldown_period:
            #             # reward += 2000
            #             reward = max(-math.log10((self.step_since_last_reset+1)/10000), .05)
            #             self.player_killed_opponent_cooldown_period = True
            #             print(f"step {self.cur_step} player killed opponent reward {reward}")
            #     elif player_killed_opponent_detected == DetectionState.DETECTED_NOTHING:
            #         if self.player_killed_opponent_cooldown_period:
            #             self.player_killed_opponent_cooldown_period = False
            #             print(f"step {self.cur_step} player killed opponent cooldown period terminated episode")
            #             return player_obs, 0, True, False, {}
    
            #         self.player_killed_opponent_cooldown_period = False  
            # except Exception as e:
            #     print(f"step {self.cur_step} player elim detect failed {e}")

            try:
                score_detected = self.score_detected(player_full_img)
                if score_detected == DetectionState.DETECTED_TARGET:
                    if not self.score_detected_cooldown_period:
                        reward += 5
                        self.score_detected_cooldown_period = True
                        print(f"step {self.cur_step} score detected add five {reward}")
                elif score_detected == DetectionState.DETECTED_NOTHING:
                    self.score_detected_cooldown_period = False  
            except Exception as e:
                print(f"step {self.cur_step} score detect detect failed {e}")

            # try:
            #     killcount_ocr = self.killcount(player_full_img)
            #     if (killcount_ocr is not None):
            #         # print('have killcount', killcount_ocr)
            #         if (killcount_ocr > self.score and killcount_ocr < self.score + 5):
            #             reward = killcount_ocr - self.score
            #             print(f"step {self.cur_step} killcount reward {reward}")
            #             self.score = killcount_ocr
            #         if (killcount_ocr == self.last_step_killcount):
            #             print("killcount ocr same as last step {killcount_ocr}")
            #             self.score = killcount_ocr
            #         self.last_step_killcount = killcount_ocr

            # except Exception as e:
            #     print(f"step {self.cur_step} killcount ocr failed {e}")

            try:
                # opponent_killed_player_detected = self.got_killed_detected(player_full_img)
                opponent_killed_player_detected = self.got_killed_by_guard_detected(player_full_img)

                if opponent_killed_player_detected == DetectionState.DETECTED_TARGET:
                    if not self.opponent_killed_player_cooldown_period:
                        # reward -= 2000
                        # reward = min(math.log10((self.step_since_last_reset+1)/10000), -.05)
                        reward -= 10
                        self.opponent_killed_player_cooldown_period = True
                        print(f"step {self.cur_step} opponent killed player punish {reward}")
                elif opponent_killed_player_detected == DetectionState.DETECTED_NOTHING:
                    if self.opponent_killed_player_cooldown_period:
                        self.opponent_killed_player_cooldown_period = False
                        print(f"step {self.cur_step} opponent killed player cooldown period terminated episode")
                        return player_obs, 0, True, False, {}
                    
                    self.opponent_killed_player_cooldown_period = False  
            except Exception as e:
                print(f"step {self.cur_step} opponent killed player detect failed {e}")

            try:
                feet_location = pyautogui.locateOnScreen('media/feet.png', confidence=0.6)
                if ((feet_location.top  + feet_location.height / 2) < HEIGHT / 2):
                    reward += 0.02
                    # print("punish for feet")
                    print("reward for feet")
            except Exception as e:
                pass
                # print(f"step {self.cur_step} feet detection failed {e}")

            # if self.use_yolo_reward:
            #     try:
            #         objects_detected = yolo_model(player_full_img, verbose=False) #channels inverted but might not matter
            #         crosshair_over_opponent = False
            #         for object in objects_detected:
            #             for i in range(len(object.boxes.cls)):
            #                 if object.boxes.cls[i] == 0:
            #                     # dist_to_crosshair = math.sqrt((object.boxes.xywh[i][0] - (WIDTH / 2))**2 + (object.boxes.xywh[i][1] - (HEIGHT / 2))**2)
            #                     # print(f"step {self.cur_step} normalized dist_to_crosshair: {dist_to_crosshair/MAX_DISTANCE_TO_CROSSHAIR}")
            #                     # reward += round((1 - dist_to_crosshair/MAX_DISTANCE_TO_CROSSHAIR) * 100)
            #                     if object.boxes.xywh[i][0] < (WIDTH / 2) and object.boxes.xywh[i][0] + object.boxes.xywh[i][2] > (WIDTH / 2) and object.boxes.xywh[i][1] < (HEIGHT / 2) and object.boxes.xywh[i][1] + object.boxes.xywh[i][3] > (HEIGHT / 2):
            #                         reward += 11
            #                         crosshair_over_opponent = True
            #                         break
            #             if crosshair_over_opponent:
            #                 break
            #         # if not found_center:
            #         #     reward -= 100
            #         # objects_detected = yolo_model(player_full_img, verbose=False) #channels inverted but might not matter
            #         # found_center = False
            #         # for object in objects_detected:
            #         #     for i in range(len(object.boxes.cls)):
            #         #         if object.boxes.cls[i] == 0:
            #         #             dist_to_crosshair = math.sqrt((object.boxes.xywh[i][0] - (WIDTH / 2))**2 + (object.boxes.xywh[i][1] - (HEIGHT / 2))**2)
            #         #             # print(f"step {self.cur_step} normalized dist_to_crosshair: {dist_to_crosshair/MAX_DISTANCE_TO_CROSSHAIR}")
            #         #             reward += round((1 - dist_to_crosshair/MAX_DISTANCE_TO_CROSSHAIR) * 100)
            #         #             found_center = True
            #         #             break
            #         #     if found_center:
            #         #         break
            #         # if not found_center:
            #         #     reward -= 100
                        
            #     except Exception as e:
            #         print(f"step {self.cur_step} yolo failed {e}")

        
        truncated = False
        info = {}

        if player_obs is None:
            player_obs = np.zeros((HEIGHT//4, WIDTH//4, N_CHANNELS), dtype=np.uint8)

        # global has_at_least_one_nonzero_reward_during_learn_phase
        # if reward != 0:
        #     has_at_least_one_nonzero_reward_during_learn_phase = True
            # print(f"{self.cur_step} reward: ", reward)

        self.cur_step += 1
        self.step_since_last_reset += 1
        return player_obs, reward, terminated, truncated, info

    def render(self):
        pass

    def reset(self, seed=None, options=None):
        print("reset")
        player_obs = None
        try:
            player_obs = self.quarter_sized_screencap_np(self.cam.grab())
        except:
            player_obs = np.zeros((HEIGHT//4, WIDTH//4, N_CHANNELS), dtype=np.uint8)
        self.prev_right_thumb_x = 0
        self.prev_right_thumb_y = 0
        gamepad.reset()
        gamepad.update()
        for key in holdable_vertical_move_keys:
            pyautogui.keyUp(key)
        for key in holdable_horizontal_move_keys:
            pyautogui.keyUp(key)
        for key in holdable_keys:
            pyautogui.keyUp(key)
        self.step_since_last_reset = 0
        # Image.fromarray(player_obs).show()
        return player_obs, {}

    def close(self):
        print("start cleanup")
        gamepad.reset()
        gamepad.update()
        for key in holdable_vertical_move_keys:
            pyautogui.keyUp(key)
        for key in holdable_horizontal_move_keys:
            pyautogui.keyUp(key)
        for key in holdable_keys:
            pyautogui.keyUp(key)
        print("done cleanup")