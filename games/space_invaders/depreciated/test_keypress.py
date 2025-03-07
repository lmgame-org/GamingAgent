import pyautogui
import time

print("Manually click the game window and wait for 3s to start...")
time.sleep(3)

print("Test Left Movement...")
pyautogui.keyDown("left")
time.sleep(1)  
pyautogui.keyUp("left")
time.sleep(2)

print("Test Right Movement...")
pyautogui.keyDown("right")
time.sleep(1)
pyautogui.keyUp("right")
time.sleep(2)

print("Test shooting...")
pyautogui.press("space")
time.sleep(2)

print("Finished Testing")
