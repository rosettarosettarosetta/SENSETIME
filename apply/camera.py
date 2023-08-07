from se_openhw.kit.nano import PTCamera
import se_openhw.platform as platform
import threading
import time
#166

ptc =PTCamera()

def rotate_clockwise():
    global ptc
    for i in range(-45, 45):
        ptc.set_pitch(20)
        ptc.set_yaw(i)
        time.sleep(0.5)

# 逆时针转动180度
def rotate_counterclockwise():
    global ptc
    for i in range(-45, 45) :
        ptc.set_pitch(20)
        ptc.set_yaw(-i)
        time.sleep(0.5)


while True:
    rotate_clockwise()#-90 到90
    rotate_counterclockwise() #90 到 -90
