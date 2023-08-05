import time
import cv2
from se_openhw.kit.nano import PTCamera
import se_openhw.platform as platform

ptc=PTCamera()
frame=ptc.get_frame()
print("cap successfully")

ptc.set_pitch(10)

cv2.imshow("frame",frame)

cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image