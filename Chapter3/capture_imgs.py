import pyrealsense2 as rs
import numpy as np
import cv2
import os

pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
profile=pipe.start(config)

def get_frame():
    frames = pipe.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    return color_image

save_path = '/home/user/saving_dir....'
color_path = os.path.join(save_path, 'color')

os.makedirs(color_path, exist_ok=True)

i = 0
while True:
    color = get_frame()

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite(os.path.join(color_path, str(i)+'.png'), color)
        print('save:' + str(i))
        i += 1

    cv2.imshow('color', color)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
