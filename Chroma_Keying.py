import cv2
import numpy as np

#####################################################################

# define video capture with access to camera 0

video_prompt = "C:/Users/chenx/Desktop/zhangyuanzhang-bg_with_torso-2024-6-14_300000_driven_zhangyuanzhang (1).mp4"
camera = cv2.VideoCapture(video_prompt)

# define display window

window_name = "HSV - colour selected image"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

#####################################################################

keep_processing = True

while (keep_processing):

    # read an image from the camera

    _, image = camera.read()

    # convert the RGB images to HSV

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # print the HSV values of the middle pixel

    height, width, _ = image.shape
    print('centre pixel HSV value: ', image_hsv[int(height/2)][int(width/2)])
    print()

    # define the range of hues to detect
    # - adjust these to detect different colours

    lower_green = np.array([55, 40, 40])
    upper_green = np.array([95, 255, 255])

    # create a mask that identifies the pixels in the range of hues

    mask = cv2.inRange(image_hsv, lower_green, upper_green)
    mask = cv2.dilate(mask, np.ones((7,7),np.uint8))
    mask_inverted = cv2.bitwise_not(mask)

    # create a grey image and black out the masked area

    # image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image_grey = cv2.bitwise_and(image_grey, image_grey, mask=mask_inverted)

    # black out unmasked area of original image

    image_masked = cv2.bitwise_and(image, image, mask=mask_inverted)

    # combine the two images for display

    # image_grey = cv2.cvtColor(image_grey, cv2.COLOR_GRAY2BGR)
    # image_combined = cv2.add(image_grey, image_masked)
    image_combined = image_masked

    # display image

    cv2.imshow(window_name, image_combined)
    cv2.imwrite("test.png", image_masked)

    # start the event loop - if user presses "x" then exit

    # wait 40ms or less for a key press from the user
    # (i.e. 1000ms / 25 fps = 40 ms)

    key = cv2.waitKey(40) & 0xFF

    if (key == ord('x')):
        keep_processing = False