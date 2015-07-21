"""This defines image utilities for dealing with the 
    images we get back from the environment
"""

import cv2

# TODO
# we need to actually make a function
# that takes in the dims and returns a scaler by itself
# but for now we can bake in the sizes

ale_scaler(imageBuffer):

    resized_width = 84
    resized_height = 84

    greyscaled = cv2.cvtColor(imageBuffer, cv2.COLOR_RGB2GRAY)

    return cv2.resize(greyscaled,
                              (resized_width, resized_height),
                              interpolation=cv2.INTER_LINEAR)
