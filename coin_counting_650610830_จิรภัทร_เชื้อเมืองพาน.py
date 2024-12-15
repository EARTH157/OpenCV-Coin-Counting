#Download images from https://drive.google.com/file/d/1KqllafwQiJR-Ronos3N-AHNfnoBb8I7H/view?usp=sharing

import cv2
import numpy as np

def coinCounting(filename):
    im = cv2.imread("C:\\Users\\earth\\Documents\\computer_vision-master\\CoinCounting\\coin"+str(filename)+".jpg")
    target_size = (int(im.shape[1]/1),int(im.shape[0]/1))
    im = cv2.resize(im,target_size)
    im = cv2.resize(im, (360,490))
    kernel1 = np.ones((7 ,7),np.uint8)
    kernel_for_blue = np.ones((2,2),np.uint8)
    kernel2 = np.ones((10 ,5),np.uint8)

    #mask for yellow color
    mask_yellow = cv2.inRange(im, (20, 140, 0), (120, 255, 255))
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel1)
    mask_yellow = cv2.erode(mask_yellow,kernel1,iterations = 1)
    mask_yellow = cv2.medianBlur(mask_yellow, 5)
    mask_yellow = cv2.dilate(mask_yellow,kernel1,iterations = 1)

    #mask for blue color
    mask_blue_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    mask_blue_background = cv2.GaussianBlur(mask_blue_gray, (61, 61), 0)
    mask_blue_background = np.clip(mask_blue_background, 1, 200)
    mask_blue_background_color = cv2.merge([mask_blue_background] * 3)
    mask_blue_norm_im = (im / mask_blue_background_color) * 190
    mask_blue_norm_im = np.clip(mask_blue_norm_im, 0, 255).astype(np.uint8)
    mask_blue_erode_im = cv2.erode(mask_blue_norm_im, np.ones((19,18), np.uint8))
    mask_blue_dilated = cv2.dilate(mask_blue_erode_im, np.ones((6,4), np.uint8))
    mask_blue_range = cv2.inRange(mask_blue_dilated,(180,145,0),(255,255,130))
    mask_blue_erode = cv2.erode(mask_blue_range,kernel2,iterations = 1)
    mask_blue = cv2.dilate(mask_blue_erode,kernel_for_blue,iterations = 1)

    contours_yellow, hierarchy_yellow = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_blue, hierarchy_blue = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    yellow = len(contours_yellow)
    blue = len(contours_blue)

    print('Yellow = ',yellow)
    print('Blue = ', blue)

    cv2.putText(im,"Blue : "+str(blue),(50,100),
                    fontFace = cv2.FONT_HERSHEY_PLAIN,
                    fontScale = 2,
                    thickness = 2,
                    color = (255,0,0)) 
    
    cv2.putText(im,"Yellow : "+str(yellow),(100,50),
                    fontFace = cv2.FONT_HERSHEY_PLAIN,
                    fontScale = 2,
                    thickness = 2,
                    color = (0,255,255))

    cv2.imshow('Original Image',im)
    #cv2.imshow('Yellow Coin', mask_yellow)
    #cv2.imshow('Blue Coin', mask_blue_erode)
    
    cv2.waitKey() & 0xFF == ord('q')

    return [yellow,blue]


for i in range(1,11):
    print(i,":",coinCounting(i))
