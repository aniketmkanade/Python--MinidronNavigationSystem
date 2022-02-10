import drone as drone
import numpy as np
import math
from pyparrot.Minidrone import Mambo
import cv2
import time

mambo = Mambo(None, use_wifi=True) #address is None since it only works with WiFi anyway
print("trying to connect to mambo now")
success = mambo.connect(num_retries=3)
print("connected: %s" % success)

if (success):
    # get the state information
    print("sleeping")

#Task1: Take off till VERTICAL_DISTANCE
    mambo.safe_takeoff(5)
    print("took off")
    mambo.fly_direct(roll=0,pitch=25,yaw=0,vertical_movement=0,duration=5)
    print("Up by 40")
    
 #Task2: Delete previous pictures, Take a photo and display
    #delete previous pictures
    picture_names1 = mambo.groundcam.get_groundcam_pictures_names()
    for file in picture_names1:
            mambo.groundcam._delete_file(file)

    # take a new photo
    pic_success = mambo.take_picture()
    print("Took Picture")
    # need to wait a bit for the photo to show up
    mambo.smart_sleep(0.5)

    #get_groundcam_pictures_names -> Retruns a list with the names of the pictures stored on the Mambo.
    picture_names = mambo.groundcam.get_groundcam_pictures_names() #get list of availible files
    print(picture_names)

    #get_groundcam_picture -> Downloads the specified picture from the Mambo and stores it into a tempfile
    frame = mambo.groundcam.get_groundcam_picture(picture_names[0],True) #get frame which is the first in the array
    
    blurred_frame = cv2.GaussianBlur(frame, (5,5),0)

#Task 3: Target all the red objects using opencv
    lower1 = np.array([0,50,50])
    upper1 = np.array([20,255,255])

    lower2 = np.array([160,50,50])
    upper2 = np.array([180,255,255])

    image = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(image, lower1, upper1)
    mask2 = cv2.inRange(image, lower2, upper2)

    mask = cv2.bitwise_or(mask1, mask2)

#Task 4: Finding and drawing Contours
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    x=0
    y=0
    w=0
    h=0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= 1:
            cv2.drawContours(frame, contour, -1, (0,255,0), 3)

#Task 5: Get coordinates of the target (as per image system)
        x,y,w,h = cv2.boundingRect(contour)
        print("printing (x,y,w,h) coordinates wrt image : ")
        print(x)
        print(y)
        print(w)
        print(h)
        print("-----------")
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)
        break

#Task 6: Display Images
    cv2.imshow("xx", frame)
    #cv2.waitKey(-1)
    cv2.imwrite("originalImage.png", frame)
    cv2.imshow("mask", mask)
    cv2.imwrite("mask.png", mask)
    #cv2.waitKey(-1)

#Task 7: Get coordinates of the target as per  
    x1 = frame.shape[0]
    y1 = frame.shape[1]

    X_target = x - x1/2
    Y_target = y - y1/2
    print("Coordinates wrt drone : ")
    print(X_target)
    print(Y_target)

    X_drone = 0
    Y_drone = 0
    print("*************************")

#Task 8: Get angle of the drone with target
    
    side_length = y-y1/2
    hypotenius =  math.sqrt((X_target - X_drone)**2 + (Y_target - Y_drone)**2)
    print("Hypotenius : ")
    print(hypotenius)
    print("Angle : ")
    angle_wrt_yaxis = math.acos(side_length/hypotenius)
    print(math.degrees(angle_wrt_yaxis))
    angle_wrt_yaxis = math.degrees(angle_wrt_yaxis)
    print("*************************")

#Task 9: Turn the drone as per target  
      
    if(X_target >= 0 and Y_target >= 0):
       angle_wrt_yaxis = angle_wrt_yaxis     
    elif(X_target >= 0 and Y_target <= 0):
        angle_wrt_yaxis = 90 - angle_wrt_yaxis 
    elif(X_target <= 0 and Y_target >= 0):
        angle_wrt_yaxis =  angle_wrt_yaxis -180    
    else:
        angle_wrt_yaxis =  - angle_wrt_yaxis  
    
    print("Turn degrees")
    mambo.turn_degrees(int(angle_wrt_yaxis))
    #mambo.fly_direct(roll=0,pitch=0,yaw=angle_wrt_yaxis,vertical_movement=0,duration=2)

#Task 10: Navigate drone to target
    #Navigate the drone
    mambo.fly_direct(roll=0,pitch=hypotenius,yaw=0,vertical_movement=0,duration=2)

#Task 11: safeland and disconnect
    mambo.safe_land(5)
    mambo.disconnect()