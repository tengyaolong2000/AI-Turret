# IMPORTING RELEVANT LIBRARIES
import cv2
import time
import numpy as np 
import jetson.inference
import jetson.utils
import board
import digitalio
from adafruit_servokit import ServoKit
import math
import random as random

# SERVOS AND OSCILLATIONS RELATED
# Creating ServoKit object
myKit = ServoKit(channels=16) 
# Constants
MAXANGLE = 175      # most left/top (0-180 from RIGHT TO LEFT, BOTTOM TO TOP)
MINANGLE = 5        # most right/bottom
INCREMENTANGLE = 5
OSCILLATIONANGLE = 5
BOUNDARYSHIFT = 20
RESETCAMLIMIT = 60 
# Variables
panAngle = 90       # tracks rotational angle of pan servo
tiltAngle = 90      # tracks rotational angle of tilt servo
resetCamCounter = 0 # counts the number of frames that has elapsed since the last object that was detected
# Flags                                                                                         
oscillateFlag = False
outOfBounds = False
topFlag=False
bottomFlag=False
leftFlag=False
rightFlag=False
# Dictionaries
imaginaryBox = {    # holds the coordinates of the bounding box that enclose all detected birds
    'top':0,
    'bottom': 0,
    'left':0,
    'right':0
}
theBox = {          # secondary bounding box that is used when servos hit their rotational limit
    'top':0,        
    'bottom': 0,
    'left':0,
    'right':0
}


# CAMERA AND DISPLAY RELATED
# Constants
DISPLAYWIDTH=1280                                                           
DISPLAYHEIGHT=720
# Objects
cam=jetson.utils.gstCamera(DISPLAYWIDTH,DISPLAYHEIGHT,'/dev/video0')      # display object
display=jetson.utils.glDisplay()                                          # camera object
timeStamp=time.time()                                                     # time object


# CROSSHAIR COORDINATES
SQUARESIZE = 15
MIDLEFT = 640 - SQUARESIZE
MIDBOT = 360 + SQUARESIZE
MIDRIGHT = 640 + SQUARESIZE
MIDTOP = 360 - SQUARESIZE                                                                                          


# TARGET RELATED
target = 'person'
antiTarget = 'bird'
birdList = []
detected = False


# LASER RELATED
laser_pin = digitalio.DigitalInOut(board.D5)
laser_pin.direction = digitalio.Direction.OUTPUT


# OBJECT DETECTION ALGORITHM RELATED
net=jetson.inference.detectNet('ssd-mobilenet-v2',threshold=.5)     # detectNet object 

#FUNCTIONS
def max(x,y):
    if x>y:
        return x
    elif x<y:
        return y
    else:
        return x

def min(x,y):
    if x<y:
        return x
    elif x>y:
        return y
    else:
        return x


def resetCam():
    '''
    Resets servos to default position when:
    -   Either servos reach their rotational limit
    -   No objects detected after a certain period of time (60 frames)
    '''
    #Flags needed
    global panAngle
    global tiltAngle
    # Reseting panAngle and tiltAngle
    panAngle=90
    tiltAngle=120   
    # Reseting servos
    myKit.servo[0].angle = panAngle
    myKit.servo[1].angle = tiltAngle
    return


def centerCam():
    '''
    Function Description:
    -   Calculates midpoint of the enclosing bounding box (imaginaryBox) and rotate servos to align crosshair with the midpoint
    -   If the center of imaginaryBox is unreachable in the case where either servos have reached their rotational limit,
        a secondary bounding box (theBox) will be used. This is controlled with outOfBounds flag
    -   Whenever the center of the primary bounding box becomes reachable again, 
        centerCam() will revert back to using the primary bounding (imaginaryBox)
    -   returns True if crosshair is centered and False if otherwise
    '''
    
    #Flags needed
    global panAngle, tiltAngle, INCREMENTANGLE
    global imaginaryBox, theBox
    global outOfBounds, topFlag, bottomFlag, leftFlag, rightFlag

    # USING PRIMARY BOUNDING BOX (imaginaryBox)
    if not outOfBounds:
        x = (imaginaryBox['right'] + imaginaryBox['left'])/2    #mid x-xcoordinate of imaginary box
        y = (imaginaryBox['top'] + imaginaryBox['bottom'])/2    #mid y-xcoordinate of imaginary box

        #LEFT RIGHT
        # Crosshair is centered on the x-axis
        if MIDLEFT<=x<=MIDRIGHT:
            print("Centered HORIZONTALLY on PRIMARY bounding box")
            pass
        # Crosshair is not centered on the x-axis
        else:
            # Updating panAngle accordingly                
            panAngle = (panAngle+INCREMENTANGLE) if (x<MIDLEFT) else (panAngle-INCREMENTANGLE)

            # If panAngle is too small (pan servo too far RIGHT), center of imaginary box to the RIGHT of crosshair and crosshair in imaginary box, undo changes to panAngle and set Flags
            if (panAngle < MINANGLE and imaginaryBox['left']<=MIDLEFT):
                theBox['right'] = MIDRIGHT
                outOfBounds=True
                rightFlag=True
                panAngle+=INCREMENTANGLE
                print("Switching to SECONDARY bounding box")
         
            # If panAngle is too small (too far RIGHT), center of imaginary box to the RIGHT of crosshair but crosshair NOT in imaginary box, undo changes to panAngle and resetCam
            elif (panAngle < MINANGLE and imaginaryBox['left']>MIDLEFT):
                print("Bird(s) too far to the right, resetting device...")
                resetCam()
                return False

            # If panAngle is too great (too far LEFT), center of imaginary box to the LEFT of crosshair and crosshair in imaginary box, undo changes to panAngle and set Flags
            elif (panAngle > MAXANGLE and imaginaryBox['right']>=MIDRIGHT):
                theBox['left'] = MIDLEFT              
                outOfBounds=True
                leftFlag=True
                panAngle-=INCREMENTANGLE
                print("Switching to SECONDARY bounding box")
    
            # If panAngle is too great (too far LEFT), center of imaginary box to the LEFT of crosshair but crosshiar NOT in imnaginary box, undo changes to panAngle and resetCam
            elif (panAngle > MAXANGLE and imaginaryBox['right']<MIDRIGHT):
                print("Bird(s) too far to the right, resetting device...")
                resetCam()
                return False

            # Rotate pan servo by new angle
            print("\tin progress for horizontal...")
            myKit.servo[0].angle = panAngle
            
        # UP DOWN
        # Crosshair is centered on both x-axis and y-axis
        if MIDLEFT<=x<=MIDRIGHT and MIDTOP<=y<=MIDBOT:
            print("Centered VERTICALLY on PRIMARY bounding box\n\tCROSSHAIR IS CENTERED")
            return True

        # Crosshair is not centered on y-axis
        else:                  
            # Updating tiltAngle accordingly
            tiltAngle = (tiltAngle+INCREMENTANGLE) if (y<MIDTOP) else (tiltAngle-INCREMENTANGLE)

            # If tiltAngle is too great (too far UP), and crosshair in imaginary box, undo changes to tiltAngle and set Flags
            if (tiltAngle > MAXANGLE and imaginaryBox['bottom']>=MIDBOT):
                theBox['top'] = MIDTOP
                outOfBounds=True
                topFlag=True
                tiltAngle-=INCREMENTANGLE
                if not (leftFlag or rightFlag):
                    print("Switching to SECONDARY bounding box")

            # If tiltAngle is too great (too far UP), and crosshair NOT in imaginary box, undo changes to tiltAngle and resetCam
            elif (tiltAngle > MAXANGLE and imaginaryBox['bottom']<MIDBOT):
                print("Bird(s) too high up, resetting device...")
                resetCam()
                return False

            # If tiltAngle is too small (too far DOWN), and crosshair in imaginary box, undo changes to tiltAngle and create new center
            elif (tiltAngle < MINANGLE and imaginaryBox['top']<=MIDTOP):
                theBox['bottom'] = MIDBOT
                outOfBounds=True
                bottomFlag=True
                tiltAngle+=INCREMENTANGLE
                if not (leftFlag or rightFlag):
                    print("Switching to SECONDARY bounding box")
            
            # If tiltAngle is too small (too far DOWN), and crosshiar NOT in imnaginary box, undo changes to tiltAngle and resetCam
            elif (tiltAngle < MINANGLE and imaginaryBox['top']>MIDTOP):
                print("Bird(s) too close to device, resetting device...")
                resetCam()
                return False

            # Rotate tilt servo by new angle
            print("\tin progress for vertical...")
            myKit.servo[1].angle = tiltAngle

            # Return False since crosshair is not yet centered
            return False


    # USING SECONDARY BOUNDING BOX (theBox)
    else:
        x = (imaginaryBox['right'] + imaginaryBox['left'])/2    #midpoint x-coordinate of PRIMARY bounding box (imaginaryBox)
        y = (imaginaryBox['top'] + imaginaryBox['bottom'])/2    #midpoint y-coordinate of PRIMARY bounding box (imaginaryBox)
        x2 = (theBox['right'] + theBox['left'])/2               #midpoint x-coordinate of SECONDARY bounding box (theBox)
        y2 = (theBox['top'] + theBox['bottom'])/2               #midpoint y-coordinate of SECONDARY bounding box (theBox)

        # Check if the center of primary bounding box is reachable by the crosshair
        if (MIDLEFT<=x<=MIDRIGHT) and (MIDTOP<=y<=MIDBOT):
            outOfBounds = False
            topFlag = bottomFlag = leftFlag = rightFlag = False
            print('Reverting to PRIMARY bounding box')
            return False

        # LEFT RIGHT
        # Crosshair is centered on x-axis
        if MIDLEFT<=x2<=MIDRIGHT: 
            print("Centered HORIZONTALLY on SECONDARY bounding box")
            pass
        else:
            # Initial Case where pan servo rotated too far to the RIGHT and center of SECONDARY bounding box is initially to the LEFT of crosshair (reachable)
            if x2<MIDLEFT: # Means center of SECONDARY bounding box is LEFT of crosshair
                panAngle += INCREMENTANGLE

                # Attempting to keep the edge of the SECONDARY bounding box over the same position of the subject as the pan servo move
                if rightFlag:
                    theBox['right'] += BOUNDARYSHIFT # adjusting SECONDARY bounding box size to allow wider oscillations while ensuring crosshair stays within boundaries
                elif leftFlag:
                    theBox['left'] += BOUNDARYSHIFT

                # When width of SECONDARY bounding box becomes less than or equal to the width of the crosshair, there is no point oscillating so reset device
                if (theBox['right'] - theBox['left']) <= 2*SQUARESIZE:
                    print("Width too small, resetting device...")
                    outOfBounds = False
                    topFlag = bottomFlag = leftFlag = rightFlag = False
                    resetCam()
                    return False

                # Failsafe for case when birds move further to the RIGHT causing center of SECONDARY BOX to become unreachable (pan servo will reach its rotational limit again) 
                # Needed since BOUNDARYSHIFT may not ensure that the edge of theBox that we manipulate is always over the same position of the subject
                if panAngle>=MAXANGLE:
                    print('Pan servo exceeds rotational limit, resetting device...')
                    outOfBounds = False
                    topFlag = bottomFlag = leftFlag = rightFlag = False                   
                    resetCam()
                    return False


            # Initial Case where pan servo rotated too far to the LEFT and center of SECONDARY bounding box is initially to the RIGHT of crosshair (rechable)
            elif x2>MIDRIGHT:# Means center of SECONDARY bounding box is RIGHT of crosshair
                panAngle -= INCREMENTANGLE 
                
                # Attempting to keep the edge of the SECONDARY bounding box over the same part of the subject as the pan servo move 
                if leftFlag:
                    theBox['left'] -= BOUNDARYSHIFT # adjusting SECONDARY bounding box size to allow wider oscillations while ensuring crosshair stays within boundaries
                elif rightFlag:
                    theBox['right'] -= BOUNDARYSHIFT

                # When width of SECONDARY bounding box becomes less than or equal to the width of the crosshair, there is no point oscillating so reset device
                if (theBox['right'] - theBox['left']) <= 2*SQUARESIZE:
                    print("Width too small, resetting device...")
                    outOfBounds = False
                    topFlag = bottomFlag = leftFlag = rightFlag = False
                    resetCam()
                    return False

                # Failsafe for case when birds move further to the LEFT causing center of SECONDARY BOX to become unreachable (pan servo will reach its rotational limit again)
                # Needed since BOUNDARYSHIFT may not ensure that the edge of theBox that we manipulate is always over the same position of the subject
                if panAngle<=MINANGLE:
                    print('Pan servo exceeds rotational limit, resetting device...')
                    outOfBounds = False
                    topFlag = bottomFlag = leftFlag = rightFlag = False
                    resetCam()
                    return False

            # Rotate pan servo by new angle
            print("\tin progress for horizontal...")
            myKit.servo[0].angle = panAngle        
            

        # UP DOWN
        # Crosshair is centered on both x-axis and y-axis
        if MIDLEFT<=x2<=MIDRIGHT and MIDTOP<=y2<=MIDBOT:
            print("Centered VERTICALLY on SECONDARY bounding box\n\tCROSSHAIR IS CENTERED")
            return True
        else:
            # Initial Case where tilt servo rotated too far DOWN and center of SECONDARY bounding box is initially at the TOP of crosshair (reachable)               
            if y<MIDTOP: # Means center of SECONDARY bounding box is ABOVE crosshair
                tiltAngle += INCREMENTANGLE

                # Attempting to keep the edge of the SECONDARY bounding box over the same part of the subject as the tilt servo move 
                if bottomFlag:
                    theBox['bottom'] += BOUNDARYSHIFT #adjusting theBox size for wider oscillations while ensuring crosshair stays within boundaries
                elif topFlag:
                    theBox['top'] += BOUNDARYSHIFT

                # When height of SECONDARY bounding box becomes less than or equal to the height of the crosshair, there is no point oscillating so reset device
                if (theBox['bottom'] - theBox['top']) <= 2*SQUARESIZE:
                    print("Height too small, resetting device...")
                    outOfBounds = False
                    topFlag = bottomFlag = leftFlag = rightFlag = False
                    resetCam()
                    return False

                # Failsafe for case when birds move further UP causing center of SECONDARY BOX to become unreachable (tilt servo will reach its rotational limit again)
                # Needed since BOUNDARYSHIFT may not ensure that the edge of theBox that we manipulate is always over the same position of the subject 
                if tiltAngle>=MAXANGLE:
                    print('Tilt servo exceeds rotational limit, resetting device...')
                    outOfBounds = False
                    topFlag = bottomFlag = leftFlag = rightFlag = False
                    resetCam()
                    return False

            # Initial Case where tilt servo rotated too far UP and center of SECONDARY bounding box is initially at the BOTTOM of crosshair (reachable)
            elif y>MIDBOT: # Means center of SECONDARY bounding box is BELOW crosshair
                tiltAngle -= INCREMENTANGLE 

                # Attempting to keep the edge of the SECONDARY bounding box over the same part of the subject as the tilt servo move 
                if topFlag:
                    theBox['top'] -= BOUNDARYSHIFT #adjusting theBox size for wider oscillations while ensuring crosshair stays within boundaries
                elif bottomFlag:
                    theBox['bottom'] -= BOUNDARYSHIFT

                # When height of SECONDARY bounding box becomes less than or equal to the height of the crosshair, there is no point oscillating so reset device
                if (theBox['bottom'] - theBox['top']) <= 2*SQUARESIZE:
                    print("Height too small, resetting device...")
                    outOfBounds = False
                    topFlag = bottomFlag = leftFlag = rightFlag = False
                    resetCam()
                    return False

                # Failsafe for case when birds move further UP causing center of SECONDARY BOX to become unreachable (tilt servo will reach its rotational limit again)
                # Needed since BOUNDARYSHIFT may not ensure that the edge of theBox that we manipulate is always over the same position of the subject    
                if tiltAngle<=MINANGLE:
                    print('Tilt servo exceeds rotational limit, resetting device...')
                    outOfBounds = False
                    topFlag = bottomFlag = leftFlag = rightFlag = False
                    resetCam()
                    return False

            # Rotate tilt servo by new angle
            print("\tin progress for vertical...")
            myKit.servo[1].angle = tiltAngle
            return False


def imaginaryBox1(birdList):
    '''
        Function Description:
        -   Update the coordinates of both primary and secondary bounding boxes when given a list of bird coordinates
        -   If the secondary bounding box is used by the centerCam() function, the coordinates that we aren't manipulating will be correctly updated by this function
    '''

    #Flags needed
    global imaginaryBox
    global theBox
    global topFlag, bottomFlag, leftFlag, rightFlag
    
    leftbox = 1280          # to find the smallest value
    rightbox = 0            # to find the largest value
    topbox = 720            # to find the smallest value
    bottombox = 0           # to find the largest value

    # Finding the coordinates of the primary bounding box (imaginaryBox)
    for bird in birdList:
        leftbox = min(bird.Left, leftbox)
        rightbox = max(bird.Right, rightbox)
        topbox = min(bird.Top, topbox)
        bottombox = max(bird.Bottom, bottombox)
                                                                     
    # Updating coordinates of the primary bounding box (imaginaryBox)
    imaginaryBox['top'] = topbox
    imaginaryBox['bottom'] = bottombox
    imaginaryBox['left'] = leftbox
    imaginaryBox['right'] = rightbox
    
    # Using PRIMARY bounding box IN centerCam() when neither servos have reached their rotational limit                                                                  
    if not outOfBounds:
        theBox = imaginaryBox.copy() # Let SECONDARY bounding box have the same coordinates as the PRIMARY bounding box

    # Using SECONDARY bounding box IN centerCam()                                                             
    else:
        # Selective updating of SECONDARY bounding box coordinates
        if topFlag: # manipulate TOP edge in centerCam()
            theBox['bottom'] = bottombox
            theBox['left'] = leftbox
            theBox['right'] = rightbox
            
        if bottomFlag: # manipulate BOTTOM edge in centerCam()
            theBox['top'] = topbox
            theBox['left'] = leftbox
            theBox['right'] = rightbox
            
        if leftFlag: #fixed LEFT edge in centerCam() 
            theBox['top'] = topbox
            theBox['bottom'] = bottombox
            theBox['right'] = rightbox
            
        if rightFlag: #fixed RIGHT edge in centerCam()
            theBox['top'] = topbox
            theBox['bottom'] = bottombox
            theBox['left'] = leftbox

    return


def oscillate():
    ''' 
    Function Description:
    -   For each function call, rotate both servos in a random direction by a fixed angle
    -   Returns True if oscillation is successful and False if the crosshair is already outside the bounding box or when either servos have reached their rotational limit
    '''

    # Flags needed
    global outOfBounds, topFlag, bottomFlag, leftFlag, rightFlag
    global panAngle, tiltAngle

    # Variables to check if panAngle or tiltAngle is greater than the max/min servo rotational angle
    toPan = panAngle
    toTilt = tiltAngle

    # Using Primary bounding box (imaginaryBox)
    if not outOfBounds:
        # When crosshair is within boundaries of primary bounding box
        if MIDLEFT>=imaginaryBox['left'] and MIDRIGHT<= imaginaryBox['right'] and MIDTOP>=imaginaryBox['top'] and MIDBOT<=imaginaryBox['bottom']:
            choice = random.choice(['pan', 'tilt', 'both'])
            rd = random.choice([OSCILLATIONANGLE,-OSCILLATIONANGLE])

            if choice == 'pan':
                toPan+=rd

            if choice == 'tilt':
                toTilt+=rd

            if choice == 'both':
                toPan+=rd
                toTilt+=rd

            #returns false when rotation exceeds min/max servo rotational angle
            if ((toPan <= MINANGLE) or (toPan >= MAXANGLE)) or ((toTilt <= MINANGLE) or (toTilt >= MAXANGLE)):
                print("Servos hit rotational limit, stopping oscilaltions...")
                return False

            # Update panAngle, tiltAngle and servos accordingly if all's good
            panAngle = toPan
            tiltAngle = toTilt
            myKit.servo[0].angle = panAngle
            myKit.servo[1].angle = tiltAngle
            print("\tin progress...")
            return True 

        #returns false if crosshair already escaped the boundary of imaginary box
        else:
            print("Crosshair no longer in bounding box, stopping oscillations...")
            return False

    # Using Secondary bounding box (theBox)
    else:
        # If secondary bounding box width/height gets too small, there is no point oscillating. Thus re-center based on primary bounding box
        if ((theBox['right'] - theBox['left']) <= 2*SQUARESIZE) or ((theBox['bottom'] - theBox['top']) <= 2*SQUARESIZE):
            print("Width or Height of secondary bounding box too small, stopping oscillations to recenter device...")
            outOfBounds = False
            topFlag = bottomFlag = leftFlag = rightFlag = False
            return False

        # When crosshair is within secondary bounding box
        if MIDLEFT>=theBox['left'] and MIDRIGHT<= theBox['right'] and MIDTOP>=theBox['top'] and MIDBOT<=theBox['bottom']:
            choice = random.choice(['pan', 'tilt', 'both'])
            rd = random.choice([OSCILLATIONANGLE,-OSCILLATIONANGLE])

            if choice == 'pan':
                toPan+=rd

            if choice == 'tilt':
                toTilt+=rd

            if choice == 'both':
                toPan+=rd
                toTilt+=rd

            #returns false when rotation exceeds min/max
            if ((toPan <= MINANGLE) or (toPan >= MAXANGLE)) or ((toTilt <= MINANGLE) or (toTilt >= MAXANGLE)):
                print("Servos hit rotational limit, stopping oscilaltions...")
                return False

            # Ensuring that manipulated edges of secondary bounding box stays over roughly the same position as device oscillates
            if topFlag:
                if toTilt>0:
                    theBox['top'] += BOUNDARYSHIFT
                else:
                    theBox['top'] -= BOUNDARYSHIFT

            if bottomFlag:
                if toTilt>0:
                    theBox['bottom'] += BOUNDARYSHIFT
                else:
                    theBox['bottom'] -= BOUNDARYSHIFT

            if leftFlag:
                if toPan>0:
                    theBox['left'] += BOUNDARYSHIFT
                else:
                    theBox['left'] -= BOUNDARYSHIFT

            if rightFlag:
                if toPan>0:
                    theBox['right'] += BOUNDARYSHIFT
                else:
                    theBox['right'] -= BOUNDARYSHIFT

            # Update panAngle, tiltAngle and servos accordingly if all's good
            panAngle = toPan
            tiltAngle = toTilt
            myKit.servo[0].angle = panAngle
            myKit.servo[1].angle = tiltAngle
            print("\tin progress...")
            return True

        #returns false if crosshair already escaped the boundary of imaginary box
        else:
            print("Crosshair no longer in bounding box, stopping oscillations...")
            return False




# MAIN CODE

# reset device to starting position
resetCam()

while display.IsOpen():
    print('NEW FRAME')                                                                  
    img, width, height= cam.CaptureRGBA()
    detections=net.Detect(img, width, height)                       
    
    # Iterating through all detected objects
    for detect in detections:
        # Getting class of detected object
        ID=detect.ClassID
        item=net.GetClassDesc(ID)

        # If detected object is a bird, append it to birdList
        if (item == target):
            birdList.append(detect)
            detected = True
        # If a human is in the frame, break while loop and don't shoot
        if (item == antiTarget):
            print("Human detected!")
            detected = False
            break
            
    # BIRD DETECTED
    if detected == True:
        # reset counter to 0 each time there is a bird detected
        resetCamCounter = 0

        # Updating coordinates of both primary and secondary bounding boxes
        imaginaryBox1(birdList)

        # Drawing crosshair and primary bounding boxes
        jetson.utils.cudaDrawRect(img, (MIDLEFT,MIDBOT,MIDRIGHT,MIDTOP), (255,0,0,200))                                                             # Crosshair
        jetson.utils.cudaDrawRect(img, (imaginaryBox['left'],imaginaryBox['bottom'],imaginaryBox['right'],imaginaryBox['top']), (0,255,255,200))    # PRIMARY bounding box
        # Drawing secondary bounding box when it is in use
        if outOfBounds:
            jetson.utils.cudaDrawRect(img, (theBox['left'],theBox['bottom'],theBox['right'],theBox['top']), (0,255,0,200))                          # SECONDARY bounding box

        # If crosshair is within bounding box, TURN ON laser
        if (imaginaryBox['left'] <= MIDLEFT and imaginaryBox['right'] >= MIDRIGHT and imaginaryBox['top'] <= MIDTOP and imaginaryBox['bottom'] >= MIDBOT and item == target):
            laser_pin.value = True

        # Centering crosshair over middle of whichever bounding box if crosshair was not previously centered
        if not oscillateFlag:
            print("Centering phase")
            oscillateFlag = centerCam()  
        # Oscillating servos to shine laser in random directions once crosshair is centered in previous frame
        else:
            print("Oscillating phase")
            oscillateFlag = oscillate()
            


    # NO BIRD DETECTED
    else:
        # Draw crosshair only
        jetson.utils.cudaDrawRect(img, (MIDLEFT,MIDBOT,MIDRIGHT,MIDTOP), (0,255,0,200))
        laser_pin.value  = False
        resetCamCounter+=1

    # Rendering display and relevant description
    display.RenderOnce(img,width,height)
    dt=time.time()-timeStamp                                    # calculate duration of frame
    timeStamp=time.time()
 
    # Miscellaneous things to ensure before ending iteration
    laser_pin.value  = False
    birdList.clear()
    detected = False
    
    # Resetting device if counter hits limit after no birds detected for some time
    if resetCamCounter == RESETCAMLIMIT:
        outOfBounds = False
        topFlag = bottomFlag = leftFlag = rightFlag = False
        resetCam()
    
    # To exit loop (aka terminate program), press q
    if cv2.waitKey(1)==ord('q'):
        laser_pin.value = False
        resetCam()
        break

# Ensure laser is off
laser_pin.value = False
# Close window
cv2.destroyAllWindows()