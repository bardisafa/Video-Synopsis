# -*- coding: utf-8 -*-
"""
Computer Vision Final Project

Video Synopsis

Bardia Safaei 95101909

"""

# import packages
import cv2
import numpy as np
from scipy.spatial.distance import cdist




# path of video
video_path = 'Video2.avi'


# read video
cap = cv2.VideoCapture(video_path)

# properties of video
fps = cap.get(cv2.CAP_PROP_FPS)
video_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
video_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

fourcc = cv2.VideoWriter_fourcc(*'XVID')

video_size = (int(video_w), int(video_h))


# output video
output_path = 'Summerized ' + video_path
output = cv2.VideoWriter(output_path, fourcc, 30.0, (800, 480))
 

# initialize our background subtraction method
subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16,
                                          detectShadows=True)

# get first frame as static backgound
ret, BG = cap.read()


if ret is False:
    print("Unable to open the file")


# parameters
num_frame = 0
obj_id = 0    # number of distinct objects in entire video
update_bg_rate = 20
alpha = 0.1    # factor of combining backgrounds
BG_frame_dynamic = np.zeros((int(video_h),    # dynamic background of each frame
                             int(video_w),
                             3))  
BG_frame = np.zeros((int(video_h),    # static background of each frame
                     int(video_w),
                     3))
blob_dict_prev = {}    # to save previous frame's information    
tracking = []    # to keep track of ids in order to be able to add new ids
obj_dict = {}         # dictionary of objects.     


while (cap.isOpened):
    
    ret, frame = cap.read()
    
    if ret is True:
        
        '''
        -----------------
         apply gaussian filter to blur frame
        -----------------
        '''
        frame_blur = cv2.GaussianBlur(frame, (9, 9), 0)
        cv2.imshow("MASK", frame_blur)
        
        '''
        ---------------
        apply MOG2 algorithm and gaussian filter
        ---------------
        '''
        fg_mask = subtractor.apply(frame_blur)    # foreground mask
        fg_mask_blur = cv2.GaussianBlur(fg_mask,    # after gaussian filter
                                        (15, 15), 0)    
                                                                
        '''
        ---------------
        thresholding : intensity>150 --> white
        ---------------
        '''
        _, fg_mask_thresh = cv2.threshold(fg_mask_blur,
                                          150, 255, cv2.THRESH_BINARY)
        
        cv2.imshow('before dilation', fg_mask_thresh)
        
        '''
        -------------------
        morphological operations
        -------------------
        '''
        kernel_size = 7
        element = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_size, kernel_size))
        dilated_mask = cv2.dilate(fg_mask_thresh, element, iterations = 1)
        
        cv2.imshow("dilated mask", dilated_mask)
        
        '''
        ---------------------------------------
        Finding contours for the dilated mask
        ---------------------------------------
        '''
        img2, contours, hierarchy = cv2.findContours(dilated_mask,
                                                     cv2.RETR_TREE,
                                                     cv2.CHAIN_APPROX_SIMPLE)
        '''
        ------------------------------------------
        # create hull array for convex hull points
        ------------------------------------------
        '''
        hull = []
        
        # calculate points for each contour
        for i in range(len(contours)):
            
            if cv2.contourArea(contours[i]) >= 0:
                # creating convex hull object for each contour
                hull.append(cv2.convexHull(contours[i], False))
        
        '''
        -----------------------
        draw the convex hall
        -----------------------
        '''
        # create an empty black image
        drawing = np.zeros((dilated_mask.shape[0], dilated_mask.shape[1], 3),
                            np.uint8)
        Mask = np.zeros((dilated_mask.shape[0], dilated_mask.shape[1], 3),
                            np.uint8)
        # copy of frame. we know drawContours changes main frame
        frame2 = frame.copy()
        # draw contours and hull points
        for i in range(len(contours)):
            color_contours = (0, 255, 0)    # green color for contours
            color = (255, 255, 255)    # blue color for convex hull
            # draw ith contour
            cv2.drawContours(drawing, contours, i, color_contours, 1,
                             8, hierarchy)
            cv2.drawContours(frame2, contours, i, color_contours, 1,
                             8, hierarchy)
            # draw ith convex hull object
            cv2.drawContours(drawing, hull, i, color, 1, 8)
            cv2.drawContours(Mask, hull, i, color, -1, 8)            
            
        
        # show
        cv2.imshow("contours & hulls", drawing)
        cv2.imshow("contours of moving objects", frame2)
        
        
        '''
        -----------------------------
        update background
        -----------------------------
        '''
        if not (num_frame % update_bg_rate):
            
            '''
            -------------------------------------
            # obtain moving objects of the frame
            -------------------------------------
            '''
            Mask_normalized = (Mask / 255).astype(np.uint8)
            
            # objects of frame
            frame_obj = cv2.multiply(frame, Mask_normalized)
            frame_obj.astype(np.uint8)
            
            cv2.imshow("Moving objects", frame_obj)
            
            '''
            -------------------------------------
            # obtain static background of the frame
            -------------------------------------
            '''
            Mask_inv = cv2.bitwise_not(Mask)
            
            cv2.imshow("Inverse of Mask", Mask_inv)
            
            
            Mask_inv_norm = (Mask_inv / 255).astype(np.uint8)     # Normalized Inverse Mask.
            BG_frame = cv2.multiply(frame, Mask_inv_norm)
            
            # Convert np.array of type float64 to type uint8:
            BG_frame = BG_frame.astype(np.uint8)
            
            # Show Static Background in Each Frame:
            cv2.imshow("Static Background in Each Frame", BG_frame)
            
            
            # replace moving objects in each frame with corresponding parts of
            # static background
            BG_frame_move_repl = cv2.multiply(BG, Mask_normalized)
            
            # Convert np.array of type float64 to type uint8:
            BG_frame_move_repl = BG_frame_move_repl.astype(np.uint8)
            
            # replacement of moving objects in each frame:
            cv2.imshow("Replace Moving Objects in Each Frame",
                       BG_frame_move_repl)
            
            
            
            BG_frame_final = cv2.addWeighted(BG_frame, 1,
                                             BG_frame_move_repl, 1, 0)
            
            cv2.imshow("final background in each frame",
                       BG_frame_final)
            
            
            BG = cv2.addWeighted(BG, 0.9, BG_frame_final, 0.1, 0)
            BG = BG.astype(np.uint8)
            BG1 = BG.copy()
            cv2.putText(BG1, f'Frame Number :  {num_frame}', (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow("Updating Background", BG1)
            
            
        '''
        ---------------------------------------------------
        create a dictionary of moving objects' attributes
        ---------------------------------------------------
        '''
        blob_dict = {}                  # contains attributes of each blob in current frame
        blob_dict['center'] = []        # contains center of each blob
        blob_dict['corners'] = []       # contains corners of each blob
        blob_dict['area'] = []          # contains total number of pixels for each blob
        blob_dict['id'] = []            # contains id of each blob
        blob_dict['frame count'] = []   # contains frame count
        
        # Link : https://www.learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
        blob_id = 1
        for i in range(len(contours)):
            
            # empty frame
            blob = np.zeros((dilated_mask.shape[0], dilated_mask.shape[1]),
                        np.uint8)
            # mask of each blob in current frame using hull
            cv2.drawContours(blob, hull, i, color, -1, 8)
            
            # calculate moments of binary blob
            M = cv2.moments(contours[i])
            
            # calculate x, y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # convert grayscale to RGB and putting text
            blob = cv2.cvtColor(blob, cv2.COLOR_GRAY2RGB)
            
            cv2.circle(blob, (cX, cY), 5, (0, 255, 0), -1)
            cv2.putText(blob, "Centroid", (cX-20, cY-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # show
            cv2.imshow("centroids", blob)
            
            
            # area of each contour
            area = cv2.contourArea(contours[i])
            
            
            '''
            -------------------------
            fill in the dictionary
            -------------------------
            '''
            
            # prune off too small or too big objects
            minArea = 200
            maxArea = 200000
            
            if minArea < area < maxArea:
                blob_dict['area'].append(area)
                blob_dict["frame count"].append(num_frame)
                blob_dict['corners'].append(hull[i])
                blob_dict['id'].append(blob_id)
                blob_dict["center"].append([cX, cY])
                
                blob_id += 1
                
        '''
        --------------------------
        tracking objects
        --------------------------
        '''
        # relabling id field starting from zero for convenience (only for 1st frame)
        if (num_frame == 0) and (len(blob_dict['id']) > 0):
            for i in range(len(blob_dict['id'])):
                # blob_dict['id'][i] -= 1
                blob_dict['id'][i] = obj_id
                obj_id += 1
                
        elif len(blob_dict['id']) > 0:
            min_dist_id = -1
            min_dist = 0
                    
            if len(blob_dict_prev) > 0:
                if len(blob_dict_prev['id']) > 0:
                
                    for cnt in range(len(blob_dict['id'])):
                        
                        old_centers = blob_dict_prev['center']
                        new_center = blob_dict['center'][cnt]
                        
                        distances = cdist([new_center], old_centers)
                        nearest_idx = distances.argmin()
                        
                        min_dist = distances[0][nearest_idx]
                        min_dist_id = blob_dict_prev['id'][nearest_idx]
                        
                        '''
                        ---------------------------------------------------------
                        make sure that each old id has exactly one corresponding 
                        new id in new frame's objects
                        ---------------------------------------------------------
                        '''
                        flag = True
                        if min_dist_id != -1:
                            for i in range(len(tracking)):
                                if tracking[i][0] == min_dist_id:
                                    if min_dist < tracking[i][2]:
                                        tracking[i][2] = min_dist
                                        tracking[i][1] = blob_dict['id'][cnt]
                                    flag = False
                                    break
                            if flag:
                                Correspond = []
                                Correspond.append(min_dist_id)
                                Correspond.append(blob_dict['id'][cnt])
                                Correspond.append(min_dist)
                                tracking.append(Correspond)
                                
            '''
            --------------------------------------------------------------
            label new objects which are not associated with existing ids
            --------------------------------------------------------------
            '''
            for i in range(len(blob_dict['id'])):
                flag1 = False    # indicates whether an object is found or not
                for obj in range(len(tracking)):
                    if blob_dict['id'][i] == tracking[obj][1]:
                        flag1 = True
                        blob_dict['id'][i] = tracking[obj][0]
                        break
                if not flag1:
                    blob_dict['id'][i] = obj_id
                    obj_id += 1
            
            '''
            -----------------------------------------------
            show the results
            -----------------------------------------------
            '''
            # create an empty black image
            colored_output = np.zeros((dilated_mask.shape[0],
                                       dilated_mask.shape[1], 3), np.uint8)
            Time = num_frame / fps
            for i in range(len(blob_dict['id'])):
                label = blob_dict['id'][i]
                
                # create mask for each object in current frame:
               
                cv2.drawContours(colored_output, blob_dict['corners'], i, (255,255,255), -1, 8)
                
                # obtain time of frame and put this on each objects:
                second = str(int(Time % 60))
                minute = str(int(Time / 60))
                if len(second) < 2:
                    second = '0' + second
                if len(minute) < 2:
                    minute = '0' + minute
                time_str = minute + ':' + second
                center = (blob_dict['center'][i][0],blob_dict['center'][i][1])
                # cv2.putText(colored_output, time_str, center,
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0),2)
                cv2.putText(colored_output, f'Id {label}', center,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0),2)
               
            cv2.imshow("IDs of Object", colored_output) 
            
        
        '''
        ---------------------------------------------------
        save objects based on their ids
        ---------------------------------------------------
        '''
        if len(blob_dict['id']) > 0:
            for i in range(len(blob_dict['id'])):
                ID = blob_dict['id'][i]
                hull = blob_dict['corners']
                
                # mask for each object of the frame
                mask = np.zeros((dilated_mask.shape[0],
                                       dilated_mask.shape[1], 3), np.uint8)
                cv2.drawContours(mask, hull, i, (255, 255, 255), -1, 8)
                
                mask_norm = (mask / 255).astype(np.uint8)     # Normalized Mask.
                colored_obj = cv2.multiply(frame, mask_norm)
            
                # Convert np.array of type float64 to type uint8:
                colored_obj = colored_obj.astype(np.uint8)
                
                
                
                '''
                ----------------------------------
                create a dictionary of all objects in the video grouped by IDs
                ----------------------------------
                '''
                IDs = [Id for Id in obj_dict]
                
                # we had object with ID in previous frames
                if ID in IDs:
                    obj_dict[ID]['center'].append(blob_dict['center'][i])
                    obj_dict[ID]['frame'].append(blob_dict['frame count'][i])
                    obj_dict[ID]['corners'].append(hull[i])
                    obj_dict[ID]['obj'].append(colored_obj)
                    
                # object with ID has not arrived so far
                else:
                    obj_dict[ID] = {}
                    obj_dict[ID]['center'] = [blob_dict['center'][i]]
                    obj_dict[ID]['frame'] = [blob_dict['frame count'][i]]
                    obj_dict[ID]['corners'] = [hull[i]]
                    obj_dict[ID]['obj'] = [colored_obj]
        
    else:
        break
    
    k = cv2.waitKey(1)
    if k == ord('e'):
        break
    
    blob_dict_prev = blob_dict.copy()
    blob_dict.clear()
    tracking.clear()
    num_frame += 1
    '''
    -------------------------
    END OF WHILE LOOP!
    -------------------------
    '''
    


# release
cap.release()
cv2.destroyAllWindows()

# save final background
bg_path = 'Final Background of' + video_path.replace('.avi', '') + '.jpg'
cv2.imwrite(bg_path, BG)
print("Video Processing Finished")



'''
----------------------------------------------
video synopsis 
----------------------------------------------
'''

'''
------------------------------------
functions for detecting color of objects
------------------------------------
'''

# Converts images to masks of colors
def img2Color(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_mask = {}
    color_mask['Red'] = cv2.inRange(hsv_frame, (0, 100, 100), (10, 255, 255)) | cv2.inRange(hsv_frame, (160, 100, 100), (179, 255, 255))
    color_mask['Yellow'] = cv2.inRange(hsv_frame, (28, 100, 100),
                                       (31, 255, 255))
    color_mask['Green'] = cv2.inRange(hsv_frame, (32, 100, 100),
                                      (90, 255, 255))
    color_mask['Blue'] = cv2.inRange(hsv_frame, (91, 100, 100),
                                     (130, 255, 255))
    color_mask['Purple'] = cv2.inRange(hsv_frame, (131, 100, 100),
                                       (159, 255, 255))
    color_mask['Black'] = cv2.inRange(hsv_frame, (0, 0, 0),
                                      (180, 200, 10))
    color_mask['White'] = cv2.inRange(hsv_frame, (0, 0, 210),
                                      (180, 10, 255))
    return color_mask



# remove objects which appear in less than 10 frames (fake objects)
obj_synopsis = {}    # dictionary of real objects

for Id in obj_dict:
    if len(obj_dict[Id]['center']) > 10:
        obj_synopsis[Id] = obj_dict[Id]
del obj_dict


max_objs_in_frame = 8    # maximum number of objects in one frame

IDs = [Id for Id in obj_synopsis]
flag_IDs = np.ones((1, len(IDs)))
velocity = np.zeros((1, len(IDs)))
# create frames of summerized video
while True:
    if obj_synopsis:
        frame_synopsis = BG.copy()
        
        # get objects associated with 8 ids
        if len(obj_synopsis) >= max_objs_in_frame:
            IDs_synopsis = IDs[:max_objs_in_frame]
        else:
            IDs_synopsis = IDs
            
        for Id in IDs_synopsis:
            
            # calculate relative velocity
            if flag_IDs[0, IDs.index(Id)]:
                
                flag_IDs[0, IDs.index(Id)] = 0
                
                first_frame = obj_synopsis[Id]['frame'][0]
                last_frame = obj_synopsis[Id]['frame'][-1]
                
                [cx1, cy1] = obj_synopsis[Id]['center'][0]
                [cx2, cy2] = obj_synopsis[Id]['center'][-1]
                
                displacement = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
                velocity[0, IDs.index(Id)] = displacement/(last_frame - first_frame)
            # center of each object
            [cx, cy] = obj_synopsis[Id]['center'][0]
            
            # frame number of each object
            num_frame = obj_synopsis[Id]['frame'][0]
            
            
            # corners of each object
            hull = obj_synopsis[Id]['corners']
            
            # mask for each object
            mask_inv = np.ones((dilated_mask.shape[0],
                                dilated_mask.shape[1],
                                3), np.uint8)
            mask = np.zeros((dilated_mask.shape[0],
                                dilated_mask.shape[1]), np.uint8)
            
            # object's location is white
            cv2.drawContours(mask, hull, 0, (255, 255, 255), -1, 8)
            # object's location is black
            cv2.drawContours(mask_inv, hull, 0, (0, 0, 0), -1, 8)
            frame_synopsis = cv2.multiply(frame_synopsis, mask_inv)
            frame_synopsis = frame_synopsis.astype(np.uint8)
            
            # add object to frame_synopsis using mask saved in our dictionary
            frame_synopsis = cv2.add(frame_synopsis,
                                     obj_synopsis[Id]['obj'][0])
            
            # obtain color of object
            obj_colors = img2Color(obj_synopsis[Id]['obj'][0])
            
            score_max = -1
            color_max = 'Black'
            for color in obj_colors:
                color_score = np.sum(mask * obj_colors[color])
                if color_score > score_max:
                    score_max = color_score
                    color_max = color
            
            
            # using frame number and fps to obtain time
            Time = num_frame / fps
            
            second = str(int(Time % 60))
            minute = str(int(Time / 60))
            if len(second) < 2:
                second = '0' + second
            if len(minute) < 2:
                minute = '0' + minute
            time_str = minute + ':' + second
            
            # add time to frame_synopsis
            cv2.putText(frame_synopsis, time_str + '  ' + color_max, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame_synopsis, str(velocity[0, IDs.index(Id)])[:4],
                        (cx-20, cy-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            if len(obj_synopsis[Id]['center']) == 1:
                # remove object from lists
                IDs.remove(Id)
                obj_synopsis.pop(Id)
            else:
                obj_synopsis[Id]['center'].pop(0)
                obj_synopsis[Id]['frame'].pop(0)
                obj_synopsis[Id]['corners'].pop(0)
                obj_synopsis[Id]['obj'].pop(0)
        
        # add this frame to output video
        output.write(frame_synopsis)
                
            
    else:
        break
            
output.release()
print("Summerized Video Is Ready!")


            

