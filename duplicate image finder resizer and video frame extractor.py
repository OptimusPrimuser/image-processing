import os
from cv2 import cv2
import numpy as np

#path to video files folder 
video_files=os.listdir("example_clips") 
video_frame=[]
#path to output folder
output_path="C:\\Users\\Optimus\\Desktop\\lol\\keras-video-classification\\test\\"

#getting frames from videos and keeping them seperate video_frame[video_index][frame of video]
for i in range(len(video_files)):
    path="example_clips\\"+video_files[i]
    cap=cv2.VideoCapture(path)
    #making video_frame[video_index]
    video_frame.append([])
    while(cap.isOpened()):
        #getting every 25 frames
        ret, frame = cap.read()
        for d in range(24):
            ret, frame = cap.read()
        if ret == False:
            break
        #resizing frame    
        frame=cv2.resize( frame,(224,224))        
        #adding frame to video_frame[video_index]
        video_frame[i].append(frame)
    
    print(np.array(video_frame[i]).shape) 
    cap.release()

#image indexing
image_no=0
#getting similar images in one place similar_images[video_index][frames of video]
similar_images=[]
#getting processed frames of a video in one place done_images[video_index][frames of video]
done_images=[]
print("please wait")
for c in range(len(video_frame)):
    #Making similar_images[video_index]
    similar_images.append([])
    #making done_images[video_index]
    done_images.append([])
    for b in range(len(video_frame[c])):
        #if image is present in similar_images[video_index] or done_images[video_index]
        #                then skip
        if(b in similar_images[c] or b in done_images[c]):
            continue
        #loading image1 as grayscale
        img1=video_frame[c][b]
        img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        width,height=img1.shape
        for a in range(len(video_frame[c])):
            if(a==b):
                continue
            #loading image2 as gray scale
            img2=video_frame[c][a]
            img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
            #getting difference between image1 and image 2, and using it to detect duplicate images
            count=np.subtract(img1,img2)
            diffrence=(np.sum(count**2)/(width*height))**0.5
            if diffrence<7.2:
                if cv2.waitKey(1000)==ord("q"):
                    exit()
                #adding frame to similar_images[video_index]
                similar_images[c].append(a)
        image_no=image_no+1
        #saving the processed image and adding it to done_images[video_index]
        cv2.imwrite(output_path+str(image_no)+'.jpg',video_frame[c][b])
        similar_images[c].append(b)
