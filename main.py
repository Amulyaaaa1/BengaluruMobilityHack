import cv2
import torch
import numpy as np
from tracker import *


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap=cv2.VideoCapture('highway.mp4')

count=0
tracker = Tracker()

def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)

#the yellow rectangle is reffered to as polylines .for that we need 4 coordinates 
area1=[(207,77),(515,85),(507,103),(173,105)]
area2=[(540,82),(778,105),(829,132),(527,107)]
area3=[(22,122),(7,257),(85,248),(163,99)]
area4=[(7,337),(6,593),(121,595),(130,339)]
area5=[(845,334),(868,598),(1010,594),(1006,337)]
area6=[(869,133),(998,154),(1014,302),(884,302)]
area_1=set()
area_2=set()
area_3=set()
area_4=set()
area_5=set()
area_6=set()
while True:
    ret,frame=cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,600))
    
    results=model(frame)
    list=[] #make a list 
    cv2.putText(frame, "A", (207,75), cv2.FONT_HERSHEY_SIMPLEX, 1, (234, 123, 250), 2)
    cv2.putText(frame, "B", (540,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (234, 123, 250), 2)
    cv2.putText(frame, "C", (22,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (234, 123, 250), 2)
    cv2.putText(frame, "D", (7,335), cv2.FONT_HERSHEY_SIMPLEX, 1, (234, 123, 250), 2)
    cv2.putText(frame, "E", (845,332), cv2.FONT_HERSHEY_SIMPLEX, 1, (234, 123, 250), 2)
    cv2.putText(frame, "F", (869,131), cv2.FONT_HERSHEY_SIMPLEX, 1, (234, 123, 250), 2)
    for index,rows in results.pandas().xyxy[0].iterrows():
       if rows['name'] == 'person':
            continue 
       x=int(rows[0]) #xmin     
       y=int(rows[1]) #ymin
       x1=int(rows[2]) #xmax
       y1=int(rows[3]) #ymax
       b=str(rows['name']) #name - car or person etc
       #cv2.rectangle(frame,(x,y),(x1,y1),(0,0,255),3) #to add red bounding boxes on detected objects
       list.append([x,y,x1,y1] ) #inside list pass 4 coordinates [x,y,x1,y1]
       idx_bbox=tracker.update(list) #call the tracker function to update . inside this we pass list 
      # print(idx_bbox) # so now we get 4 coordinates display
      #id is the number given to different type of object detected ex - 0 is car , 1 is person etc 
    for bbox in idx_bbox:
        x2,y2,x3,y3,id=bbox
        cv2.rectangle(frame,(x2,y2),(x3,y3),(0,0,255),2) # here 2 is thickness of red box ..(0,0,255) is red colour 
        #we draw rectangle / bounding box with help of tracker
        #to put id on rectangle use putText 
        #######(to display id above frame)cv2.putText(frame,str(id),(x2,y2),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2) #on bounding box frame we want is so use str(id) that is string id 
        #for font use cv2.FONT_HERSHEY_PLAIN # scale is 3 #color is blue (255,0,0) # thickness is 2
       #at this stage id that is in blue keeps changing soo we draw a rectangular box in yellow .so when vehicle passes throught that recatangle then counter is incremented
        #to draw circle
        cv2.circle(frame,(x3,y3),4,(0,255,0),-1) #if (x3,y3) coordinates inside the polyline then count. make radius of the circle 4 .color green.to fill the circle -1
        result=cv2.pointPolygonTest(np.array(area1,np.int32),(x3,y3),False)
        result1=cv2.pointPolygonTest(np.array(area2,np.int32),(x3,y3),False)
        result2=cv2.pointPolygonTest(np.array(area3,np.int32),(x3,y3),False)
        result3=cv2.pointPolygonTest(np.array(area4,np.int32),(x3,y3),False)
        result4=cv2.pointPolygonTest(np.array(area5,np.int32),(x3,y3),False)
        result5=cv2.pointPolygonTest(np.array(area6,np.int32),(x3,y3),False)
        if result>0:
            area_1.add(id) #pass id inside set when circle enters polyline 
        if result1>0:
            area_2.add(id)   
        if result2>0:
            area_3.add(id) #pass id inside set when circle enters polyline 
        if result3>0:
            area_4.add(id) 
        if result4>0:
            area_5.add(id) #pass id inside set when circle enters polyline 
        if result5>0:
            area_6.add(id) 

    
    # to draww polyline we call polylines function
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,255,255),2)
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,255,255),2)
    cv2.polylines(frame,[np.array(area3,np.int32)],True,(0,255,255),2)
    cv2.polylines(frame,[np.array(area4,np.int32)],True,(0,255,255),2)
    cv2.polylines(frame,[np.array(area5,np.int32)],True,(0,255,255),2)
    cv2.polylines(frame,[np.array(area6,np.int32)],True,(0,255,255),2)
    a1=len(area_1)
    cv2.putText(frame,str(a1),(210,38),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2) #count number display coordinates (210,38)
    a2=len(area_2)
    cv2.putText(frame,str(a2),(805,36),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2) #count number display coordinates (805,36)
    a3=len(area_3)
    cv2.putText(frame,str(a3),(50,61),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2)
    a4=len(area_4)
    cv2.putText(frame,str(a4),(239,578),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2)
    a5=len(area_5)
    cv2.putText(frame,str(a5),(701,587),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2)
    a6=len(area_6)
    cv2.putText(frame,str(a6),(943,56),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2)

    cv2.imshow("FRAME",frame)
    if cv2.waitKey(6)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()