import cv2
import torch
import numpy as np
from tracker import *
from collections import defaultdict

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap=cv2.VideoCapture('highway.mp4')
track_history = defaultdict(lambda: [])

count=0
tracker = Tracker()
route_dict = {
    "EA": 0,
    "CA": 0,
    "BC": 0,
    "ED": 0,
    "BF": 0,
    "CF": 0,
}

def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)

#the yellow rectangle is reffered to as polylines .for that we need 4 coordinates 
areaA=[(207,77),(515,85),(507,103),(173,105)]
areaB=[(540,82),(778,105),(829,132),(527,107)]
areaC=[(22,122),(7,335),(85,335),(163,99)]
areaD=[(7,337),(6,593),(121,595),(130,339)]
areaE=[(845,334),(868,598),(1010,594),(1006,337)]
areaF=[(869,133),(998,154),(1014,302),(884,302)]
area_A=set()
area_B=set()
area_C=set()
area_D=set()
area_E=set()
area_F=set()
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
        # x=int(rows[0]) #xmin     
        # y=int(rows[1]) #ymin
        # x1=int(rows[2]) #xmax
        # y1=int(rows[3]) #ymax
        # b=str(rows['name']) #name - car or person etc
        x = int(rows.iloc[0]) #xmin
        y = int(rows.iloc[1]) #ymin
        x1 = int(rows.iloc[2]) #xmax
        y1 = int(rows.iloc[3]) #ymax
        #cv2.rectangle(frame,(x,y),(x1,y1),(0,0,255),3) #to add red bounding boxes on detected objects
        list.append([x,y,x1,y1] ) #inside list pass 4 coordinates [x,y,x1,y1]
        idx_bbox=tracker.update(list) #call the tracker function to update . inside this we pass list 
        #print(idx_bbox) # so now we get 4 coordinates display
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
        result_A=cv2.pointPolygonTest(np.array(areaA,np.int32),(x3,y3),False)
        result_B=cv2.pointPolygonTest(np.array(areaB,np.int32),(x3,y3),False)
        result_C=cv2.pointPolygonTest(np.array(areaC,np.int32),(x3,y3),False)
        result_D=cv2.pointPolygonTest(np.array(areaD,np.int32),(x3,y3),False)
        result_E=cv2.pointPolygonTest(np.array(areaE,np.int32),(x3,y3),False)
        result_F=cv2.pointPolygonTest(np.array(areaF,np.int32),(x3,y3),False)
        if result_A>0:
            if id in area_E:
                route_dict["EA"] += 1
            if id in area_C:
                route_dict["CA"] += 1
            area_A.add(id) #pass id inside set when circle enters polyline 
        if result_B>0:
            area_B.add(id)   
        if result_C>0:
            if id in area_B:
                route_dict["BC"] += 1
            area_C.add(id) #pass id inside set when circle enters polyline 
        if result_D>0:
            if id in area_E:
                route_dict["ED"] += 1
            area_D.add(id) 
        if result_E>0:
            area_E.add(id) #pass id inside set when circle enters polyline 
        if result_F>0:
            if id in area_B:
                route_dict["BF"] += 1
            if id in area_C:
                route_dict["CF"] += 1
            area_F.add(id) 

    
    # to draww polyline we call polylines function
    cv2.polylines(frame,[np.array(areaA,np.int32)],True,(0,255,255),2)
    cv2.polylines(frame,[np.array(areaB,np.int32)],True,(0,255,255),2)
    cv2.polylines(frame,[np.array(areaC,np.int32)],True,(0,255,255),2)
    cv2.polylines(frame,[np.array(areaD,np.int32)],True,(0,255,255),2)
    cv2.polylines(frame,[np.array(areaE,np.int32)],True,(0,255,255),2)
    cv2.polylines(frame,[np.array(areaF,np.int32)],True,(0,255,255),2)
    a1=len(area_A)
    cv2.putText(frame,str(a1),(210,38),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2) #count number display coordinates (210,38)
    a2=len(area_B)
    cv2.putText(frame,str(a2),(805,36),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2) #count number display coordinates (805,36)
    a3=len(area_C)
    cv2.putText(frame,str(a3),(50,61),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2)
    a4=len(area_D)
    cv2.putText(frame,str(a4),(239,578),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2)
    a5=len(area_E)
    cv2.putText(frame,str(a5),(701,587),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2)
    a6=len(area_F)
    cv2.putText(frame,str(a6),(943,56),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2)

    cv2.imshow("FRAME",frame)
    if cv2.waitKey(6)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()

for key, value in route_dict.items():
    print(f"Route {key}: {value}")