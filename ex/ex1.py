from ultralytics import YOLO
import cv2
import torch

model = YOLO("yolo11x-pose.pt")
count=0
path=r".\ex1-26.png"
results = model(path,conf=0.25)
# 1人目のキーポイント(x,y)を獲得する．
# 2人目はresults[0].keypoints.xy[1]に格納されている．
img = results[0].orig_img

# 認識した物体領域を取得する．
boxes = results[0].boxes

for box in boxes:
    if int(box.cls[0])==0:
        count+=1

    # 物体領域のxy座標を取得する．
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        thickness=3,
    )

cv2.putText(
    img,                     
    f"person: {count}",       
    (30, 50),                
    cv2.FONT_HERSHEY_SIMPLEX, 
    1.5,                     
    (0, 0, 255),             
    3                        
)
    
cv2.imshow("", img)
cv2.waitKey(0)
cv2.destroyAllWindows()