from ultralytics import YOLO
import cv2
import torch

model = YOLO("best260408.pt")
count=0
path=r".\ex2-26.png"
print(model.names)
results = model(path, conf=0.1, imgsz=1536)

img = results[0].orig_img

boxes = results[0].boxes

for box in boxes:
    cls=int(boxes.cls[0])

    if cls==0:
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
print("Helmet count:",count)
    
cv2.imshow("", img)
cv2.waitKey(0)
cv2.destroyAllWindows()