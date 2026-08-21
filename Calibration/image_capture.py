import cv2

cap = cv2.VideoCapture(4)  
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

img_count = 0
print("Press SPACE to capture, ESC to finish")
print("Move chessboard to different positions/orientations")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Draw guide
    cv2.putText(frame, f"Images captured: {img_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imshow('Capture', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # SPACE to capture
        filename = f"calib_{img_count:03d}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        img_count += 1
        
    elif key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
print(f"Captured {img_count} images")
