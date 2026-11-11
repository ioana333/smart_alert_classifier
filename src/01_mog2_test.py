import cv2

PATH = "data/test.avi"
RESULT_PATH = "results/mog2"
TRESHOLD = 5

# deschidere videoclip
cap = cv2.VideoCapture(PATH)
if not cap.isOpened():
    raise SystemExit("path not found")

# implementare MOG2
mog2 = cv2.createBackgroundSubtractorMOG2(
    history=500,            #cate cadre ia in considerare
    varThreshold=TRESHOLD,  #cat de diferit sa fie un pixel ca sa fie considerat miscare
    detectShadows=True      #ia in considerare umbrele ca sa nu le clasifice drept miscare
)

frame_nr = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break       #s-a terminat videoclipul

    mask = mog2.apply(frame)    #alb = miscare & negru = fundal

    #salvare cadre pentru analiza
    if frame_nr in (100, 200, 300): 
        cv2.imwrite(f"{RESULT_PATH}/frame_{frame_nr}_original(th={TRESHOLD}).png", frame)
        cv2.imwrite(f"{RESULT_PATH}/frame_{frame_nr}_mask(th={TRESHOLD}).png", mask)

    #afisare
    cv2.imshow("Original", frame)
    cv2.imshow("Masca MOG2", mask)

    # iesire pe Q
    if cv2.waitKey(30) and 0xFF == ord("q"):
        break

    frame_nr += 1

cap.release()
cv2.destroyAllWindows()
print(f"END\n{frame_nr} frames processed")
        

