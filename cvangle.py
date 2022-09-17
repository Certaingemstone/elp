import cv2
import numpy as np
import csv

np.seterr(all='raise')

display = False
outfile = r"C:\Users\jade2\Downloads\MIT\FA2022\JLab\pendulum\C0424.csv"
filename = r"C:\Users\jade2\Downloads\MIT\FA2022\JLab\pendulum\C0424.MP4"
canny_thresh1 = 100
canny_thresh2 = 300
hough_thresh = 400
angular_res = 0.001 # about 0.05 degree resolution
expected_lines = 2 # left and right edge of the paracord

# Data
times = []
rhos = []

# Video reading
capture = cv2.VideoCapture(filename)
fr = float(capture.get(cv2.CAP_PROP_FPS))
print(f"Initialized Capture on {filename}, {fr} frame/s")

# Processing
idx = 0
frame_time = 1 / fr

while capture.isOpened():
    # get a frame
    ret, frame = capture.read()
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if not ret:
        print(f"Frame stream end. Exiting.")
        break
    # convert image to grayscale
    framegs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # perform edge detection
    dst = cv2.Canny(framegs, canny_thresh1, canny_thresh2, None, 3)
    # apply Hough transform for lines
    lines = cv2.HoughLines(dst, 1, angular_res, hough_thresh)

    # record data
    rho_temp = []
    for i in range(len(lines)):
        rho = lines[i][0][0]
        rho_temp.append(rho)
        if display:
            theta = lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(frame, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

    # split the lines between left and right side of pendulum by clustering on rho:
    rho_temp = np.array(rho_temp)
    midpt = 0.5 * (np.max(rho_temp) - np.min(rho_temp)) + np.min(rho_temp)
    mask = rho_temp > midpt
    try:
        above = np.mean(rho_temp[mask])
        below = np.mean(rho_temp[np.invert(mask)])
    except:
        print("Warning: either no upper or lower edge has been detected. Change thresholds.")

    # get the average (rho for the overall line)
    final_rho = 0.5 * (above + below)
    rhos.append(final_rho)
    times.append(frame_time * idx)
    
    # show the stuff
    print(f"Frame {idx}, {final_rho}")
    if display:
        imS = cv2.resize(frame, (480, 640)) 
        cv2.imshow("Lines", imS)
        cv2.waitKey()
    idx += 1

capture.release()

# Export data to csv
with open(outfile, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for t, r in zip(times, rhos):
        writer.writerow([t, r])
    print("File written to", outfile)