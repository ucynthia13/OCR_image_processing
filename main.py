import cv2
import pytesseract

# Specify the Tesseract path
tesseract_path = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"  # Adjust path if needed
pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Load the image
img = cv2.imread("./image.png")

# Preprocess the image for better OCR results
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Find the single largest contour (assuming a single ROI)
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
largest_cnt = max(cnts, key=cv2.contourArea)

# Extract the ROI based on the largest contour
x, y, w, h = cv2.boundingRect(largest_cnt)
roi = img[y:y+h, x:x+w]

# Perform OCR on the ROI
text = pytesseract.image_to_string(roi, config='--psm 10')

# Display the extracted text
print(text)

# Optionally display the image with the ROI highlighted (for verification)
if cv2.waitKey(0) == ord("q"):  # Quit on 'q' key press
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw a green rectangle around the ROI
    cv2.imshow("Image with ROI", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
