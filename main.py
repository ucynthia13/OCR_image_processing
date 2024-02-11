import cv2
import pytesseract

# Configure Tesseract path (update if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

# Load the image
image = cv2.imread("./image.png")

# Preprocess the image
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresholded_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Find contours (adjust for potential multiple digits)
contours = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

# Extract the largest region of interest (ROI)
largest_contour = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(largest_contour)
roi = image[y:y+h, x:x+w]

# Perform OCR on the ROI
extracted_text = pytesseract.image_to_string(roi, config='--psm 10')

# Display the extracted text
print(extracted_text)
