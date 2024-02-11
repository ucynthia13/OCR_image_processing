import cv2
import pytesseract
tesseract_path = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe" 
pytesseract.pytesseract.tesseract_cmd = tesseract_path

img = cv2.imread("./image.png")
img = cv2.resize(img, (1000, 450))
cv2.imshow("Image", img)
text = pytesseract.image_to_string(img)
print(text)
cv2.waitKey(0)
cv2.destroyAllWindows()