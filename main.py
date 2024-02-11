import cv2
import pytesseract

# Set the path to the Tesseract OCR executable (change this according to your installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_roi(image_path, roi_coordinates):
    # Read the image
    image = cv2.imread(image_path)

    # Extract the region of interest (ROI)
    x, y, w, h = roi_coordinates
    roi = image[y:y + h, x:x + w]

    # Convert the ROI to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply any additional preprocessing if needed (e.g., thresholding, filtering)

    # Use Tesseract OCR to extract text
    text = pytesseract.image_to_string(gray_roi)

    return text

if __name__ == "__main__":
    # Specify the image path and ROI coordinates (x, y, width, height)
    image_path = "./image.png"
    roi_coordinates = (100, 50, 300, 200)  # Adjust these coordinates based on your image

    # Extract text from the specified ROI
    extracted_text = extract_text_from_roi(image_path, roi_coordinates)

    # Print the extracted text
    print("Extracted Text:")
    print(extracted_text)
