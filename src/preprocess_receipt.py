import cv2
import numpy as np

def deskew(image):
    """Deskew image using Hough transform to estimate dominant angle."""
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

    if lines is None:
        return image

    angles = []
    for rho, theta in lines[:, 0]:
        angle = (theta * 180 / np.pi) - 90
        if -45 < angle < 45:
            angles.append(angle)

    if len(angles) == 0:
        return image

    median_angle = np.median(angles)
    h, w = image.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def preprocess_receipt(img):
    """Preprocess a receipt image given as a numpy array (BGR)."""

    # 1. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3. Deskew
    deskewed = deskew(enhanced)

    # 4. Light smoothing (remove micro-noise)
    cleaned = cv2.GaussianBlur(deskewed, (3, 3), 0)

    # 5. Resize to width 1024 (keep aspect ratio)
    target_width = 1024
    h, w = cleaned.shape
    scale = target_width / float(w)
    new_h = int(h * scale)
    resized = cv2.resize(cleaned, (target_width, new_h))

    # 6. Convert to 3-channel BGR (EasyOCR expects color)
    processed = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

    return img, processed
