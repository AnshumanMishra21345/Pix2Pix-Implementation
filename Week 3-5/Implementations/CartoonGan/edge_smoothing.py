import cv2
from PIL import Image

def edge_smoothing(pil_img, low_threshold=100, high_threshold=200):
    """
    Given a PIL image, apply Canny edge detection, dilate, and then Gaussian blur
    to simulate the smoothed edge effect.
    """
    img = np.array(pil_img.convert('L'))  # convert to grayscale
    edges = cv2.Canny(img, low_threshold, high_threshold)
    # Dilate edges
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    # Blur the dilated edges
    edges_blurred = cv2.GaussianBlur(edges_dilated, (7, 7), 0)
    # Convert single channel back to 3 channels
    smoothed = cv2.cvtColor(edges_blurred, cv2.COLOR_GRAY2RGB)
    smoothed = Image.fromarray(smoothed)
    return smoothed