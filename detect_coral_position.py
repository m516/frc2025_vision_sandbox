import cv2
import numpy as np
import apriltag
import os

# Initialize AprilTag detector
options = apriltag.DetectorOptions(families="tag36h11")
detector = apriltag.Detector(options)

# Path to image sequence
sequence_path = "renders/"
image_files = sorted([f for f in os.listdir(sequence_path) if f.endswith('.jpg')])
image_index = 0
point_coords = [0, 0, 0]  # Initial point in AprilTag space

# Define 3D points of the cube (global)
cube_points = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
    [0, 0, -1], [1, 0, -1], [1, 1, -1], [0, 1, -1]  # Top face
], dtype=np.float32)

# Camera intrinsic matrix (adjust based on your setup)
K = np.array([[736, 0, 640],
              [0, 736, 400],
              [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros(4, dtype=np.float32)  # Assuming no distortion


def get_projected_point(corners, relative_xyz, selection="left"):
    """
    Given relative XYZ coordinates, returns the best projected point in image space.
    
    Args:
    - corners (np.array): 2D image coordinates of the AprilTag.
    - relative_xyz (list): XYZ coordinates relative to the tag's frame.
    - selection (str): "left" to select the leftmost solution, "right" for the rightmost solution.

    Returns:
    - best_imgpt (tuple): The chosen projected point in image coordinates.
    - all_imgpts (list): The projected points for all solutions.
    """
    success, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        cube_points[:4], corners, K, dist_coeffs, flags=cv2.SOLVEPNP_IPPE
    )
    
    if not success or len(rvecs) == 0:
        print("Error: solvePnPGeneric failed.")
        return None, []

    best_imgpt = None
    all_imgpts = []
    extreme_x = float("inf") if selection == "left" else float("-inf")

    for rvec, tvec in zip(rvecs, tvecs):
        imgpt, _ = cv2.projectPoints(
            np.array([relative_xyz], dtype=np.float32), rvec, tvec, K, dist_coeffs
        )
        imgpt = tuple(map(int, imgpt.ravel()))
        all_imgpts.append(imgpt)

        if (selection == "left" and imgpt[0] < extreme_x) or (selection == "right" and imgpt[0] > extreme_x):
            extreme_x = imgpt[0]
            best_imgpt = imgpt

    return best_imgpt, all_imgpts


def draw_cubes(image, corners, tag_size=1.0):
    """Draw cubes for all pose solutions."""
    scaled_cube_points = cube_points * tag_size

    success, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        cube_points[:4], corners, K, dist_coeffs, flags=cv2.SOLVEPNP_IPPE
    )

    if not success or len(rvecs) == 0:
        print("Error: solvePnPGeneric failed.")
        return image

    for rvec, tvec in zip(rvecs, tvecs):
        imgpts, _ = cv2.projectPoints(scaled_cube_points, rvec, tvec, K, dist_coeffs)
        imgpts = np.int32(imgpts).reshape(-1, 2)

        for i, j in zip([0, 1, 2, 3], [1, 2, 3, 0]):  # Bottom face
            cv2.line(image, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 2)
        for i, j in zip(range(4), range(4, 8)):  # Vertical edges
            cv2.line(image, tuple(imgpts[i]), tuple(imgpts[j]), (0, 255, 0), 2)
        for i, j in zip([4, 5, 6, 7], [5, 6, 7, 4]):  # Top face
            cv2.line(image, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 2)

    return image


def update_image(selection="left"):
    """Load image, detect tag, and update the UI."""
    global image_index, point_coords

    image = cv2.imread(os.path.join(sequence_path, image_files[image_index]))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)
    print(detections)

    for largest_tag in detections:
        corners = np.array(largest_tag.corners, dtype=np.float32)
        print(corners)

        if corners.shape[0] >= 4:
            image = draw_cubes(image, corners)

            # Get projected red dot position
            best_imgpt, all_imgpts = get_projected_point(corners, point_coords, selection=selection)

            # Draw blue dots for all solutions
            for imgpt in all_imgpts:
                cv2.circle(image, imgpt, 5, (255, 0, 0), -1)  # Blue dots

            # Draw 50final selected red dot
            if best_imgpt:
                cv2.circle(image, best_imgpt, 5, (0, 0, 255), -1)  # Red dot

    print (best_imgpt)
    return image


def on_slider_change(val):
    global image_index
    image_index = val
    cv2.imshow('Image Viewer', update_image())


def on_point_change(axis, val):
    global point_coords
    point_coords[axis] = (val - 50) / 5.0  # Map slider to [-10, 10] range
    cv2.imshow('Image Viewer', update_image())


# Create UI
cv2.namedWindow('Image Viewer')
cv2.createTrackbar('Image', 'Image Viewer', 0, len(image_files) - 1, on_slider_change)
cv2.createTrackbar('X', 'Image Viewer', 48, 100, lambda val: on_point_change(0, val))
cv2.createTrackbar('Y', 'Image Viewer', 41, 100, lambda val: on_point_change(1, val))
cv2.createTrackbar('Z', 'Image Viewer', 57, 100, lambda val: on_point_change(2, val))

# Initial display
cv2.imshow('Image Viewer', update_image())
cv2.waitKey(0)
cv2.destroyAllWindows()
