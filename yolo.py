import cv2
import numpy as np
import torch
import os
import matplotlib
from depth_anything_v2.dpt import DepthAnythingV2
import math, time
from ultralytics import YOLO



model = YOLO('best2.pt')


# Relative path to the input video
video_path = 'assets/examples_video/fira.mp4'
output_dir = './vis_video_depth'
output_file = os.path.join(output_dir, 'fira_depth.mp4')

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Model configuration for 'vits'
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
}

# Load the model
depth_anything = DepthAnythingV2(**model_configs['vits'])
depth_anything.load_state_dict(torch.load('checkpoints/depth_anything_v2_vits.pth', map_location='cpu'))
depth_anything = depth_anything.to(DEVICE).eval()

# Colormap for visualization
cmap = matplotlib.colormaps.get_cmap('Spectral_r')

# Open the video file
raw_video = cv2.VideoCapture(video_path)
if not raw_video.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# Get video properties
frame_width = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))

# Output video writer
out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))

print(f"Processing video: {video_path}")
print(f"Output will be saved to: {output_file}")


def get_nearest_bbox(frame, bboxes, depth_model, input_size=518, depth_threshold=0.1, width_threshold=0.1):
    """
    Finds the indices of bounding boxes within a given depth and width threshold from the nearest bounding box.

    Args:
        frame (numpy.ndarray): The input frame (BGR image).
        bboxes (list): List of bounding boxes in the format [[x1, y1, w1, h1], ...].
        depth_model (DepthAnythingV2): The depth model to infer depth.
        input_size (int): The input size for the depth model.
        depth_threshold (float): The depth threshold to include nearby boxes.
        width_threshold (float): The width threshold to include nearby boxes.

    Returns:
        list: Indices of bounding boxes within the depth and width threshold from the nearest box.
    """

    # Calculate frame width locally
    frame_width = frame.shape[1]

    # Calculate width threshold as 0.1 of the frame width
    width_threshold = width_threshold * frame_width
    
    # Resize the frame to the input size for depth inference
    resized_frame = cv2.resize(frame, (input_size, input_size))

    # Infer depth
    depth = depth_model.infer_image(resized_frame, input_size)

    # Normalize depth values to the range [0, 1]
    depth = (depth - depth.min()) / (depth.max() - depth.min())

    # Resize depth map back to the original frame size
    depth_resized = cv2.resize(depth, (frame.shape[1], frame.shape[0]))

    # Initialize variables to track the nearest bounding box
    max_depth = -float('inf')
    nearest_bbox_index = -1
    bbox_depths = []

    # Iterate through each bounding box
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        x_min = int(x - w / 2)
        y_min = int(y - h / 2)
        x_max = int(x + w / 2)
        y_max = int(y + h / 2)

        # Extract the depth values within the bounding box
        bbox_depth = depth_resized[y_min:y_max, x_min:x_max]

        # Calculate the average depth value within the bounding box
        avg_depth = bbox_depth.max()
        bbox_depths.append(avg_depth)
        print("avg_depth:::::::::::: ", avg_depth)
        # Update the nearest bounding box if the current one is closer
        if avg_depth > max_depth:
            max_depth = avg_depth
            nearest_bbox_index = i
    print("nearest box depth:::::::::::: ", max_depth)
    # Find all bounding boxes within the depth and width threshold
    indices_within_threshold = [
        i for i, avg_depth in enumerate(bbox_depths)
        if abs(avg_depth - bbox_depths[nearest_bbox_index]) <= depth_threshold
        and abs(bboxes[i][2] - bboxes[nearest_bbox_index][2]) <= width_threshold
    ]
    print("count of indices_within_threshold:::::::::::: ", len(indices_within_threshold))

    return indices_within_threshold

def filtering(img, results):
    j = -1


    #PINK techOlympic
    # lowerBound = np.array([87, 22, 0])
    # upperBound = np.array([179, 192, 255])

    #PURPLE home
    # lowerBound = np.array([90, 64, 16])
    # upperBound = np.array([146, 159, 90])

    #RED fira
    lowerBound = np.array([0, 128, 61])
    upperBound = np.array([17, 255, 143])

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    imgMasked = cv2.inRange(hsv,lowerBound, upperBound)

    boxes = results[0].boxes.xywh.tolist()

    keypoints = results[0].keypoints.xy.tolist()

    satisfying_boxes = []
    satisfying_keypoints = []

    closest_satisfying_keypoint = []
    closest_satisfying_box = []

    isGate = False


    for box in boxes:
        j += 1
        x_min = int(box[0]- box[2]/2)
        y_min = int(box[1]- box[3]/2)
        x_max = int(box[0]+ box[2]/2)
        y_max = int(box[1]+ box[3]/2)
        box_area = imgMasked[y_min:y_max, x_min:x_max]
        ones_count = np.sum(box_area == 255)
        total_pixels = box_area.size

        if ones_count >= total_pixels / 6 :
            satisfying_boxes.append(box)
            satisfying_keypoints.append(keypoints[j])
            isGate = True
    
    if isGate:
        # max_index = max(range(len(satisfying_boxes)), key=lambda i: satisfying_boxes[i][3]) # get the index of the satisfying box with the highest Height value (it belongs to the closest gate)
        # max_index = get_nearest_bbox(img, satisfying_boxes, depth_anything) # get the index of the satisfying box with the highest depth value (it belongs to the closest gate)
        
        # closest_satisfying_box = [satisfying_boxes[max_index]]
        # closest_satisfying_keypoint = [satisfying_keypoints[max_index]]

        # Update closest_satisfying_box and closest_satisfying_keypoint
        indices_within_threshold = get_nearest_bbox(img, satisfying_boxes, depth_anything)  # Get all indices within the threshold

        closest_satisfying_box = [satisfying_boxes[i] for i in indices_within_threshold]  # Select all satisfying boxes
        closest_satisfying_keypoint = [satisfying_keypoints[i] for i in indices_within_threshold]  # Select all satisfying keypoints

    return isGate, closest_satisfying_box, closest_satisfying_keypoint


def main():
    while raw_video.isOpened():
        ret, raw_frame = raw_video.read()
        if not ret:
            break

        results = model(raw_frame, verbose=False)
        annotated_frame = results[0].plot()

        isGate, boxes, keypoints = filtering(raw_frame, results)

        if isGate:
            for i in range(len(boxes)):
                annotated_frame = cv2.rectangle(
                    annotated_frame,
                    (int(boxes[i][0] - boxes[i][2] / 2), int(boxes[i][1] - boxes[i][3] / 2)),
                    (int(boxes[i][0] + boxes[i][2] / 2), int(boxes[i][1] + boxes[i][3] / 2)),
                    (0, 255, 0),
                    3,
                )

        # Write the annotated frame to the output video
        out.write(annotated_frame)

    # Release resources
    raw_video.release()
    out.release()
    print("Processing complete.")

# def main():
 
#     while raw_video.isOpened():
#         ret, raw_frame = raw_video.read()
#         if not ret:
#             break

#         results = model(raw_frame, verbose = False)
#         annotated_frame = results[0].plot()

#         isGate, boxes, keypoints = filtering(raw_frame, results)
       
#         if isGate:
#             for i in range(len(boxes)):
#                 annotated_frame = cv2.rectangle(annotated_frame,(int(boxes[i][0]-boxes[i][2]/2) , int(boxes[i][1]-boxes[i][3]/2)), (int(boxes[i][0]+boxes[i][2]/2) , int(boxes[i][1]+boxes[i][3]/2)), (0, 255, 0), 3)
#                 # annotated_frame = cv2.rectangle(annotated_frame,(int(boxes[0][0]-boxes[0][2]/2) , int(boxes[0][1]-boxes[0][3]/2)), (int(boxes[0][0]+boxes[0][2]/2) , int(boxes[0][1]+boxes[0][3]/2)), (0, 255, 0), 3) 
#                 # print("nearest gate found")
#         # # Resize the frame to the input size
#         # input_size = 518
#         # resized_frame = cv2.resize(raw_frame, (input_size, input_size))

#         # # Infer depth
#         # depth = depth_anything.infer_image(resized_frame, input_size)

#         # # Normalize depth to 0-255
#         # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
#         # depth = depth.astype(np.uint8)

#         # # Apply colormap
#         # depth_colored = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)
#         # depth_colored = cv2.resize(depth_colored, (frame_width, frame_height))

#         # # Combine original frame and depth visualization
#         # split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
#         # combined_frame = cv2.hconcat([annotated_frame, split_region, depth_colored])

#         # Write the combined frame to the output video
#         out.write(raw_frame)

#     # Release resources
#     raw_video.release()
#     out.release()
#     print("Processing complete.")

if __name__ == '__main__':
    main()