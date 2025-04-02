import cv2
import numpy as np
import torch
import matplotlib
from depth_anything_v2.dpt import DepthAnythingV2

def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Resize the frame to the input size
        input_size = 518
        raw_image = cv2.resize(frame, (input_size, input_size))

        # Infer depth
        depth = depth_anything.infer_image(raw_image, input_size)

        # Normalize depth to 0-255
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)

        # Apply colormap
        depth_colored = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)

        # Display the depth output
        cv2.imshow('Depth Output - VITS', depth_colored)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()