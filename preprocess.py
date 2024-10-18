import cv2
import torch

def mask_pre_process(frame, threshold: int = 3, downsampling_factor: int = 8):
    """
    Get a binary mask from a frame, ignoring black pixels (background). 

    Args:
        frame (np.array): A frame from a video. In format (H, W, C).
        threshold (int): The threshold value for the binary thresholding.
        downsampling_factor (int): The factor by which to downsample the mask. Downsampling is done to get a coarse mask.

    Returns:
        np.array: A binary mask of the frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    downsampled_mask = cv2.resize(binary_mask, (binary_mask.shape[1]//downsampling_factor, binary_mask.shape[0]//downsampling_factor), interpolation=cv2.INTER_AREA)
    upsampled_mask = cv2.resize(downsampled_mask, (binary_mask.shape[1], binary_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    upsampled_mask[upsampled_mask != 0] = 255
    height, width = upsampled_mask.shape
    return upsampled_mask

def crop_video(video, top_percent=20, bottom_percent=20):
    """
    Crop the video to remove the top and bottom borders. Useful when most of the objects are in the middle of the frame.

    Args:
        video (torch.Tensor): A video in format (N, C, H, W).
        top_percent (int): The percentage of the top border to remove.
        bottom_percent (int): The percentage of the bottom border to remove.

    Returns:
        torch.Tensor: A cropped video in format (N, C, H, W).
    """
    _, _, height, _ = video[0].shape
    top_crop = int(height * (top_percent / 100))
    bottom_crop = int(height * (bottom_percent / 100))
    return video[:, :, :, top_crop:height-bottom_crop, :]
