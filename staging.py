import cv2

def mask_pre_process(frame, threshold: int = 3, downsampling_factor: int = 8, border_percentage: float = 13.0):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    downsampled_mask = cv2.resize(binary_mask, (binary_mask.shape[1]//downsampling_factor, binary_mask.shape[0]//downsampling_factor), interpolation=cv2.INTER_AREA)
    upsampled_mask = cv2.resize(downsampled_mask, (binary_mask.shape[1], binary_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    upsampled_mask[upsampled_mask != 0] = 255
    height, width = upsampled_mask.shape
    border_width = int((border_percentage / 100) * width)
    return upsampled_mask

def crop_video(video, top_percent=20, bottom_percent=20):
    _, _, height, _ = video[0].shape
    top_crop = int(height * (top_percent / 100))
    bottom_crop = int(height * (bottom_percent / 100))
    return video[:, :, :, top_crop:height-bottom_crop, :]