import argparse
import os
import torch
import cv2
import numpy as np
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor
import datetime as dt

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

def main(args):
    workdir = args.workdir
    input_file_path = os.path.join(workdir, args.input_file)
    input_file_name = os.path.splitext(os.path.basename(args.input_file))[0]

    if not args.output_folder:
        date = dt.datetime.now().strftime("%Y%m_%d%H")
        output_folder = os.path.join(workdir, f'{input_file_name}__{date}')
    else:
        output_folder = os.path.join(workdir, args.output_folder)
    os.makedirs(output_folder, exist_ok=True)

    original_numpy = read_video_from_path(input_file_path)
    original_video = torch.from_numpy(original_numpy).permute(0, 3, 1, 2)[None].float()
    original_video = crop_video(original_video, top_percent=args.crop_top, bottom_percent=args.crop_bottom)

    len_video = original_video.shape[1]
    num_digitos = len(str(len_video))

    print(f'O vídeo contém {len_video} frames.')

    next_frame = 0
    frame_list = []
    videos_list = []

    model = CoTrackerPredictor(checkpoint=os.path.join('./checkpoints/cotracker2.pth'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    if args.save_video:
        vis = Visualizer(fps=30, show_first_frame=0, save_dir=output_folder)

    while abs(len_video - next_frame) > 5:
        video = original_video[:, next_frame:next_frame+args.frames_per_inf].to(device)
        first_frame = video[0, 0]
        segm_mask = mask_pre_process(first_frame.permute(1, 2, 0).cpu().numpy())
        seg_proportion = (segm_mask > 0).sum() / (segm_mask.shape[0] * segm_mask.shape[1])
        grid_size = args.base_grid_size // (seg_proportion ** (1/3))
        pred_tracks, pred_visibility = model(video, grid_size=int(grid_size), segm_mask=torch.from_numpy(segm_mask)[None, None])
        sums = pred_visibility[0].sum(axis=1)
        tolerance_p = round((args.points_threshold/100) * (pred_tracks.shape[2]))
        tolerance_mask = (sums <= tolerance_p).nonzero()

        try:
            index = tolerance_mask[0].item()
            frame_list.append(str(next_frame))
            if args.save_frames:
                saved_frame = original_numpy[next_frame]
                saved_frame = cv2.cvtColor(saved_frame, cv2.COLOR_RGB2BGR)
                frames_folder = os.path.join(output_folder, f'frames_{input_file_name}')
                os.makedirs(frames_folder, exist_ok=True)
                cv2.imwrite(os.path.join(frames_folder, f"{str(next_frame).zfill(num_digitos)}.png"), saved_frame)
        except IndexError:
            index = video.shape[1]
        next_frame += index

        if args.save_video:
            part_render = vis.visualize(video=video[:, :index], tracks=pred_tracks, visibility=pred_visibility, save_video=False)
            videos_list.append(part_render)

    frames_txt_path = os.path.join(frames_folder, 'frames.txt')
    with open(frames_txt_path, 'w') as f:
        f.write(",".join(frame_list))

    if args.save_video:
        final_video = torch.cat(videos_list, dim=1)
        vis.save_video(video=final_video, filename=os.path.join(output_folder, f"{input_file_name}_cotracker"))

    print(f"Foram processados {len(frame_list)} frames: {', '.join(frame_list)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoTracker Video Processing")
    parser.add_argument('--workdir', type=str, required=True, help='Work directory, where the input file is located')
    parser.add_argument('--input_file', type=str, required=True, help='Input file')
    parser.add_argument('--output_folder', type=str, default=None, help='Output Folder (optional)')
    parser.add_argument('--base_grid_size', type=int, default=4, help='Grid size height and width assuming that no mask is present.')
    parser.add_argument('--points_threshold', type=int, default=5, help='Percentage. When points limits are reached, a new frame is collected.')
    parser.add_argument('--frames_per_inf', type=int, default=150, help='Número de frames por inferência')
    parser.add_argument('--crop_top', type=int, default=10, help='Porcentagem para crop no topo')
    parser.add_argument('--crop_bottom', type=int, default=10, help='Porcentagem para crop na parte inferior')
    parser.add_argument('--save_video', action='store_true', help='Salvar vídeos processados')
    parser.add_argument('--save_frames', type=int, default=1, help='Save the selected frames (1 or 0)')
    
    args = parser.parse_args()
    main(args)
