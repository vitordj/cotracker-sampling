# Raw Video Cut/Resize and Frame Sampling Guide

## Introduction

This guide provides instructions for two main tasks in video processing and analysis:

1. Video Extraction: Cutting and resizing video files based on QR code timings. These QR codes were used to identify the start and end of the segments to be processed and this repo assumes that these timings are already known.
2. CoTracker Sampling: Analyzing video content using the CoTracker model to identify key frames.

These tools are designed to help researchers and analysts process large amounts of video data efficiently, extracting relevant segments and identifying frames of interest for further analysis.

## Video Extraction Guide

### Data Preparation

To use the `extract_videos.py` script, you need to prepare your data in the following format:

1. Create an Excel spreadsheet containing the QR code timing data. The spreadsheet should have the following columns:
   - `arquivo_video`: The name of the video file.
   - `t_inicial`: The start time of the segment in the format `mm:ss`.
   - `t_final`: The end time of the segment in the format `mm:ss`.
   - `qr_inicial`: The initial QR code number (integer).
   - `qr_final`: The final QR code number (integer).

   Your Excel spreadsheet should look like this:

   |     | fila   |   volta |   gopro | arquivo_video   |   qr_inicial |   qr_final | t_inicial   | t_final   | comentario                                                                                                   |
   |----:|:-------|--------:|--------:|:----------------|-------------:|-----------:|:------------|:----------|:-------------------------------------------------------------------------------------------------------------|
   |   0 | F2     |       0 |       2 | GX010145.mp4    |           32 |         31 | 00:13:44    | 00:14:13  | Se detuvo el tractor | Avanzó más lento                                                                      |
   |   1 | F2     |       0 |       2 | GX010145.mp4    |           31 |         30 | 00:14:15    | 00:14:29  | nan                                                                                                          |
   |   2 | F2     |       0 |       2 | GX010145.mp4    |           30 |         29 | 00:14:31    | 00:14:45  | nan                                                                                                          |
   |   3 | F2     |       0 |       2 | GX010145.mp4    |           29 |         28 | 00:14:46    | 00:15:00  | nan                                                                                                          |
   |   4 | F2     |       0 |       2 | GX010145.mp4    |           28 |         27 | 00:15:02    | 00:15:16  | nan                                                                                                          |

2. Create a JSON file containing the paths to your Excel files and the sheets you want to process. For example:
   ```json
   {
       "path/to/your/file.xlsx": ["Sheet1", "Sheet2"]
   }
   ```

3. Ensure that the video files mentioned in the Excel spreadsheet are available in the specified input folder.

### Running the Script

1. Install the required libraries:
   ```sh
   pip install pandas ffmpeg-python
   ```

2. Ensure `ffmpeg` is installed and available in the PATH. You can verify this by running:
   ```sh
   ffmpeg -version
   ```

3. Execute the script with the required arguments:
   ```sh
   python extract_videos.py --file_sheets_json path/to/your/file_sheets.json --input_folder path/to/input/folder --output_folder path/to/output/folder --resize_width 1080 --resize_height 1920
   ```

   - `--file_sheets_json`: Path to the JSON file containing the Excel file paths and sheet names.
   - `--input_folder`: Path to the folder containing the input video files.
   - `--output_folder`: Path to the folder where the processed video files will be saved.
   - `--resize_width`: Width to resize the video (default: 1080).
   - `--resize_height`: Height to resize the video (default: 1920).

The script will process the video files based on the QR code timings provided in the Excel spreadsheet, cut and resize the segments, and save them in the specified output folder.

## CoTracker Sampler Guide

### Requirements

- Python 3.7 or higher
- PyTorch
- OpenCV (cv2)
- NumPy
- CoTracker (and its dependencies)

### Installation

1. Install the required libraries:
    ```sh
    pip install torch opencv-python numpy
    ```

2. Install CoTracker following the instructions in its repository.

3. Download the CoTracker checkpoint file and place it in the `./checkpoints/` directory.

### Usage

1. Prepare your input video file and place it in your working directory.

2. Run the script with the required arguments:
    ```sh
    python sampler.py --workdir /path/to/working/directory --input_file input_video.mp4
    ```

3. Additional optional arguments:
    - `--output_folder`: Specify a custom output folder name
    - `--base_grid_size`: Set the base grid size (default: 4)
    - `--points_threshold`: Set the percentage threshold for collecting new frames (default: 5)
    - `--frames_per_inf`: Number of frames per inference (default: 150)
    - `--crop_top`: Percentage to crop from the top of the video (default: 10)
    - `--crop_bottom`: Percentage to crop from the bottom of the video (default: 10)
    - `--save_video`: Flag to save processed videos
    - `--save_frames`: Save selected frames (1) or not (0) (default: 1)

### Output

The script will generate:
1. A text file with the selected frame numbers.
2. (Optional) Saved frames as PNG images.
3. (Optional) A video visualization of the CoTracker results.

### Example

```sh
python sampler.py --workdir /path/to/project --input_file video.mp4 --base_grid_size 6 --points_threshold 10 --save_video
```

This command will process `video.mp4` in the specified working directory, using a base grid size of 6 and a points threshold of 10%. It will also save the visualization video.
