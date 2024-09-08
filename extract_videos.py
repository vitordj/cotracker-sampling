import os
import pandas as pd
import ffmpeg
from datetime import datetime
import argparse
import json

def cut_resize_mp4(df: pd.DataFrame, input_folder: str, output_folder: str | None = None, resize_to: tuple[int, int] | None = (1080, 1920)) -> None:
    """
    Cut a video file based on the start and end times of a QR code.

    Args:
        df (pd.DataFrame): A dataframe with the video file and QR code information.
        input_folder (str): The folder containing the video files.
        output_folder (str | None): The folder to save the processed video files. If None, use current directory.
        resize_to (tuple[int, int] | None): The size to resize the video to. Defaults to (1080, 1920).
    """
    # Get current date and time
    data_hora = datetime.now().strftime("%Y%m%d_%H%M")
    
    if output_folder is None:
        output_folder = f"video_cut_{data_hora}"
    else:
        output_folder = os.path.join(output_folder, f"video_cut_{data_hora}")
    os.makedirs(output_folder, exist_ok=True)

    for _, row in df.iterrows():
        if not os.path.exists(os.path.join(input_folder, row['arquivo_video'])):
            raise FileNotFoundError(f"File {row['arquivo_video']} not found in {input_folder}")

    for _, row in df.iterrows():
        row['qr_inicial'] = int(row['qr_inicial']) if not pd.isna(row['qr_inicial']) else ''
        row['qr_final'] = int(row['qr_final']) if not pd.isna(row['qr_final']) else ''
        output_file = f"{row['arquivo_video'].split('.')[0]}__{row['qr_inicial']}-{row['qr_final']}.mp4"
        if output_folder:
            output_file_path = os.path.join(output_folder, output_file).replace('\\', '/')
        else:
            output_file_path = output_file

        if input_folder:
            input_file_path = os.path.join(input_folder, row['arquivo_video']).replace('\\', '/')
        else:
            input_file_path = row['arquivo_video']
        
        print(f"Processing {output_file_path}")
        print(input_file_path)
        
        stream = ffmpeg.input(input_file_path, ss=row['t_inicial'], to=row['t_final'])
        stream = ffmpeg.output(stream, output_file_path, c='copy', y=None)  # Add 'y=None' to force overwrite
        ffmpeg.run(stream)
        
        if resize_to:
            temp_output_file_path = output_file_path.replace('.mp4', '_resized.mp4')
            stream = ffmpeg.input(output_file_path)
            stream = ffmpeg.output(stream, temp_output_file_path, vf=f'scale={resize_to[0]}:{resize_to[1]}', y=None)  # Add 'y=None' to force overwrite
            ffmpeg.run(stream)
            os.replace(temp_output_file_path, output_file_path)  # Replace the original file with the resized one
    return output_folder

def process_excel_files(file_sheets: dict[str, list[str]]) -> pd.DataFrame:
    """
    Process all excel files and sheets, and return a concatenated dataframe.
    The dataframe is expected to have the following columns:
    - arquivo_video: str
    - t_inicial: str
    - t_final: str
    - qr_inicial: int
    - qr_final: int

    Args:
        file_sheets (dict[str, list[str]]): A dictionary with the excel files and sheets to process.

    Returns:
        pd.DataFrame: A concatenated dataframe with the processed data.
    """
    data_frames = []
    for file, sheets in file_sheets.items():
        for sheet in sheets:
            try:
                df = pd.read_excel(file, sheet_name=sheet)
                data_frames.append(df)
            except Exception as e:
                print(f"Error processing file '{file}', sheet '{sheet}': {e}")
    
    if data_frames:
        df = pd.concat(data_frames, ignore_index=True)
        df['t_inicial'] = '00:' + df['t_inicial'].str.strip().astype(str).str[:5]
        df['t_final'] = '00:' + df['t_final'].str.strip().astype(str).str[:5]
        df['qr_inicial'] = df['qr_inicial'].astype(int, errors='raise')
        df['qr_final'] = df['qr_final'].astype(int, errors='raise')
        
        assert len(df[(df.t_inicial > df.t_final)]) == 0  # there should not be any case where t_inicial is greater than t_final
        assert len(df[(df.qr_inicial.isna()) | (df.qr_final.isna())]) == 0 # there should not be any case where qr_inicial or qr_final is null

        return df
    else:
        raise ValueError("No data frames to concatenate.")

def main(args):
    with open(args.file_sheets_json, 'r') as f:
        file_sheets = json.load(f)
    json_path = os.path.dirname(args.file_sheets_json)
    file_sheets = {os.path.join(json_path, file): sheets for file, sheets in file_sheets.items()}
    print(file_sheets)
    df = process_excel_files(file_sheets)

    if args.video_file:
        df = df.query(f"arquivo_video == '{args.video_file}'")
    
    new_folder = cut_resize_mp4(
        df.iloc[:1],
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        resize_to=(args.resize_width, args.resize_height)
    )
    print(f'\n{new_folder}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Video Extraction and Processing")
    parser.add_argument('--file_sheets_json', type=str, required=True, help='Path to JSON file containing file sheets information')
    parser.add_argument('--video_file', type=str, default=None, help='Name of the video file to process')
    parser.add_argument('--input_folder', type=str, default=None, help='Input folder containing video files')
    parser.add_argument('--output_folder', type=str, default=None, help='Output folder for processed videos')
    parser.add_argument('--resize_width', type=int, default=1080, help='Width to resize the video')
    parser.add_argument('--resize_height', type=int, default=1920, help='Height to resize the video')
    
    args = parser.parse_args()
    main(args)