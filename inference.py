import argparse
import torch
import yaml
import time
import pathlib
import os
import glob
from tqdm import tqdm
import soundfile as sf
import numpy as np
import sys

from ml_collections import ConfigDict
from utils import get_model_from_config, demix_track, load_start_checkpoint
from torch import nn
from pydub import AudioSegment
from typing import Union, Dict
import subprocess

def run_folder(model, args, config, device, verbose = False):
    start = time.time()
    model.eval()
    mp3_paths = [os.path.join(os.getcwd(), args.input_folder, path) for path in os.listdir(args.input_folder) if pathlib.Path(path).suffix == '.mp3']

    for path in mp3_paths:
        src = path
        dst = os.path.join(os.getcwd(), args.input_folder, f'{os.path.basename(path).split('.')[0]}.wav')
        subprocess.call(['ffmpeg', '-i', src, dst])

    all_mixture_paths = glob.glob(args.input_folder + '/*.wav')
    print(f'Total track found: {len(all_mixture_paths)}')

    instruments = config.training.instruments
    if config.training.target_instrument is not None:
        instruments = [config.training.target_instrument]
    
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    if not verbose:
        all_mixture_paths = tqdm(all_mixture_paths)

    first_chunk_time = None

    for track_number, path in enumerate(all_mixture_paths, 1):
        print(f"\nProcessing track {track_number}/{len(all_mixture_paths)} : {os.path.basename(path)}")
        mix, sr = sf.read(path)

        original_mono = False
        if len(mix.shape) == 1:
            original_mono = True
            mix = np.stack([mix, mix], axis = -1)

        mixture = torch.tensor(mix.T, dtype = torch.float32)
        if first_chunk_time is not None:
            total_length = mixture.shape[1]
            num_chunks = (total_length + config.inference.chunk_size // config.inference.num_overlap - 1) // (config.inference.chunk_size // config.inference.num_overlap)
            estimated_total_time = first_chunk_time * num_chunks
            print(f"Estimated total processing time for this track: {estimated_total_time:.2f} seconds")
            sys.stdout.write(f"Estimated time remaining: {estimated_total_time:.2f} seconds\r")

            sys.stdout.flush()

        res, first_chunk_time = demix_track(config, model, mixture, device, first_chunk_time)

        for instrument in instruments:
            vocal_output = res[instrument].T
            if original_mono:
                vocal_output = vocal_output[:, 0]

            vocal_path = f"{args.output_dir}/{os.path.basename(path)[:-4]}_{instrument}.wav"
            sf.write(vocal_path, vocal_output, sr, subtype = 'FLOAT')

        vocal_output = res[instruments[0]].T
        if original_mono:
            vocal_output = vocal_output[:, 0]
        
        original_mix, _ = sf.read(path)
        instrumental = original_mix - vocal_output

        instrumental_path = f"{args.output_dir}/{os.path.basename(path)[:-4]}_instrumental.wav"
        sf.write(instrumental_path, instrumental, sr, subtype = 'FLOAT')

    time.sleep(1)
    print(f"Elapsed time: {(time.time() - start):.2f} sec")

def parse_args_inference(dict_args: Union[Dict, None]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type = str, help = "Path to config yaml file")
    parser.add_argument("--model_path", type = str, default = '', help = "Location of the model")
    parser.add_argument("--input_folder", type = str, help = "Folder with song to process")
    parser.add_argument("--output_dir", type = str, help = "Path to store model outputs")
    parser.add_argument("--device_ids", nargs = "+", type = int, default = 0, help = "List of GPU ids")

    if dict_args is not None:
        args = parser.parse_args([])
        args_dict = vars(args)
        args_dict.update(dict_args)
        args = argparse.Namespace(**args_dict)
    else:
        args = parser.parse_args()

    return args

def proc_folder(dict_args):
    args = parse_args_inference(dict_args)
    
    model_load_start_time = time.time()

    torch.backends.cudnn.benchmark = True
    
    config = ConfigDict(yaml.load(open(args.config_path), Loader = yaml.FullLoader))
    model = get_model_from_config(config)
    
    if args.model_path != '':
        print(f"Using model: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
        load_start_checkpoint(args, model, checkpoint, type_='inference')

    if torch.cuda.is_available():
        device_ids = args.device_ids
        if type(device_ids) == int:
            print('CUDA is available')
            device = torch.device(f"cuda:{device_ids}")
            model = model.to(device)
        else:
            device = torch.device(f"cuda:{device_ids[0]}")
            model = nn.DataParallel(model, device_ids = device_ids).to(device)
    else:
        device = 'cpu'
        print('CUDA is not available. Run inteference on CPU. It will be very slow...')
        model = model.to(device)
    
    print(f'Model load time: {(time.time() - model_load_start_time):.2f} sec')

    run_folder(model, args, config, device, verbose = False)

if __name__ == '__main__':
    proc_folder(None)