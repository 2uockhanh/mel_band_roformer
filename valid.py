import torch
import time
import argparse
import os
import librosa
import numpy as np
import soundfile as sf
import yaml
import gc

from torch import nn
from ml_collections import ConfigDict
from typing import Optional, List, Tuple, Dict, Union, Any
from pathlib import Path
from tqdm import tqdm
from utils import demix_track, read_audio_transposed, apply_tta, denormalize_audio, draw_2_mel_spectrogram, logging, write_results_in_file, get_model_from_config, load_start_checkpoint
from metrics import get_metrics

def find_mixture_files(root_dir):
    wav_files = os.path.join(root_dir, 'mixture.wav')
    if os.path.exists(wav_files):
        return wav_files

def get_mixture_paths(args: argparse.Namespace, verbose: bool, config: ConfigDict, extension: str = 'wav') -> List[str]:
    ddp_mode = torch.distributed.is_initialized()
    should_print = (not ddp_mode) or (torch.distributed.get_rank() == 0)
    try:
        valid_path = args.valid_path
    except Exception as e:
        if should_print:
            print('No valid path in args')
        raise e
    
    if isinstance(valid_path, str):
        valid_paths: List[str] = [valid_path]
    else:
        valid_paths = list(valid_path)
    
    all_mixtures_path: List[str] = []
    for path in valid_paths:
        child_paths = [child for child in os.listdir(path) if os.path.isdir(os.path.join(path, child))]
        for child in child_paths:
            part = find_mixture_files(os.path.join(path, child))
            #print(os.path.join(valid_path, child))
            if not part and verbose and should_print:
                print(f'No validation data found in: {os.path.join(valid_path, child)}')
            all_mixtures_path.append(part)

    if verbose and should_print:
        inference = config.inference or None
        if inference is None and isinstance(config, dict):
            inference = config.inference or None
        def _get(obj, name, default = None):
            if obj is None:
                return default
            if isinstance(obj, dict):
                return obj.get(name, default)
            return getattr(obj, name, default)
        num_overlap = _get(inference, 'num_overlap', '?')
        print(f'Total mixtures: {len(all_mixtures_path)}')
        print(f'Overlap : {num_overlap}')
    return all_mixtures_path
# Cập nhật kết quả đánh giá và thanh tiến trình
def update_metrics_and_pbar(track_metrics: Dict[str, float], all_metrics: Dict[str, Dict[str, Union[Dict[str, float], List[float]]]], instr: str, 
                            pbar_dict: Dict[str, float], mixture_paths: Optional[Union[List[str], tqdm]], verbose: bool = False, path: Optional[str] = None, ) -> None:
    ddp_mode = torch.distributed.is_initialized()
    should_print = (not ddp_mode) or (torch.distributed.get_rank() == 0)

    if ddp_mode and path is None:
        raise ValueError("`path` must be provided when torch.distributed is initialized.")
    
    for metric_name, metric_value in track_metrics.items():
        if verbose and should_print:
            print(f'Metric {metric_name:11s} value: {metric_value:.4f}')
        if metric_name not in all_metrics:
            all_metrics[metric_name] = {}
        if instr not in all_metrics[metric_name]:
            all_metrics[metric_name][instr] = {} if ddp_mode else []
        if ddp_mode:
            all_metrics[metric_name][instr][path] = metric_value
        else:
            all_metrics[metric_name][instr].append(metric_value)
        pbar_dict[f'{metric_name} _{instr}'] = metric_value

    if mixture_paths is not None and hasattr(mixture_paths, 'set_postfix'):
        try:
            mixture_paths.set_postfix(pbar_dict)
        except Exception:
            pass
# Vạch ra khoảng đầu-cuối trên mỗi world size của tập hợp các đường dẫn file mixture.wav dùng để thực hiện việc đánh giá
def block_bounds(num_tracks: int, world_size: int, rank: int) -> Tuple[int, int]:
    if num_tracks % world_size != 0:
        raise ValueError(f'n ({num_tracks}) must be divisible by world_size ({world_size})')

    chunk = num_tracks // world_size
    start = rank * chunk
    end = start + chunk

    return start, end
# Xử lý các file audio
def process_audio_files(mixture_paths: List[str], model: nn.Module, args: Any, config: ConfigDict, device: torch.device, verbose: bool = False, 
                        is_tqdm: bool = True) -> Dict[str, Dict[str, Union[Dict[str, float], List[float]]]]:
    ddp_mode = torch.distributed.is_initialized()
    should_print = (not ddp_mode) or (torch.distributed.get_rank() == 0)

    instruments = prefer_target_instrument(config)
    use_tta = getattr(args, 'use_tta', False)
    output_dir = getattr(args, 'output_dir', '')
    extension = 'wav'

    if ddp_mode:
        all_metrics: Dict[str, Dict[str, Dict]] = { metric: {instr: {} for instr in config.training.instruments} for metric in args.metrics }
    else:
        all_metrics: Dict[str, Dict[str, List[float]]] = { metric: {instr: [] for instr in config.training.instruments} for metric in args.metrics }
    
    if is_tqdm and should_print:
        mixture_paths = tqdm(mixture_paths)
    # Lấy các file âm thanh thành phần (vocals, other...)
    def get_instruments(path: str) -> dict[str, str]:
        real_instruments: dict[str, str] = {}
        for instr in instruments:
            file_path = Path(path) / f"{instr}.wav"
            if file_path.exists():
                real_instruments[instr] = 'wav'
                break
        return real_instruments

    for path in mixture_paths:
        start = time.time()
        mix, sr = read_audio_transposed(path)
        mix_orig = mix.copy()
        folder = os.path.dirname(path)
        real_instruments = get_instruments(folder)

        if 'audio' in config and 'sample_rate' in config.audio:
            target_sr = config.audio['sample_rate']
            if sr != target_sr:
                orig_length = mix.shape[-1]
                if verbose and should_print:
                    print(f'Warning: sample rate is different. In config: {target_sr} in file {path}: {sr}')
                # Thay đổi sampling rate của chuổi thời gian audio
                mix = librosa.resample(
                    mix, 
                    orig_sr = sr, # Sampling rate của audio trong tập dữ liệu
                    target_sr = target_sr, # Sampling rate chuyển đổi đến (Theo config.yaml là 44100 Hz)
                    res_type = 'kaiser_best' # Loại resample, với 'kaiser_best' (chế độ high quality) là giá trị mặc định
                )

        if verbose and should_print:
            print(f'Song: {os.path.abspath(folder)} Shape: {mix.shape}')

        norm_params = None
        # Thực hiện việc phân tách hỗn hợp âm thanh, thu được waveform của chuỗi âm thanh thành phần được phân tách
        waveforms_orig, _ = demix_track(config, model, mix.copy(), device)
        # Áp dụng kỹ thuật TTA (Test-Time Augmentation) để chuyển đổi dữ liệu waveform gốc này ở khoảng thời gian suy luận nhằm cải thiện hiệu suất mô hình
        if use_tta:
            waveforms_orig = apply_tta(config, model, mix, waveforms_orig, device)
        pbar_dict = {}

        for instr, extension in real_instruments.items():
            if verbose and should_print:
                print(f'Instr: {instr}')        
            # Đọc dữ liệu hoán vị của audio từ từng thư mục bài hát trong tập dữ liệu
            if instr != 'other' or not getattr(config.training, 'other_fix', False):
                track, sr1 = read_audio_transposed(f'{folder}/{instr}.{extension}', instr, skip_err = True)
                if track is None:
                    continue
            else:
                track, sr1 = read_audio_transposed(f'{folder}/vocals.{extension}')
                track = mix_orig - track
            # Lấy dữ liệu thành phần âm thanh đã được phân tách
            estimates = waveforms_orig[instr]

            if 'audio' in config and 'sample_rate' in config.audio:
                target_sr = config.audio['sample_rate']
                if sr != target_sr:
                    estimates = librosa.resample(estimates, orig_sr = target_sr, target_sr = sr, res_type = 'kaiser_best')
                    # Cắt hoặc đệm chuỗi NumPy của đoạn âm thanh được phân tách sao cho đúng độ dài mong muốn dọc theo một trục nhất định
                    estimates = librosa.util.fix_length(estimates, size = orig_length)

            if norm_params is not None:
                estimates = denormalize_audio(estimates, norm_params)
            # Lưu các file dữ liệu đã được phân tách vào trong thư mục đầu ra (nếu có)
            if output_dir:
                os.makedirs(output_dir, exist_ok = True)
                base = f'{output_dir}/{os.path.basename(folder)}_{instr}'
                peak = float(np.abs(estimates).max())
                if peak > 1.0:
                    out_path = f'{base}.wav'
                    sf.write(out_path, estimates.T, sr, subtype = 'FLOAT')
                draw_spec = getattr(args, 'draw_spectro', 0)

                if draw_spec and draw_spec > 0:
                    draw_2_mel_spectrogram(estimates.T, track.T, sr, draw_spec, base)
            k = 10 # k_sdr

            track_metrics = get_metrics(args.metrics, track, estimates, mix_orig, device = device, k = k)

            if ddp_mode:
                update_metrics_and_pbar(
                    track_metrics, 
                    all_metrics, 
                    instr, 
                    pbar_dict, 
                    mixture_paths = mixture_paths, 
                    verbose = verbose and should_print, 
                    path = path
                )
            else:
                update_metrics_and_pbar(
                    track_metrics, 
                    all_metrics, 
                    instr, 
                    pbar_dict, 
                    mixture_paths = mixture_paths, 
                    verbose = verbose and should_print
                )

        if verbose and should_print:
            print(f'Time for song: {time.time() - start:.2f} sec')

    return all_metrics

def compute_metric_avg(output_dir: str, args, instruments: List[str], config: ConfigDict, all_metrics: Dict[str, Dict[str, Union[List[float], Dict[str, float]]]], 
        start_time: float) -> Dict[str, float]:
    ddp_mode = torch.distributed.is_initialized()
    should_print = (not ddp_mode) or (torch.distributed.get_rank() == 0)

    logs: List[str] = []

    verbose_logging = bool(output_dir) and should_print
    if verbose_logging:
        logs.append(str(args))
    logs = logging(logs, text = f'Num overlap: {config.inference.num_overlap}', verbose_logging = verbose_logging)

    metric_sum: Dict[str, float] = {}

    for instr in instruments:
        for metric_name in all_metrics:
            per_instr_container = all_metrics[metric_name]
            values_obj = per_instr_container.get(instr, []) if isinstance(per_instr_container, dict) else []

            if isinstance(values_obj, dict):
                vals = list(values_obj.values())
            else:
                vals = list(values_obj)

            arr = np.asarray(vals, dtype = float)

            if arr.size == 0:
                mean_val = float('nan')
                std_val = float('nan')
            else:
                mean_val = float(arr.mean())
                std_val = float(arr.std())

            logs = logging(logs,text = f'Instr {instr} {metric_name}: {mean_val:.4f} (Std: {std_val:.4f})',verbose_logging = verbose_logging)

            metric_sum[metric_name] = metric_sum.get(metric_name, 0.0) + mean_val

    metric_avg: Dict[str, float] = {}
    denom = max(len(instruments), 1)

    for metric_name in all_metrics:
        metric_avg[metric_name] = metric_sum.get(metric_name, float('nan')) / denom
    if len(instruments) > 1:
        for metric_name, avg in metric_avg.items():
            logs = logging(logs, text = f'Metric avg {metric_name:11s}: {avg:.4f}', verbose_logging = verbose_logging)

    logs = logging(logs, text = f'Elapsed time: {time.time() - start_time:.2f} sec', verbose_logging = verbose_logging)
    if output_dir:
        write_results_in_file(output_dir, logs)
    return metric_avg
# Thực hiện đánh giá trong một bộ xử lý con trong nhiều bộ xử lý GPU hoặc thiết bị
def validate_in_subprocess(proc_id: int, queue: torch.multiprocessing.Queue, all_mixtures_path: List[str], model: nn.Module, args, config: ConfigDict, 
        device: str, return_dict) -> None:
    m1 = model.eval().to(device)
    if proc_id == 0:
        progress_bar = tqdm(total = len(all_mixtures_path))
    all_metrics = {
        metric: {instr: [] for instr in config.training.instruments} for metric in args.metrics
    }
    while True:
        current_step, path = queue.get()
        if path is None:
            break
        single_metrics = process_audio_files([path], m1, args, config, device, False, False)
        pbar_dict = {}
        for instr in config.training.instruments:
            for metric_name in all_metrics:
                all_metrics[metric_name][instr] += single_metrics[metric_name][instr]
                if len(single_metrics[metric_name][instr]) > 0:
                    pbar_dict[f'{metric_name}_{instr}'] = f'{single_metrics[metric_name][instr][0]:.4f}'
        if proc_id == 0:
            progress_bar.update(current_step - progress_bar.n)
            progress_bar.set_postfix(pbar_dict)
    return_dict[proc_id] = all_metrics
    return

def prefer_target_instrument(config: ConfigDict) -> List[str]:
    if getattr(config.training, 'target_instrument', None):
        return [config.training.target_instrument]
    else:
        return config.training.instruments

def run_parallel_validation(verbose: bool, all_mixtures_path: List[str], config: ConfigDict, model: nn.Module, device_ids: List[int], args, return_dict) -> None:
    model = model.to('cpu')
    try:
        model = model.module
    except:
        pass
    queue = torch.multiprocessing.Queue()
    processes = []
    for i, device in enumerate(device_ids):
        if torch.cuda.is_available():
            device = f'cuda:{device}'
        else:
            device = 'cpu'
        p = torch.multiprocessing.Process(
            target = validate_in_subprocess, 
            args = (i, queue, all_mixtures_path, model, args, config, device, return_dict)
        )
        p.start()
        processes.append(p)
    for i, path in enumerate(all_mixtures_path):
        queue.put((i, path))
    for _ in range((len(device_ids))):
        queue.put((None, None))
    for p in processes:
        p.join()
    return

def valid_multi_gpu(model: nn.Module, args, config: ConfigDict, device_ids: Optional[List[int]] = None, verbose: bool = False) -> Tuple[Dict[str, float], Dict]:
    start = time.time()
    # Lấy các thông số trong mục inference của config.yaml
    inference = getattr(config, 'inference', None)
    if inference is None and isinstance(config, dict):
        inference = config.get('inference', {})
    # Lấy tất cả các file mixture.wav từ tập dữ liệu
    extension = 'wav'
    all_mixture_path = get_mixture_paths(args, verbose, config, extension)

    ddp_mode = torch.distributed.is_initialized()
    if ddp_mode:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        device = torch.device(f'cuda:{rank}')

        model.to(device)
        model.eval()
        # Số lượng file mixture.wav thu thập được
        num_tracks = len(all_mixture_path)
        
        pad_needed = (-num_tracks) % world_size

        if pad_needed and num_tracks > 0:
            all_mixture_path += all_mixture_path[:pad_needed]

        padded_num_tracks = len(all_mixture_path)
        target_len = padded_num_tracks // world_size
        start, end = block_bounds(padded_num_tracks, world_size, rank)
        per_rank_data = all_mixture_path[start:end]

        local_metrics = { metric: {instr: [] for instr in config.training.instruments} for metric in args.metrics }
        # Xử lý các file audio
        with torch.no_grad():
            single_metrics = process_audio_files(per_rank_data, model, args, config, device, verbose = verbose)
            for instr in config.training.instruments:
                for metric_name in args.metrics:
                    local_metrics[metric_name][instr] = single_metrics[metric_name][instr]
        all_metrics: Dict[str, Dict[str, List[float]]] = {m: {} for m in args.metrics}

        for metric in args.metrics:
            for instr in config.training.instruments:
                all_metrics[metric][instr] = []
                per_instr = local_metrics[metric][instr]

                if isinstance(per_instr, dict):
                    local_data = list(per_instr.values())
                else:
                    local_data = list(per_instr)
                
                if len(local_data) == 0:
                    local_tensor = torch.zeros(target_len, dtype = torch.float32, device = device)
                else:
                    if len(local_data) < target_len:
                        local_data = local_data + [0.0] * (target_len - len(local_data))
                    local_tensor = torch.tensor(local_data, dtype = torch.float32, device = device)
                gathered_list = [torch.zeros_like(local_tensor) for _ in range(world_size)]
                torch.distributed.all_gather(gathered_list, local_tensor)
                cat_vals = torch.cat(gathered_list).tolist()[:num_tracks]
                all_metrics[metric][instr] = cat_vals

        if torch.distributed.get_rank() == 0:
            instruments = prefer_target_instrument(config)
            metric_avg = compute_metric_avg(getattr(args, 'output_dir', ""), args, instruments, config, all_metrics, start)
            return metric_avg, all_metrics
        return None, None
    output_dir = getattr(args, 'ouput_dir', "")
    return_dict = torch.multiprocessing.Manager().dict()

    run_parallel_validation(verbose, all_mixture_path, config, model, device_ids, args, return_dict)

    all_metrics: Dict[str, Dict[str, List[float]]] = {m: {} for m in args.metrics}
    for metric in args.metrics:
        for instr in config.training.instruments:
            merged: List[float] = []
            for i in range(len(device_ids)):
                merged += return_dict[i][metric][instr]
            all_metrics[metric][instr] = merged

    instruments = prefer_target_instrument(config)
    metric_avg = compute_metric_avg(output_dir, args, instruments, config, all_metrics, start)

    return metric_avg, all_metrics

def valid(model: nn.Module, args, config: ConfigDict, device: torch.device, verbose: bool = False) -> Tuple[dict, dict]:
    start = time.time()
    model.eval().to(device)
    output_dir = getattr(args, 'output_dir', '')
    extension = 'wav'
    all_mixtures_path = get_mixture_paths(args, verbose, config, extension)
    all_metrics = process_audio_files(all_mixtures_path, model, args, config, device, verbose, not verbose)
    instruments = prefer_target_instrument(config)
    return compute_metric_avg(output_dir, args, instruments, config, all_metrics, start), all_metrics

def parse_args_valid(dict_args: Union[Dict, None]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type = str, help = 'Path to config file')
    parser.add_argument('--model_path', type = str, default = '', help = 'Initial checkpoint to valid weights')
    parser.add_argument('--valid_path', nargs = '+', type = str, help = 'Validate path')
    parser.add_argument('--output_dir', type = str, default = '', help = 'Path to store results as wav file')
    parser.add_argument('--device_ids', nargs = '+', type = int, default = [0], help = 'List of gpu ids')
    parser.add_argument('--num_workers', type = int, default = 0, help = 'Dataloader num workers')
    parser.add_argument('--metrics', nargs = '+', type = str, default = ['sdr'], 
        choices = ['k_sdr', 'sdr', 'l1_freq', 'si_sdr', 'neg_log_wmse', 'aura_stft', 'aura_mrstft', 'bleedless', 'fullness', 'l1_snr'], help = 'List of metrics to use')
    parser.add_argument("--use_tta", action='store_true', 
        help="Flag adds test time augmentation during inference (polarity and channel inverse). While this triples the runtime, it reduces noise and slightly improves prediction quality.")
    parser.add_argument("--draw_spectro", type=float, default=0, 
        help="If --store_dir is set then code will generate spectrograms for resulted stems as well. Value defines for how many seconds os track spectrogram will be generated.")
    parser.add_argument("--lora_checkpoint_peft", type=str, default='', help="Initial checkpoint to LoRA weights")

    if dict_args is not None:
        args = parser.parse_args([])
        args_dict = vars(args)
        args_dict.update(dict_args)
        args = argparse.Namespace(**args_dict)
    else:
        args = parser.parse_args()
    
    return args

def check_validation(dict_args):
    # Truyền thông số được nhập từ Command Prompt
    args = parse_args_valid(dict_args)
    # Một cờ cấu hình trong PyTorch dùng để tối ưu hiệu năng khi chạy mô hình trên GPU với cuDNN (thư viện tăng tốc elearning của NVIDIA)
    torch.backends.cudnn.benchmark = True

    try:
        torch.multiprocessing.set_start_method('spawn')
    except Exception as e:
        pass
    
    config = ConfigDict(yaml.load(open(args.config_path), Loader = yaml.FullLoader))
    model = get_model_from_config(config)

    if args.model_path:
        checkpoint = torch.load(args.model_path, weights_only=False, map_location='cpu')
        load_start_checkpoint(args, model, checkpoint, type_ = 'valid')

    if args.lora_checkpoint_peft:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_checkpoint_peft)
        model = model.merge_and_unload()
    
    print(f'Instruments: {config.training.instruments}')
    device_ids = args.device_ids

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{device_ids[0]}')
    else:
        device = 'cpu'
        print('CUDA is not available. Run validation on CPU. It will be very slow...')
    # Đánh giá kết quả mô hình
    if torch.cuda.is_available() and len(device_ids) > 1:
        metrics = valid_multi_gpu(model, args, config, device_ids, verbose = False)
    else:
        metrics = valid(model, args, config, device, verbose = True)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return metrics

if __name__ == '__main__':
    check_validation(None)

