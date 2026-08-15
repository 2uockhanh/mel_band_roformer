import numpy as np
import math
import torch
import torch.nn.functional as F
import librosa

from torch_log_wmse import LogWMSE
from torch_l1_snr import MultiL1SNRDBLoss
from auraloss.freq import STFTLoss, MultiResolutionSTFTLoss
from typing import Tuple, Dict, List
from torchaudio.transforms import AmplitudeToDB
# Performance Measurement in Blind Audio Source Separation (2006). https://scispace.com/pdf/performance-measurement-in-blind-audio-source-separation-4la127li8d.pdf
def sdr(references: np.ndarray, estimates: np.ndarray) -> float:
    eps = 1e-8
    num = np.sum(np.square(references), axis = (1, 2)) #(1, 1)
    den = np.sum(np.square(references - estimates), axis = (1, 2)) # (1, 1)
    num += eps
    den += eps
    return 10 * np.log10(num / den)

def k_sdr(sdr: float, K: float = 10.0) -> float:
    sdr = max(min(sdr, K), -K + 1e-6)
    return 100.0 * math.log1p(sdr + K) / math.log1p(2 * K)
# SDR - Half-Baked Or Well Done? https://arxiv.org/pdf/1811.02508
def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    eps = 1e-8
    scale = np.sum(estimate * reference + eps, axis = (0, 1)) / np.sum(reference ** 2 + eps, axis = (0, 1))
    scale = np.expand_dims(scale, axis = (0, 1))

    reference = reference * scale
    si_sdr = np.mean(10 * np.log10(np.sum(reference ** 2, axis = (0, 1)) / (np.sum((reference - estimate) ** 2, axis = (0, 1)) + eps) + eps))

    return si_sdr
# On Loss Functions Add Evaluation Metrics For Music Source Separation. https://arxiv.org/pdf/2202.07968
def L1Freq_metric(reference: np.ndarray, estimate: np.ndarray, fft_size: int = 2048, hop_size: int = 1024, device: str = 'cpu') -> float:
    reference = torch.from_numpy(reference).to(device)
    estimate = torch.from_numpy(estimate).to(device)

    reference_stft = torch.stft(reference, fft_size, hop_size, return_complex = True)
    estimated_stft = torch.stft(estimate, fft_size, hop_size, return_complex = True)

    reference_mag = torch.abs(reference_stft)
    estimate_mag = torch.abs(estimated_stft)

    loss = 10 * F.l1_loss(estimate_mag, reference_mag)

    ret = 100 / (1. + float(loss.cpu().numpy()))

    return ret

def LogWMSE_metric(reference: np.ndarray, estimate: np.ndarray, mixture: np.ndarray, device: str = 'cpu', ) -> float:
    log_wmse = LogWMSE(audio_length = reference.shape[-1] / 44100, sample_rate = 44100, return_as_loss = False, bypass_filter = False, )

    reference = torch.from_numpy(reference).unsqueeze(0).unsqueeze(0).to(device)
    estimate = torch.from_numpy(estimate).unsqueeze(0).unsqueeze(0).to(device)
    mixture = torch.from_numpy(mixture).unsqueeze(0).to(device)

    res = log_wmse(mixture, reference, estimate)
    return float(res.cpu().numpy())

def MultiL1SNRDB_metric(reference: np.ndarray, estimate: np.ndarray, device: str = 'cpu', ) -> float:
    l1_snr = MultiL1SNRDBLoss(name = "l1_snr_metric", weight = 1.0, spec_weight = 0.5, l1_weight = 0.0, use_time_regularization = True, use_spec_regularization = False, )
    
    reference_t = torch.from_numpy(reference).unsqueeze(0).to(device)
    estimate_t = torch.from_numpy(estimate).unsqueeze(0).to(device)

    with torch.no_grad():
        res = l1_snr(estimate_t, reference_t)

    return -float(res.cpu().numpy())

def AuraSTFT_metric(reference: np.ndarray, estimate: np.ndarray, device: str = 'cpu', ) -> float:
    stft_loss = STFTLoss(w_log_mag = 1.0, w_lin_mag = 0.0, w_sc = 1.0, device = device, )

    reference = torch.from_numpy(reference).unsqueeze(0).to(device)
    estimate = torch.from_numpy(estimate).unsqueeze(0).to(device)

    res = 100 / (1. + 10 * stft_loss(reference, estimate))
    return float(res.cpu().numpy())

def AuraMRSTFT_metric(reference: np.ndarray, estimate: np.ndarray, device: str = 'cpu', ) -> float:
    mrstft_loss = MultiResolutionSTFTLoss(
        fft_sizes = [1024, 2048, 4096],
        hop_sizes = [256, 512, 1024],
        win_lengths = [1024, 2048, 4096],
        scale = "mel",
        n_bins = 128,
        sample_rate = 44100,
        perceptual_weighting = True,
        device = device
    )

    reference = torch.from_numpy(reference).unsqueeze(0).float().to(device)
    estimate = torch.from_numpy(estimate).unsqueeze(0).float().to(device)

    res = 100 / (1. + 10 * mrstft_loss(reference, estimate))
    return float(res.cpu().numpy())

def bleed_full(reference: np.ndarray, estimate: np.ndarray, sr: int = 44100, n_fft: int = 4096, hop_length: int = 1024, n_mels: int = 512, 
               device: str = 'cpu', ) -> Tuple[float, float]:
    reference = torch.from_numpy(reference).float().to(device)
    estimate = torch.from_numpy(estimate).float().to(device)

    window = torch.hann_window(n_fft).to(device)

    D1 = torch.abs(
        torch.stft(reference, n_fft = n_fft, hop_length = hop_length, window = window, return_complex = True, pad_mode = 'constant')
    )
    D2 = torch.abs(
        torch.stft(estimate, n_fft = n_fft, hop_length = hop_length, window = window, return_complex = True, pad_mode = 'constant')
    )

    mel_basis = librosa.filters.mel(sr = sr, n_fft = n_fft, n_mels = n_mels)
    mel_filter_bank = torch.from_numpy(mel_basis).to(device)

    S1_mel = torch.matmul(mel_filter_bank, D1)
    S2_mel = torch.matmul(mel_filter_bank, D2)

    S1_db = AmplitudeToDB(stype = 'magnitude', top_db = 80)(S1_mel)
    S2_db = AmplitudeToDB(stype = 'magnitude', top_db = 80)(S2_mel)

    diff = S2_db - S1_db

    positive_diff = diff[diff > 0]
    negative_diff = diff[diff < 0]

    average_positive = torch.mean(positive_diff) if positive_diff.numel() > 0 else torch.tensor(0.0).to(device)
    average_negative = torch.mean(negative_diff) if negative_diff.numel() > 0 else torch.tensor(0.0).to(device)

    bleedless = 100 * 1 / (average_positive + 1)
    fullness = 100 * 1 / (-average_negative + 1)

    return bleedless.cpu().numpy(), fullness.cpu().numpy()
# Lấy giá trị các chỉ số đánh giá đã được tính dựa trên âm thanh đã phân tách và âm thanh gốc
def get_metrics(metrics: List[str], reference: np.ndarray, estimate: np.ndarray, mix: np.ndarray, device: str = 'cpu', k: float = 10) -> Dict[str, float]:
    result = dict()
    
    min_length = min(reference.shape[1], estimate.shape[1])
    reference = reference[..., :min_length]
    estimate = estimate[..., :min_length]
    mix = mix[..., :min_length]

    if 'sdr' in metrics or 'k_sdr' in metrics:
        references = np.expand_dims(reference, axis = 0)
        estimates = np.expand_dims(estimate, axis = 0)
        result['sdr'] = float(sdr(references, estimates)[0])
        result['k_sdr'] = k_sdr(float(sdr(references, estimates)[0]), k)
    
    if 'si_sdr' in metrics:
        result['si_sdr'] = si_sdr(reference, estimate)

    if 'l1_freq' in metrics:
        result['l1_freq'] = L1Freq_metric(reference, estimate, device = device)
    
    if 'log_wmse' in metrics:
        result['log_wmse'] = LogWMSE_metric(reference, estimate, mix, device)
    
    if 'aura_stft' in metrics:
        result['aura_stft'] = AuraSTFT_metric(reference, estimate, device)
    
    if 'aura_mrstft' in metrics:
        result['aura_mrstft'] = AuraMRSTFT_metric(reference, estimate, device)

    if 'l1_snr' in metrics:
        result['l1_snr'] = MultiL1SNRDB_metric(reference, estimate, device)
    
    if 'bleedless' in metrics or 'fullness' in metrics:
        bleedless, fullness = bleed_full(reference, estimate, device = device)
        if 'bleedless' in metrics:
            result['bleedless'] = float(bleedless)
        
        if 'fullness' in metrics:
            result['fullness'] = float(fullness)

    return result