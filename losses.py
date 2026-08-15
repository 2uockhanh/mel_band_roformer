import argparse
import torch

from ml_collections import ConfigDict
from typing import Callable, Any, Union, Optional
from torch import nn
# Một biến đổi mạnh mẽ của MSE, trong đó lỗi theo từng phần được tính và lỗi lớn nhất được loại bỏ dựa trên quantile (điểm phân vị)
def masked_loss(y_: torch.Tensor, y: torch.Tensor, q: float, coarse: bool = True) -> torch.Tensor:
    loss = nn.MSELoss(reduction = 'none')(y_, y).transpose(0, 1)

    if coarse:
        loss = loss.mean(dim = (-1, -2))

    loss = loss.reshape(loss.shape[0], -1)
    quantile = torch.quantile(loss.detach(), q, interpolation = 'linear', dim = 1, keepdim = True)
    mask = loss < quantile

    return (loss * mask).mean()

def choice_loss(args: argparse.Namespace, config: ConfigDict) -> Callable[[Any, Any, Union[Any, None]], torch.Tensor]:
    loss_fns = []
    if 'masked_loss' in args.loss:
        loss_fns.append(
            lambda y_pred, y_true, x = None:
            masked_loss(y_pred, y_true, q = config['training']['q'], coarse = config['training']['coarse_loss_clip']) * args.masked_loss_coef
        )

    def multi_loss(y_pred: Any, y_true: Any, x: Optional[Any] = None) -> torch.Tensor:
        total = 0
        for fn in loss_fns:
            total = total + fn(y_pred, y_true, x)
        return total

    return multi_loss