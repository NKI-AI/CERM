import numpy as np

from omegaconf import DictConfig
from optuna import Trial
from enum import Enum
from math import log2, floor
from logging import getLogger

logger = getLogger(__name__)


class Hypers(Enum):
    # Decoder
    order_wavelet = "network.decoder.order_wavelet"
    init_res = "network.decoder.init_res_level"
    num_down = "network.decoder.num_levels_down"
    num_up = "network.decoder.num_levels_up"
    num_compress = "network.decoder.num_channels_compress"

    # Encoder
    len_encoder = "network.encoder.len_encoder"
    len_enc_res = "network.encoder.len_res_block"
    init_kernels = "network.encoder.init_num_kernels"

    # Prediction branch contour
    len_pred = "network.mlp_contour.depth"
    dim_pred = "network.mlp_contour.dim_latent"

    # Learning rates
    encoder_lr = "trainer.encoder.lr"
    decoder_lr = "trainer.decoder.lr"
    warmup_epoch = "trainer.warmup_epochs"


def configure(cfg: DictConfig, trial: Trial) -> None:
    hypers = Hypers

    # Learning rates & training
    min_enc_lr = cfg.trainer.encoder.init_lr
    min_dec_lr = cfg.trainer.decoder.init_lr
    trial.suggest_float(hypers.encoder_lr.value, min_enc_lr, 1e-03)
    trial.suggest_float(hypers.decoder_lr.value, min_dec_lr, 1e-02)
    trial.suggest_int(hypers.warmup_epoch.value, 2, 30)

    # Network architecture (no contraints)
    trial.suggest_int(hypers.init_res.value, 6, 9)
    trial.suggest_int(hypers.len_enc_res.value, 1, 8)
    trial.suggest_int(hypers.num_compress.value, 4, 16, 4)
    trial.suggest_categorical(hypers.init_kernels.value, [16, 32, 64])
    trial.suggest_int(hypers.len_pred.value, 1, 4)
    trial.suggest_int(hypers.dim_pred.value, 32, 128)

    # Network architecture (constraints)

    # Maximal length encoder
    min_len_encoder = 4
    max_len_encoder = 0
    while (
        cfg.dataset.dim_img[0] % 2 ** (max_len_encoder + 1) == 0
        and cfg.dataset.dim_img[1] % 2 ** (max_len_encoder + 1) == 0
    ):
        max_len_encoder += 1
    trial.suggest_int(hypers.len_encoder.value, min_len_encoder, max_len_encoder)

    # Order wavelet
    trial.suggest_int(hypers.order_wavelet.value, 3, 10)

    # Num levels down as function of order wavelet
    len_filter = 2 * trial.params[hypers.order_wavelet.value] - 1
    min_levels_down = 1
    max_levels_down = np.floor(
        trial.params[hypers.init_res.value] - np.log(len_filter) / np.log(2)
    )
    trial.suggest_int(hypers.num_down.value, min_levels_down, max_levels_down)

    # Num levels up
    trial.suggest_int(
        hypers.num_up.value,
        1,
        min(
            trial.params[hypers.num_down.value], trial.params[hypers.len_encoder.value]
        ),
    )
