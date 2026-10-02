"""First-motion polarity and onset classification with the DiTingMotion model.

Augmentation and prediction logic adapted from
https://github.com/mingzhaochina/DiTing-FOCALFLOW
"""

from __future__ import annotations

import logging
import math

import numpy as np
import tensorflow as tf
from obspy import Stream

logger = logging.getLogger(__name__)


def augment(st: Stream, n_samples: int = 128) -> np.ndarray:
    """Build the (1, n_samples, 2) input from the vertical-component window.

    Channel 0 holds the normalized waveform, channel 1 the sign of the
    first-order difference of its second half.
    """
    model_input = np.zeros([1, n_samples, 2])
    model_input[0, :, 0] = st[0].data[:n_samples]

    if np.max(model_input[0, :, 0]) == 0:
        raise ValueError("Null waveform detected")

    model_input[0, :, 0] -= np.mean(model_input[0, :, 0])
    norm_factor = np.std(model_input[0, :, 0])
    if norm_factor == 0:
        raise ValueError("Zero standard deviation detected")
    model_input[0, :, 0] /= norm_factor

    midpoint = n_samples // 2
    diff_data = np.diff(model_input[0, midpoint:, 0])
    model_input[0, midpoint + 1 :, 1] = np.sign(diff_data)
    return model_input


class PolarityClassifier:
    """Wraps the DiTingMotion Keras model."""

    def __init__(self, model_path: str, sampling_rate: float = 100.0, half_window: float = 0.64):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        shape = self.model.input_shape
        if (
            not isinstance(shape, tuple)
            or len(shape) != 3
            or shape[2] != 2
            or not isinstance(shape[1], int)
            or shape[1] < 2
            or shape[1] % 2
        ):
            raise ValueError("Model input must have shape (batch, even sample count, 2)")
        self.n_samples = shape[1]
        if not math.isclose(2 * sampling_rate * half_window, self.n_samples, abs_tol=1e-6):
            raise ValueError(
                "waveform.target_sampling_rate * 2 * waveform.model_half_window "
                f"must equal the model input sample count ({self.n_samples})"
            )
        logger.info("Loaded TensorFlow model from %s", model_path)

    def predict(self, model_input: np.ndarray) -> tuple[str, str]:
        """Return (polarity, onset) as U/D/x and I/E/x codes."""
        return self.predict_batch(model_input)[0]

    def predict_batch(self, model_inputs: np.ndarray) -> list[tuple[str, str]]:
        """Classify a batch using the DiTingMotion polarity and onset heads."""
        pred_res = self.model.predict(model_inputs, verbose=0)
        pred_fmp = (pred_res["T0D0"] + pred_res["T0D1"] + pred_res["T0D2"] + pred_res["T0D3"]) / 4
        pred_cla = (pred_res["T1D0"] + pred_res["T1D1"] + pred_res["T1D2"] + pred_res["T1D3"]) / 4

        results = []
        for fmp_row, cla_row in zip(pred_fmp, pred_cla):
            polarity = "x"
            onset = "x"
            fmp_class = np.argmax(fmp_row)
            cla_class = np.argmax(cla_row)
            if fmp_class == 1:
                polarity = "D"
            elif fmp_class == 0:
                polarity = "U"
            if polarity != "x":
                if cla_class == 0:
                    onset = "I"
                elif cla_class == 1:
                    onset = "E"
            results.append((polarity, onset))
        return results

    def classify(self, st: Stream) -> tuple[str, str]:
        return self.predict(augment(st, self.n_samples))
