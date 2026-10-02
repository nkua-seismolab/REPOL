"""Tests for the DL input augmentation and batch prediction (no model required)."""

from types import SimpleNamespace

import numpy as np
import pytest
from obspy import Stream, Trace

from repol.core.polarity import PolarityClassifier, augment, tf


def make_stream(data):
    return Stream([Trace(data=np.asarray(data, dtype=float))])


def test_augment_shape_and_normalization():
    rng = np.random.default_rng(42)
    st = make_stream(rng.normal(size=200))
    out = augment(st)
    assert out.shape == (1, 128, 2)
    assert np.isclose(out[0, :, 0].mean(), 0.0, atol=1e-9)
    assert np.isclose(out[0, :, 0].std(), 1.0, atol=1e-9)


def test_augment_diff_sign_channel():
    st = make_stream(np.arange(200, dtype=float))
    out = augment(st)
    # strictly increasing waveform -> all diff signs are +1
    assert np.all(out[0, 65:, 1] == 1.0)
    assert np.all(out[0, :65, 1] == 0.0)


def test_augment_rejects_null_waveform():
    with pytest.raises(ValueError, match="Null waveform"):
        augment(make_stream(np.zeros(200)))


def test_augment_rejects_flat_waveform():
    # constant non-zero waveform has zero std
    with pytest.raises(ValueError, match="Zero standard deviation"):
        augment(make_stream(np.full(200, -1.0)))


class _StubModel:
    """Mimics the DiTingMotion output heads for controlled class scores."""

    def __init__(self, fmp_rows, cla_rows):
        self.fmp = np.asarray(fmp_rows, dtype=float)
        self.cla = np.asarray(cla_rows, dtype=float)

    def predict(self, model_inputs, verbose=0):
        assert model_inputs.shape[1:] == (128, 2)
        assert model_inputs.shape[0] == self.fmp.shape[0]
        out = {}
        for i in range(4):
            out[f"T0D{i}"] = self.fmp
            out[f"T1D{i}"] = self.cla
        return out


def make_classifier(fmp_rows, cla_rows):
    classifier = PolarityClassifier.__new__(PolarityClassifier)
    classifier.model = _StubModel(fmp_rows, cla_rows)
    return classifier


def test_predict_batch_codes():
    # rows: U/impulsive, D/emergent, undecidable (onset forced to x)
    classifier = make_classifier(
        fmp_rows=[[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
        cla_rows=[[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.9, 0.05, 0.05]],
    )
    batch = np.zeros((3, 128, 2))
    assert classifier.predict_batch(batch) == [("U", "I"), ("D", "E"), ("x", "x")]


def test_predict_single_matches_batch_row():
    classifier = make_classifier(fmp_rows=[[0.1, 0.8, 0.1]], cla_rows=[[0.7, 0.2, 0.1]])
    assert classifier.predict(np.zeros((1, 128, 2))) == ("D", "I")


@pytest.mark.parametrize(
    "rate,half_window,samples", [(100, 0.64, 128), (50, 1.28, 128), (100, 1.28, 256)]
)
def test_retrained_model_input_settings(monkeypatch, rate, half_window, samples):
    monkeypatch.setattr(
        tf.keras.models,
        "load_model",
        lambda *args, **kwargs: SimpleNamespace(input_shape=(None, samples, 2)),
    )
    classifier = PolarityClassifier("model.hdf5", rate, half_window)
    assert classifier.n_samples == samples
    output = augment(make_stream(np.arange(samples + 1)), classifier.n_samples)
    assert output.shape == (1, samples, 2)
    assert np.all(output[0, samples // 2 + 1 :, 1] == 1)


def test_incompatible_model_window_rejected(monkeypatch):
    monkeypatch.setattr(
        tf.keras.models,
        "load_model",
        lambda *args, **kwargs: SimpleNamespace(input_shape=(None, 128, 2)),
    )
    with pytest.raises(ValueError, match="sample count"):
        PolarityClassifier("model.hdf5", 50, 0.64)


@pytest.mark.parametrize("shape", [(None, 128, 1), (None, None, 2), (None, 127, 2)])
def test_incompatible_model_shape_rejected(monkeypatch, shape):
    monkeypatch.setattr(
        tf.keras.models, "load_model", lambda *args, **kwargs: SimpleNamespace(input_shape=shape)
    )
    with pytest.raises(ValueError, match="Model input"):
        PolarityClassifier("model.hdf5")
