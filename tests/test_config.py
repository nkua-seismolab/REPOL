"""Tests for repol.config."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from repol import __main__ as cli
from repol.config import ConfigError, load_config


def write(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


def test_defaults_from_empty_file(tmp_path):
    config = load_config(write(tmp_path, ""))
    assert config.seiscomp.wait_time == 300
    assert config.logging.level == "DEBUG"
    assert config.spratio.vp == 6.0
    assert config.waveform.decimate is True


def test_partial_override(tmp_path):
    config = load_config(
        write(tmp_path, "seiscomp:\n  wait_time: 60\nspratio:\n  vp: 5.8\n  vs: 3.2\n")
    )
    assert config.seiscomp.wait_time == 60
    assert config.spratio.vp == 5.8
    assert config.spratio.vs == 3.2
    # untouched sections keep defaults
    assert config.sds.archive == "/data/archive"


@pytest.mark.parametrize(
    "key", ["bogus", "subscriptions", "pick_group", "focmech_group", "load_inventory"]
)
def test_unknown_key_rejected(tmp_path, key):
    with pytest.raises(ConfigError, match="Unknown keys"):
        load_config(write(tmp_path, f"seiscomp:\n  {key}: 1\n"))


def test_unknown_section_rejected(tmp_path):
    with pytest.raises(ConfigError, match="Unknown top-level"):
        load_config(write(tmp_path, "bogus:\n  a: 1\n"))


def test_invalid_velocities_rejected(tmp_path):
    with pytest.raises(ConfigError, match="vp > vs"):
        load_config(write(tmp_path, "spratio:\n  vp: 3.0\n  vs: 6.0\n"))


def test_invalid_filter_rejected(tmp_path):
    with pytest.raises(ConfigError, match="filter"):
        load_config(write(tmp_path, "waveform:\n  filter: [1.0]\n"))


def test_missing_file_rejected(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


@pytest.mark.parametrize(
    "text",
    [
        "false",
        "[]",
        "seiscomp: false",
        "seiscomp: []",
        "seiscomp: null",
        "seiscomp: {wait_time: true}",
        "seiscomp: {wait_time: -1}",
        "seiscomp: {reprocess: 'false'}",
        "logging: {level: nonsense}",
        "waveform: {target_sampling_rate: .nan}",
        "logging: [",
    ],
)
def test_invalid_config_rejected(tmp_path, text):
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, text))


def test_release_example():
    config = load_config(Path(__file__).resolve().parents[1] / "config.example.yaml")
    assert config.logging.level == "DEBUG"


@pytest.mark.parametrize("option", ["--help", "--version"])
def test_informational_cli_needs_no_config(monkeypatch, option):
    monkeypatch.setattr(cli.sys, "argv", ["repol", option, "--config", "/missing.yaml"])
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 0


@pytest.mark.parametrize("args", [["--wait-time", "-1"], ["-w-1"], ["--config"]])
def test_invalid_cli_rejected(args):
    with pytest.raises(SystemExit) as result:
        cli._config_path_from_argv(["repol", *args])
    assert result.value.code == 2


def test_cli_precedence(tmp_path):
    config = load_config(write(tmp_path, "seiscomp: {host: yaml-host, database: yaml-db}"))
    for arguments in (
        ["-H", "cli-host", "-d", "cli-db"],
        ["-Hcli-host", "-dcli-db"],
        ["--host=cli-host", "--database=cli-db"],
    ):
        argv = ["repol", *arguments]
        assert cli._inject_connection_args(argv, config) == argv


def test_worker_failure_stops_before_listener(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(cli.sys, "argv", ["repol", "--config", str(write(tmp_path, ""))])
    monkeypatch.setattr(cli.faulthandler, "register", Mock())
    executor = Mock()
    executor.submit.return_value.result.side_effect = RuntimeError("failed")
    monkeypatch.setattr(cli, "ProcessPoolExecutor", Mock(return_value=executor))
    listener = Mock()
    monkeypatch.setitem(
        cli.sys.modules, "repol.app", SimpleNamespace(make_dispatch=Mock(), make_finalize=Mock())
    )
    monkeypatch.setitem(
        cli.sys.modules, "repol.worker", SimpleNamespace(init_worker=Mock(), run_batch=Mock())
    )
    monkeypatch.setitem(
        cli.sys.modules, "repol.scclient.listener", SimpleNamespace(EventListenerApp=listener)
    )
    assert cli.main() == 1
    listener.assert_not_called()
    executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
    assert "Worker initialization failed" in caplog.text
    assert "Worker process ready" not in caplog.text
