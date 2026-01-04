import pytest
import yaml
import shutil
import os
import sys
from pathlib import Path

from restore_audio_hybrid import load_config

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestConfig:
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        # Backup config.yaml if it exists
        self.config_path = Path("config.yaml")
        self.backup_path = Path("config.yaml.bak")
        self.had_config = False

        if self.config_path.exists():
            self.had_config = True
            shutil.move(str(self.config_path), str(self.backup_path))

        yield

        # Restore
        if self.config_path.exists():
            self.config_path.unlink()

        if self.had_config:
            shutil.move(str(self.backup_path), str(self.config_path))

    def test_load_defaults(self):
        """Test that defaults are returned when no config file exists."""
        if self.config_path.exists():
            self.config_path.unlink()

        config, source = load_config()

        assert source == "Defaults"
        assert config["vocal_mix_volume"] == 1.0
        assert config["music_mix_volume"] == 1.0
        assert ".mp4" in config["extensions"]
        assert config["enhance_nfe"] == 128
        assert config["vocals_model"] == "model_bs_roformer_ep_317_sdr_12.9755.ckpt"

    def test_load_custom_config(self):
        """Test loading values from a custom config.yaml."""
        custom_data = {
            "vocal_mix_volume": 0.8,
            "music_mix_volume": 1.2,
            "extensions": [".webm"],
            "enhance_nfe": 64,
            "vocals_model": "custom_model.ckpt"
        }
        with open(self.config_path, "w") as f:
            yaml.dump(custom_data, f)

        config, source = load_config()

        assert source == "config.yaml"
        assert config["vocal_mix_volume"] == 0.8
        assert config["music_mix_volume"] == 1.2
        assert config["extensions"] == [".webm"]
        assert config["enhance_nfe"] == 64
        assert config["vocals_model"] == "custom_model.ckpt"

    def test_partial_config(self):
        """Test that defaults are preserved for missing keys."""
        partial_data = {
            "vocal_mix_volume": 0.5
        }
        with open(self.config_path, "w") as f:
            yaml.dump(partial_data, f)

        config, source = load_config()

        assert config["vocal_mix_volume"] == 0.5
        assert config["music_mix_volume"] == 1.0  # Default
        assert config["enhance_nfe"] == 128  # Default
