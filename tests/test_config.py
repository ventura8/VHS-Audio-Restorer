from unittest.mock import MagicMock, patch
import modules.config


@patch("modules.config.yaml.safe_load")
@patch("modules.config.Path.exists", return_value=True)
def test_load_config_fail(mock_exists, mock_load):
    """Test config loading failure handling."""
    mock_load.side_effect = Exception("YAML parse error")
    with patch("builtins.open", MagicMock()):
        conf, src = modules.config.load_config()
        assert src == "Defaults"
