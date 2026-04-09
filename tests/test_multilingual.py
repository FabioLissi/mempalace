"""Tests for multilingual embedding support (get_embedding_function)."""

import json
import os
from unittest.mock import patch, MagicMock

import pytest

from mempalace.config import (
    MempalaceConfig,
    get_embedding_function,
)


@pytest.fixture(autouse=True)
def reset_embedding_cache():
    """Reset the module-level embedding function cache before each test."""
    import mempalace.config as cfg_mod

    cfg_mod._embedding_function = None
    cfg_mod._embedding_function_resolved = False
    yield
    cfg_mod._embedding_function = None
    cfg_mod._embedding_function_resolved = False


@pytest.fixture
def config_dir(tmp_path):
    """Create a temporary config directory with config.json."""
    cfg_dir = tmp_path / ".mempalace"
    cfg_dir.mkdir()
    return cfg_dir


class TestGetEmbeddingFunctionDefault:
    """When no model is configured, get_embedding_function returns None."""

    def test_returns_none_no_config(self, tmp_path):
        config = MempalaceConfig(config_dir=str(tmp_path / "empty"))
        result = get_embedding_function(config=config)
        assert result is None

    def test_returns_none_empty_config(self, config_dir):
        config_file = config_dir / "config.json"
        config_file.write_text("{}")
        config = MempalaceConfig(config_dir=str(config_dir))
        result = get_embedding_function(config=config)
        assert result is None


class TestGetEmbeddingFunctionEnvVar:
    """MEMPALACE_EMBEDDING_MODEL env var selects the model."""

    def test_env_var_triggers_import(self, tmp_path):
        mock_ef = MagicMock()
        mock_st_cls = MagicMock(return_value=mock_ef)
        config = MempalaceConfig(config_dir=str(tmp_path / "empty"))

        with (
            patch.dict(os.environ, {"MEMPALACE_EMBEDDING_MODEL": "intfloat/multilingual-e5-base"}),
            patch(
                "mempalace.config.SentenceTransformerEmbeddingFunction",
                mock_st_cls,
                create=True,
            ),
            patch(
                "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction",
                mock_st_cls,
            ),
        ):
            result = get_embedding_function(config=config)

        assert result is mock_ef
        mock_st_cls.assert_called_once_with(model_name="intfloat/multilingual-e5-base")

    def test_env_var_overrides_config_file(self, config_dir):
        config_file = config_dir / "config.json"
        config_file.write_text(json.dumps({"embedding_model": "from-config"}))
        config = MempalaceConfig(config_dir=str(config_dir))

        mock_ef = MagicMock()
        mock_st_cls = MagicMock(return_value=mock_ef)

        with (
            patch.dict(os.environ, {"MEMPALACE_EMBEDDING_MODEL": "from-env"}),
            patch(
                "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction",
                mock_st_cls,
            ),
        ):
            get_embedding_function(config=config)

        mock_st_cls.assert_called_once_with(model_name="from-env")


class TestGetEmbeddingFunctionConfigFile:
    """config.json embedding_model key selects the model."""

    def test_config_file_model(self, config_dir):
        config_file = config_dir / "config.json"
        config_file.write_text(json.dumps({"embedding_model": "intfloat/multilingual-e5-base"}))
        config = MempalaceConfig(config_dir=str(config_dir))

        mock_ef = MagicMock()
        mock_st_cls = MagicMock(return_value=mock_ef)

        with patch(
            "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction",
            mock_st_cls,
        ):
            result = get_embedding_function(config=config)

        assert result is mock_ef
        mock_st_cls.assert_called_once_with(model_name="intfloat/multilingual-e5-base")


class TestGetEmbeddingFunctionFallback:
    """Graceful fallback when sentence-transformers is not installed."""

    def test_import_error_returns_none(self, config_dir):
        config_file = config_dir / "config.json"
        config_file.write_text(json.dumps({"embedding_model": "some-model"}))
        config = MempalaceConfig(config_dir=str(config_dir))

        with patch(
            "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction",
            side_effect=ImportError("No module named 'sentence_transformers'"),
        ):
            result = get_embedding_function(config=config)

        assert result is None


class TestGetEmbeddingFunctionCaching:
    """Result is cached after first call."""

    def test_caches_result(self, config_dir):
        config_file = config_dir / "config.json"
        config_file.write_text(json.dumps({"embedding_model": "test-model"}))
        config = MempalaceConfig(config_dir=str(config_dir))

        mock_ef = MagicMock()
        mock_st_cls = MagicMock(return_value=mock_ef)

        with patch(
            "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction",
            mock_st_cls,
        ):
            result1 = get_embedding_function(config=config)
            result2 = get_embedding_function(config=config)

        assert result1 is result2
        # Constructor called only once due to caching
        assert mock_st_cls.call_count == 1

    def test_caches_none_result(self, tmp_path):
        config = MempalaceConfig(config_dir=str(tmp_path / "empty"))
        result1 = get_embedding_function(config=config)
        result2 = get_embedding_function(config=config)
        assert result1 is None
        assert result2 is None


class TestEmbeddingModelProperty:
    """MempalaceConfig.embedding_model property."""

    def test_returns_none_by_default(self, tmp_path):
        config = MempalaceConfig(config_dir=str(tmp_path / "empty"))
        assert config.embedding_model is None

    def test_reads_from_config_file(self, config_dir):
        config_file = config_dir / "config.json"
        config_file.write_text(json.dumps({"embedding_model": "my-model"}))
        config = MempalaceConfig(config_dir=str(config_dir))
        assert config.embedding_model == "my-model"

    def test_env_var_overrides(self, config_dir):
        config_file = config_dir / "config.json"
        config_file.write_text(json.dumps({"embedding_model": "file-model"}))
        config = MempalaceConfig(config_dir=str(config_dir))
        with patch.dict(os.environ, {"MEMPALACE_EMBEDDING_MODEL": "env-model"}):
            assert config.embedding_model == "env-model"
