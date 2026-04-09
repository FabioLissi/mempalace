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
    """When no model is configured, get_embedding_function returns ChromaDB's default."""

    def test_returns_default_no_config(self, tmp_path):
        config = MempalaceConfig(config_dir=str(tmp_path / "empty"))
        result = get_embedding_function(config=config)
        # Must not be None — newer ChromaDB requires an explicit callable
        # so that `collection.add()` can compute embeddings.
        assert result is not None
        assert callable(result)

    def test_returns_default_empty_config(self, config_dir):
        config_file = config_dir / "config.json"
        config_file.write_text("{}")
        config = MempalaceConfig(config_dir=str(config_dir))
        result = get_embedding_function(config=config)
        assert result is not None
        assert callable(result)


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

    def test_import_error_falls_back_to_default(self, config_dir):
        config_file = config_dir / "config.json"
        config_file.write_text(json.dumps({"embedding_model": "some-model"}))
        config = MempalaceConfig(config_dir=str(config_dir))

        with patch(
            "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction",
            side_effect=ImportError("No module named 'sentence_transformers'"),
        ):
            result = get_embedding_function(config=config)

        # Falls back to ChromaDB's DefaultEmbeddingFunction, not None
        assert result is not None
        assert callable(result)


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

    def test_caches_default_result(self, tmp_path):
        """The default embedding function is also cached between calls."""
        config = MempalaceConfig(config_dir=str(tmp_path / "empty"))
        result1 = get_embedding_function(config=config)
        result2 = get_embedding_function(config=config)
        # Same instance returned (cached), and never None
        assert result1 is result2
        assert result1 is not None


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


class TestEmbeddingDeviceProperty:
    """MempalaceConfig.embedding_device property."""

    def test_returns_none_by_default(self, tmp_path):
        config = MempalaceConfig(config_dir=str(tmp_path / "empty"))
        assert config.embedding_device is None

    def test_reads_from_config_file(self, config_dir):
        config_file = config_dir / "config.json"
        config_file.write_text(json.dumps({"embedding_device": "mps"}))
        config = MempalaceConfig(config_dir=str(config_dir))
        assert config.embedding_device == "mps"

    def test_env_var_overrides(self, config_dir):
        config_file = config_dir / "config.json"
        config_file.write_text(json.dumps({"embedding_device": "cpu"}))
        config = MempalaceConfig(config_dir=str(config_dir))
        with patch.dict(os.environ, {"MEMPALACE_EMBEDDING_DEVICE": "mps"}):
            assert config.embedding_device == "mps"


class TestGetEmbeddingFunctionDevice:
    """MEMPALACE_EMBEDDING_DEVICE controls the device passed to the embedder."""

    def test_device_passed_to_embedder_with_explicit_model(self, tmp_path):
        """When both model and device are set, both are passed through."""
        mock_ef = MagicMock()
        mock_st_cls = MagicMock(return_value=mock_ef)
        config = MempalaceConfig(config_dir=str(tmp_path / "empty"))

        with (
            patch.dict(
                os.environ,
                {
                    "MEMPALACE_EMBEDDING_MODEL": "intfloat/multilingual-e5-base",
                    "MEMPALACE_EMBEDDING_DEVICE": "mps",
                },
            ),
            patch(
                "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction",
                mock_st_cls,
            ),
        ):
            result = get_embedding_function(config=config)

        assert result is mock_ef
        mock_st_cls.assert_called_once_with(
            model_name="intfloat/multilingual-e5-base", device="mps"
        )

    def test_device_alone_activates_default_model(self, tmp_path):
        """Setting only the device should trigger the default model on that device.

        This is the ergonomic path for Apple Silicon / CUDA users: they
        don't need to know the model name, just the device.
        """
        mock_ef = MagicMock()
        mock_st_cls = MagicMock(return_value=mock_ef)
        config = MempalaceConfig(config_dir=str(tmp_path / "empty"))

        with (
            patch.dict(os.environ, {"MEMPALACE_EMBEDDING_DEVICE": "mps"}),
            patch(
                "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction",
                mock_st_cls,
            ),
        ):
            result = get_embedding_function(config=config)

        assert result is mock_ef
        mock_st_cls.assert_called_once_with(
            model_name="sentence-transformers/all-MiniLM-L6-v2", device="mps"
        )

    def test_no_device_no_kwarg(self, tmp_path, monkeypatch):
        """When no device is set, ``device`` is NOT passed as a kwarg.

        This preserves backward compatibility with the original PR #442
        behavior where only ``model_name`` was passed.
        """
        mock_ef = MagicMock()
        mock_st_cls = MagicMock(return_value=mock_ef)
        config = MempalaceConfig(config_dir=str(tmp_path / "empty"))

        monkeypatch.setenv("MEMPALACE_EMBEDDING_MODEL", "some-model")
        monkeypatch.delenv("MEMPALACE_EMBEDDING_DEVICE", raising=False)

        with patch(
            "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction",
            mock_st_cls,
        ):
            result = get_embedding_function(config=config)

        assert result is mock_ef
        mock_st_cls.assert_called_once_with(model_name="some-model")

    def test_device_from_config_file(self, config_dir, monkeypatch):
        """Device can be set via config.json instead of env var."""
        config_file = config_dir / "config.json"
        config_file.write_text(
            json.dumps(
                {
                    "embedding_model": "intfloat/multilingual-e5-base",
                    "embedding_device": "cuda",
                }
            )
        )
        config = MempalaceConfig(config_dir=str(config_dir))

        mock_ef = MagicMock()
        mock_st_cls = MagicMock(return_value=mock_ef)

        monkeypatch.delenv("MEMPALACE_EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("MEMPALACE_EMBEDDING_DEVICE", raising=False)

        with patch(
            "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction",
            mock_st_cls,
        ):
            result = get_embedding_function(config=config)

        assert result is mock_ef
        mock_st_cls.assert_called_once_with(
            model_name="intfloat/multilingual-e5-base", device="cuda"
        )
