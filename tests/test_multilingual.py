"""Tests for multilingual embedding support (get_embedding_function)."""

import json
import os
from unittest.mock import patch, MagicMock

import pytest

from mempalace.config import (
    EmbeddingModelMismatchError,
    MempalaceConfig,
    get_embedding_function,
    get_embedding_model_name,
)
from mempalace.palace import get_collection

# All env vars that affect embedding resolution.  Tests that assume a clean
# environment must clear these to avoid leaking the host shell's config.
_EMBEDDING_ENV_VARS = (
    "MEMPALACE_EMBEDDING_MODEL",
    "MEMPALACE_EMBEDDING_DEVICE",
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
def clean_embedding_env(monkeypatch):
    """Remove all MEMPALACE_EMBEDDING_* env vars so tests start from a known state."""
    for var in _EMBEDDING_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def config_dir(tmp_path):
    """Create a temporary config directory with config.json."""
    cfg_dir = tmp_path / ".mempalace"
    cfg_dir.mkdir()
    return cfg_dir


class TestGetEmbeddingFunctionDefault:
    """When no model is configured, get_embedding_function returns ChromaDB's default."""

    def test_returns_default_no_config(self, tmp_path, clean_embedding_env):
        config = MempalaceConfig(config_dir=str(tmp_path / "empty"))
        result = get_embedding_function(config=config)
        assert result is not None
        assert callable(result)

    def test_returns_default_empty_config(self, config_dir, clean_embedding_env):
        config_file = config_dir / "config.json"
        config_file.write_text("{}")
        config = MempalaceConfig(config_dir=str(config_dir))
        result = get_embedding_function(config=config)
        assert result is not None
        assert callable(result)


class TestGetEmbeddingFunctionEnvVar:
    """MEMPALACE_EMBEDDING_MODEL env var selects the model."""

    def test_env_var_triggers_import(self, tmp_path, monkeypatch):
        mock_ef = MagicMock()
        mock_st_cls = MagicMock(return_value=mock_ef)
        config = MempalaceConfig(config_dir=str(tmp_path / "empty"))

        monkeypatch.setenv("MEMPALACE_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
        monkeypatch.delenv("MEMPALACE_EMBEDDING_DEVICE", raising=False)

        with patch(
            "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction",
            mock_st_cls,
        ):
            result = get_embedding_function(config=config)

        assert result is mock_ef
        mock_st_cls.assert_called_once_with(model_name="intfloat/multilingual-e5-base")

    def test_env_var_overrides_config_file(self, config_dir, monkeypatch):
        config_file = config_dir / "config.json"
        config_file.write_text(json.dumps({"embedding_model": "from-config"}))
        config = MempalaceConfig(config_dir=str(config_dir))

        mock_ef = MagicMock()
        mock_st_cls = MagicMock(return_value=mock_ef)

        monkeypatch.setenv("MEMPALACE_EMBEDDING_MODEL", "from-env")
        monkeypatch.delenv("MEMPALACE_EMBEDDING_DEVICE", raising=False)

        with patch(
            "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction",
            mock_st_cls,
        ):
            get_embedding_function(config=config)

        mock_st_cls.assert_called_once_with(model_name="from-env")


class TestGetEmbeddingFunctionConfigFile:
    """config.json embedding_model key selects the model."""

    def test_config_file_model(self, config_dir, clean_embedding_env):
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

    def test_import_error_falls_back_to_default(self, config_dir, clean_embedding_env):
        config_file = config_dir / "config.json"
        config_file.write_text(json.dumps({"embedding_model": "some-model"}))
        config = MempalaceConfig(config_dir=str(config_dir))

        with patch(
            "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction",
            side_effect=ImportError("No module named 'sentence_transformers'"),
        ):
            result = get_embedding_function(config=config)

        assert result is not None
        assert callable(result)


class TestGetEmbeddingFunctionCaching:
    """Result is cached after first call."""

    def test_caches_result(self, config_dir, clean_embedding_env):
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
        assert mock_st_cls.call_count == 1

    def test_caches_default_result(self, tmp_path, clean_embedding_env):
        config = MempalaceConfig(config_dir=str(tmp_path / "empty"))
        result1 = get_embedding_function(config=config)
        result2 = get_embedding_function(config=config)
        assert result1 is result2
        assert result1 is not None


class TestEmbeddingModelProperty:
    """MempalaceConfig.embedding_model property."""

    def test_returns_none_by_default(self, tmp_path, clean_embedding_env):
        config = MempalaceConfig(config_dir=str(tmp_path / "empty"))
        assert config.embedding_model is None

    def test_reads_from_config_file(self, config_dir, clean_embedding_env):
        config_file = config_dir / "config.json"
        config_file.write_text(json.dumps({"embedding_model": "my-model"}))
        config = MempalaceConfig(config_dir=str(config_dir))
        assert config.embedding_model == "my-model"

    def test_env_var_overrides(self, config_dir, monkeypatch):
        config_file = config_dir / "config.json"
        config_file.write_text(json.dumps({"embedding_model": "file-model"}))
        config = MempalaceConfig(config_dir=str(config_dir))
        monkeypatch.setenv("MEMPALACE_EMBEDDING_MODEL", "env-model")
        assert config.embedding_model == "env-model"


class TestEmbeddingDeviceProperty:
    """MempalaceConfig.embedding_device property."""

    def test_returns_none_by_default(self, tmp_path, clean_embedding_env):
        config = MempalaceConfig(config_dir=str(tmp_path / "empty"))
        assert config.embedding_device is None

    def test_reads_from_config_file(self, config_dir, clean_embedding_env):
        config_file = config_dir / "config.json"
        config_file.write_text(json.dumps({"embedding_device": "mps"}))
        config = MempalaceConfig(config_dir=str(config_dir))
        assert config.embedding_device == "mps"

    def test_env_var_overrides(self, config_dir, monkeypatch):
        config_file = config_dir / "config.json"
        config_file.write_text(json.dumps({"embedding_device": "cpu"}))
        config = MempalaceConfig(config_dir=str(config_dir))
        monkeypatch.setenv("MEMPALACE_EMBEDDING_DEVICE", "mps")
        assert config.embedding_device == "mps"


class TestGetEmbeddingFunctionDevice:
    """MEMPALACE_EMBEDDING_DEVICE controls the device passed to the embedder."""

    def test_device_passed_to_embedder_with_explicit_model(self, tmp_path, monkeypatch):
        """When both model and device are set, both are passed through."""
        mock_ef = MagicMock()
        mock_st_cls = MagicMock(return_value=mock_ef)
        config = MempalaceConfig(config_dir=str(tmp_path / "empty"))

        monkeypatch.setenv("MEMPALACE_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
        monkeypatch.setenv("MEMPALACE_EMBEDDING_DEVICE", "mps")

        with patch(
            "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction",
            mock_st_cls,
        ):
            result = get_embedding_function(config=config)

        assert result is mock_ef
        mock_st_cls.assert_called_once_with(
            model_name="intfloat/multilingual-e5-base", device="mps"
        )

    def test_device_alone_activates_default_model(self, tmp_path, monkeypatch):
        """Setting only the device should trigger the default model on that device."""
        mock_ef = MagicMock()
        mock_st_cls = MagicMock(return_value=mock_ef)
        config = MempalaceConfig(config_dir=str(tmp_path / "empty"))

        monkeypatch.delenv("MEMPALACE_EMBEDDING_MODEL", raising=False)
        monkeypatch.setenv("MEMPALACE_EMBEDDING_DEVICE", "mps")

        with patch(
            "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction",
            mock_st_cls,
        ):
            result = get_embedding_function(config=config)

        assert result is mock_ef
        mock_st_cls.assert_called_once_with(
            model_name="sentence-transformers/all-MiniLM-L6-v2", device="mps"
        )

    def test_no_device_no_kwarg(self, tmp_path, monkeypatch):
        """When no device is set, ``device`` is NOT passed as a kwarg."""
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


class TestGetEmbeddingModelName:
    """get_embedding_model_name() returns the canonical model identity string."""

    def test_default_no_config(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MEMPALACE_EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("MEMPALACE_EMBEDDING_DEVICE", raising=False)
        config = MempalaceConfig(config_dir=str(tmp_path / "empty"))
        assert get_embedding_model_name(config=config) == "chromadb-default"

    def test_explicit_model_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MEMPALACE_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
        monkeypatch.delenv("MEMPALACE_EMBEDDING_DEVICE", raising=False)
        config = MempalaceConfig(config_dir=str(tmp_path / "empty"))
        assert get_embedding_model_name(config=config) == "intfloat/multilingual-e5-base"

    def test_device_only_returns_default_model(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MEMPALACE_EMBEDDING_MODEL", raising=False)
        monkeypatch.setenv("MEMPALACE_EMBEDDING_DEVICE", "mps")
        config = MempalaceConfig(config_dir=str(tmp_path / "empty"))
        assert get_embedding_model_name(config=config) == "sentence-transformers/all-MiniLM-L6-v2"

    def test_explicit_model_from_config_file(self, config_dir, monkeypatch):
        monkeypatch.delenv("MEMPALACE_EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("MEMPALACE_EMBEDDING_DEVICE", raising=False)
        config_file = config_dir / "config.json"
        config_file.write_text(json.dumps({"embedding_model": "my-custom-model"}))
        config = MempalaceConfig(config_dir=str(config_dir))
        assert get_embedding_model_name(config=config) == "my-custom-model"

    def test_env_var_overrides_config_file(self, config_dir, monkeypatch):
        monkeypatch.delenv("MEMPALACE_EMBEDDING_DEVICE", raising=False)
        config_file = config_dir / "config.json"
        config_file.write_text(json.dumps({"embedding_model": "file-model"}))
        config = MempalaceConfig(config_dir=str(config_dir))
        monkeypatch.setenv("MEMPALACE_EMBEDDING_MODEL", "env-model")
        assert get_embedding_model_name(config=config) == "env-model"


class TestForceEmbeddingProperty:
    """MempalaceConfig.force_embedding property."""

    def test_returns_false_by_default(self, tmp_path, clean_embedding_env):
        config = MempalaceConfig(config_dir=str(tmp_path / "empty"))
        assert config.force_embedding is False

    def test_reads_from_config_file(self, config_dir, clean_embedding_env):
        config_file = config_dir / "config.json"
        config_file.write_text(json.dumps({"force_embedding": True}))
        config = MempalaceConfig(config_dir=str(config_dir))
        assert config.force_embedding is True

    def test_env_var_overrides(self, config_dir, monkeypatch):
        config_file = config_dir / "config.json"
        config_file.write_text(json.dumps({"force_embedding": False}))
        config = MempalaceConfig(config_dir=str(config_dir))
        monkeypatch.setenv("MEMPALACE_FORCE_EMBEDDING", "true")
        assert config.force_embedding is True

    def test_env_var_true_case_insensitive(self, tmp_path, monkeypatch):
        config = MempalaceConfig(config_dir=str(tmp_path / "empty"))
        monkeypatch.setenv("MEMPALACE_FORCE_EMBEDDING", "True")
        assert config.force_embedding is True

    def test_env_var_false(self, tmp_path, monkeypatch):
        config = MempalaceConfig(config_dir=str(tmp_path / "empty"))
        monkeypatch.setenv("MEMPALACE_FORCE_EMBEDDING", "false")
        assert config.force_embedding is False


class TestEmbeddingModelMismatchDetection:
    """palace.get_collection() detects model mismatches via collection metadata."""

    def test_new_collection_stamps_model(self, tmp_path, monkeypatch):
        """Creating a new collection stores the model name in metadata."""
        monkeypatch.delenv("MEMPALACE_EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("MEMPALACE_EMBEDDING_DEVICE", raising=False)
        palace_path = str(tmp_path / "palace")
        col = get_collection(palace_path)
        assert col.metadata.get("embedding_model") == "chromadb-default"

    def test_same_model_opens_fine(self, tmp_path, monkeypatch):
        """Opening with the same model succeeds without error."""
        monkeypatch.delenv("MEMPALACE_EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("MEMPALACE_EMBEDDING_DEVICE", raising=False)
        palace_path = str(tmp_path / "palace")
        get_collection(palace_path)

        # Reset embedding function cache to simulate new process
        import mempalace.config as cfg_mod

        cfg_mod._embedding_function = None
        cfg_mod._embedding_function_resolved = False

        col = get_collection(palace_path)
        assert col.metadata.get("embedding_model") == "chromadb-default"

    def test_mismatch_raises_error(self, tmp_path, monkeypatch):
        """Opening with a different model raises EmbeddingModelMismatchError."""
        monkeypatch.delenv("MEMPALACE_EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("MEMPALACE_EMBEDDING_DEVICE", raising=False)
        palace_path = str(tmp_path / "palace")

        col = get_collection(palace_path)
        assert col.metadata.get("embedding_model") == "chromadb-default"

        # Simulate a different model being stored
        col.modify(metadata={"embedding_model": "intfloat/multilingual-e5-base"})

        import mempalace.config as cfg_mod

        cfg_mod._embedding_function = None
        cfg_mod._embedding_function_resolved = False

        with pytest.raises(EmbeddingModelMismatchError) as exc_info:
            get_collection(palace_path)

        assert "intfloat/multilingual-e5-base" in str(exc_info.value)
        assert "chromadb-default" in str(exc_info.value)

    def test_mismatch_with_force_proceeds(self, tmp_path, monkeypatch):
        """Force flag bypasses the mismatch check and stamps new model."""
        monkeypatch.delenv("MEMPALACE_EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("MEMPALACE_EMBEDDING_DEVICE", raising=False)
        palace_path = str(tmp_path / "palace")

        col = get_collection(palace_path)
        col.modify(metadata={"embedding_model": "old-model"})

        import mempalace.config as cfg_mod

        cfg_mod._embedding_function = None
        cfg_mod._embedding_function_resolved = False

        col = get_collection(palace_path, force=True)
        assert col.metadata.get("embedding_model") == "chromadb-default"

    def test_legacy_palace_gets_stamped(self, tmp_path, monkeypatch):
        """A collection with no embedding_model metadata gets silently stamped."""
        monkeypatch.delenv("MEMPALACE_EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("MEMPALACE_EMBEDDING_DEVICE", raising=False)
        palace_path = str(tmp_path / "palace")

        # Create collection without metadata (simulating pre-detection palace)
        import chromadb

        os.makedirs(palace_path, exist_ok=True)
        client = chromadb.PersistentClient(path=palace_path)
        client.create_collection("mempalace_drawers")

        import mempalace.config as cfg_mod

        cfg_mod._embedding_function = None
        cfg_mod._embedding_function_resolved = False

        col = get_collection(palace_path)
        assert col.metadata.get("embedding_model") == "chromadb-default"
