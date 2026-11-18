"""Tests for publication: Kaggle metadata, upload gates, releases, cards."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.publishing.kaggle_publisher as kaggle_publisher
from src.publishing.kaggle_publisher import (
    KAGGLE_OWNER,
    MODEL_SLUG,
    build_model_metadata,
    build_variation_metadata,
    ensure_model_public,
    get_model_is_private,
    upload_variation,
)
from src.publishing.model_card import render_model_card
from src.publishing.release_manifest import (
    list_releases,
    load_release,
    write_release,
)
from tests.test_provenance import complete_provenance

ALLOWED_METADATA_KEYS = {
    "ownerSlug",
    "slug",
    "title",
    "subtitle",
    "isPrivate",
    "description",
    "licenseName",
    "licenses",
    "tags",
    "instances",
}


def test_model_metadata_only_supported_fields():
    metadata = build_model_metadata(
        "Qwen3.8-27B-akarui",
        "qlora-sft",
        "Competition-solving fine-tune of Qwen3.8-27B.",
        ["qwen", "pytorch", "qlora"],
    )
    assert set(metadata) <= ALLOWED_METADATA_KEYS
    assert "Qwen3.8-27B-akarui" in metadata["title"]
    assert metadata["tags"] == ["qwen", "pytorch", "qlora"]
    instances = metadata["instances"]
    assert instances
    assert instances[0]["framework"] == "pytorch"


def test_model_metadata_is_public_by_default():
    # Kaggle defaults a new model to private (docs: isPrivate "If not
    # specified, will be true"; the `models init` template writes
    # true). The staged metadata must therefore carry isPrivate False
    # explicitly, plus the ownerSlug/slug `kaggle models create`
    # requires — without them creation fails outright.
    metadata = build_model_metadata(
        "Qwen3.8-27B-akarui",
        "qlora-sft",
        "Competition-solving fine-tune of Qwen3.8-27B.",
    )
    assert metadata["isPrivate"] is False
    assert metadata["ownerSlug"] == KAGGLE_OWNER
    assert metadata["slug"] == MODEL_SLUG
    assert metadata["licenseName"] == "Apache 2.0"


class _FakeKaggleCli:
    """Stub for subprocess.run serving `kaggle models get/update`.

    Each `get` writes a model-metadata.json into the folder given
    with -p, with isPrivate taken from the next value in
    ``get_private`` (default False). Every command is recorded.
    """

    def __init__(self, get_private: list[bool] | None = None):
        self.get_private = list(get_private or [])
        self.commands: list[list[str]] = []

    def __call__(self, argv, **kwargs):
        self.commands.append(list(argv))
        if argv[1:3] == ["models", "get"]:
            folder = argv[argv.index("-p") + 1]
            private = self.get_private.pop(0) if self.get_private else False
            Path(folder).mkdir(parents=True, exist_ok=True)
            (Path(folder) / "model-metadata.json").write_text(
                json.dumps({"isPrivate": private})
            )
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if argv[1:3] == ["models", "update"]:
            return SimpleNamespace(returncode=0, stdout="updated", stderr="")
        raise AssertionError(f"unexpected kaggle command: {argv}")


def _patch_cli(monkeypatch, fake):
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/kaggle")
    monkeypatch.setattr("subprocess.run", fake)
    monkeypatch.delenv("KAGGLE_TOKEN", raising=False)


def test_get_model_is_private_reads_server_value(monkeypatch):
    fake = _FakeKaggleCli(get_private=[True])
    _patch_cli(monkeypatch, fake)
    assert get_model_is_private(f"{KAGGLE_OWNER}/{MODEL_SLUG}") is True
    assert fake.commands[0][:3] == ["kaggle", "models", "get"]


def test_get_model_is_private_public(monkeypatch):
    _patch_cli(monkeypatch, _FakeKaggleCli(get_private=[False]))
    assert get_model_is_private(f"{KAGGLE_OWNER}/{MODEL_SLUG}") is False


def test_get_model_is_private_refuses_failed_get(monkeypatch):
    def failing_run(argv, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="not found")

    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/kaggle")
    monkeypatch.setattr("subprocess.run", failing_run)
    monkeypatch.delenv("KAGGLE_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="cannot verify model visibility"):
        get_model_is_private(f"{KAGGLE_OWNER}/{MODEL_SLUG}")


def test_ensure_model_public_noop_when_already_public(monkeypatch):
    fake = _FakeKaggleCli(get_private=[False])
    _patch_cli(monkeypatch, fake)
    result = ensure_model_public(f"{KAGGLE_OWNER}/{MODEL_SLUG}")
    assert result == {
        "handle": f"{KAGGLE_OWNER}/{MODEL_SLUG}",
        "visibility": "public",
        "repaired": False,
    }
    assert not [c for c in fake.commands if c[2] == "update"]


def test_ensure_model_public_repairs_private_model(monkeypatch):
    fake = _FakeKaggleCli(get_private=[True, False])
    _patch_cli(monkeypatch, fake)
    result = ensure_model_public(f"{KAGGLE_OWNER}/{MODEL_SLUG}")
    assert result["visibility"] == "public"
    assert result["repaired"] is True
    update_calls = [c for c in fake.commands if c[1:3] == ["models", "update"]]
    assert len(update_calls) == 1
    assert update_calls[0][:3] == ["kaggle", "models", "update"]


def test_ensure_model_public_fails_loudly_when_still_private(monkeypatch):
    fake = _FakeKaggleCli(get_private=[True, True])
    _patch_cli(monkeypatch, fake)
    with pytest.raises(RuntimeError, match="still private"):
        ensure_model_public(f"{KAGGLE_OWNER}/{MODEL_SLUG}")


def test_ensure_model_public_fails_without_cli_or_library(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _name: None)
    monkeypatch.setattr(
        kaggle_publisher,
        "_kaggle_api",
        lambda: (_ for _ in ()).throw(
            RuntimeError("neither the kaggle CLI nor the kaggle library is available")
        ),
    )
    with pytest.raises(RuntimeError, match="neither the kaggle CLI"):
        ensure_model_public(f"{KAGGLE_OWNER}/{MODEL_SLUG}")


def test_upload_variation_records_public_visibility(monkeypatch, tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    monkeypatch.setenv("KAGGLE_TOKEN", "owner:key")
    monkeypatch.setattr(
        kaggle_publisher,
        "_upload_via_cli",
        lambda handle, variation, directory, env, **kwargs: {
            "method": "kaggle-cli",
            "model": handle,
            "variation": variation,
            "directory": str(directory),
        },
    )
    monkeypatch.setattr(
        kaggle_publisher,
        "ensure_model_public",
        lambda handle: {
            "handle": handle,
            "visibility": "public",
            "repaired": True,
        },
    )
    result = upload_variation("qlora-sft", model_dir, complete_provenance())
    assert result["method"] == "kaggle-cli"
    assert result["visibility"]["visibility"] == "public"
    assert result["visibility"]["repaired"] is True


def test_publish_dry_run_stages_public_metadata(tmp_path: Path):
    from publish_kaggle import main as publish_main
    from tests.test_lifecycle import _evaluation, _fake_adapter, _reference

    root = Path(__file__).resolve().parent.parent
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(json.dumps(complete_provenance()))
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(json.dumps(_evaluation(0.62)))
    champion_path = tmp_path / "champion.json"
    champion_path.write_text(json.dumps(_reference(0.55)))
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(_reference(0.55)))
    adapter = _fake_adapter(
        tmp_path, {"gpus": [{"name": "L4", "total_memory_gb": 24.0}]}
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("# Qwen3.8-27B-akarui\n")
    staging = tmp_path / "staging"
    rc = publish_main(
        [
            "--provenance", str(provenance_path),
            "--evaluation", str(evaluation_path),
            "--champion", str(champion_path),
            "--baseline", str(baseline_path),
            "--config", str(root / "configs" / "evaluation.yaml"),
            "--adapter", str(adapter),
            "--model-card", str(model_card),
            "--release-version", "1",
            "--model-version", "qlora-sft-v1",
            "--staging-dir", str(staging),
            "--dry-run",
        ]
    )
    assert rc == 0
    metadata = json.loads(
        (staging / "qlora-sft-v1" / "model-metadata.json").read_text()
    )
    assert metadata["isPrivate"] is False
    assert metadata["ownerSlug"] == KAGGLE_OWNER
    assert metadata["slug"] == MODEL_SLUG


def test_variation_metadata_rejects_unknown():
    with pytest.raises(ValueError, match="unknown variation"):
        build_variation_metadata("not-a-variation")
    variation = build_variation_metadata("qlora-sft")
    assert variation["variationType"] == "qlora-sft"
    assert variation["fineTunable"] is True


def test_upload_refuses_incomplete_provenance(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with pytest.raises(ValueError, match="incomplete provenance"):
        upload_variation("qlora-sft", model_dir, {"model": "Qwen3.8-27B-akarui"})


def test_release_roundtrip_and_append_only(tmp_path: Path):
    provenance = complete_provenance()
    path = write_release(
        "0.1.0",
        provenance,
        {"normalized_score": None},
        "Initial infrastructure; no model released.",
        releases_dir=tmp_path,
    )
    assert path.exists()
    loaded = load_release("0.1.0", releases_dir=tmp_path)
    assert loaded["version"] == "0.1.0"
    assert loaded["provenance"]["base_revision"] == (
        "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
    )
    assert list_releases(releases_dir=tmp_path) == ["0.1.0"]
    with pytest.raises(FileExistsError):
        write_release(
            "0.1.0",
            provenance,
            {},
            "duplicate",
            releases_dir=tmp_path,
        )


def test_release_rejects_incomplete_provenance(tmp_path: Path):
    with pytest.raises(ValueError, match="cannot write release"):
        write_release(
            "0.2.0",
            {"model": "Qwen3.8-27B-akarui"},
            {},
            "bad provenance",
            releases_dir=tmp_path,
        )


def test_model_card_marks_unmeasured():
    card = render_model_card({"model_name": "Qwen3.8-27B-akarui"})
    assert "Qwen3.8-27B-akarui" in card
    assert "Not yet measured" in card


# --- version-create command (kaggle 1.7.4) -------------------------------
# ``kaggle models version create`` does not exist: ``models`` only
# offers get/list/init/create/delete/update (kaggle 1.7.4
# ``kaggle/cli.py`` ``parse_models``). A new version is created with
# ``kaggle models instances versions create
# <owner>/<model>/<framework>/<variation> -p <dir> -n <notes>``
# (``parse_model_instance_versions``; the handle is
# ``[owner]/[model-name]/[framework]/[instance-slug]`` per
# ``KaggleApi.model_instance_version_create``).
#
# The instance slug is the BARE variation (``qlora-sft``), never a
# release label (``qlora-sft-v1``): Kaggle treats the 4th handle
# component as the instance slug, so a label there would create a
# new instance per release instead of a new version of one
# instance. Before the version create, the instance is ensured to
# exist (``kaggle models instances get``; ``kaggle models instances
# create -p <dir>`` when missing), so the first publish works.

INSTANCE_HANDLE = f"{KAGGLE_OWNER}/{MODEL_SLUG}/pytorch/qlora-sft"
VERSION_LABEL = "qlora-sft-v1"


def _fake_run_ok(monkeypatch, commands: list):
    def fake_run(argv, **kwargs):
        commands.append(list(argv))
        return SimpleNamespace(returncode=0, stdout="created", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)


def _fake_run_instance_missing(monkeypatch, commands: list, created_metadata: dict):
    """Fake where `instances get` reports the instance missing.

    The `instances create` call snapshots the metadata file it was
    pointed at (the temp dir is deleted when the call returns).
    """

    def fake_run(argv, **kwargs):
        commands.append(list(argv))
        if argv[1:4] == ["models", "instances", "get"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="not found")
        if argv[1:4] == ["models", "instances", "create"]:
            folder = argv[argv.index("-p") + 1]
            created_metadata.update(
                json.loads(
                    (Path(folder) / "model-instance-metadata.json").read_text()
                )
            )
            return SimpleNamespace(returncode=0, stdout="created", stderr="")
        return SimpleNamespace(returncode=0, stdout="created", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)


def test_bare_variation_strips_release_label():
    assert kaggle_publisher.bare_variation("qlora-sft-v1") == "qlora-sft"
    assert kaggle_publisher.bare_variation("qlora-sft") == "qlora-sft"
    assert kaggle_publisher.bare_variation("merged-bf16-v3") == "merged-bf16"
    with pytest.raises(ValueError, match="unknown variation"):
        kaggle_publisher.bare_variation("not-a-variation-v1")


def test_build_model_instance_metadata():
    metadata = kaggle_publisher.build_model_instance_metadata(
        "qlora-sft-v1", overview="o", usage="u"
    )
    assert metadata["instanceSlug"] == "qlora-sft"
    assert metadata["ownerSlug"] == KAGGLE_OWNER
    assert metadata["modelSlug"] == MODEL_SLUG
    assert metadata["framework"] == "pytorch"
    assert metadata["licenseName"] == "Apache 2.0"
    assert metadata["fineTunable"] is True
    assert metadata["overview"] == "o"
    assert metadata["usage"] == "u"


def test_ensure_instance_exists_noop_when_present(monkeypatch):
    commands: list = []
    _fake_run_ok(monkeypatch, commands)
    result = kaggle_publisher.ensure_instance_exists(INSTANCE_HANDLE, "qlora-sft")
    assert result == {"instance": INSTANCE_HANDLE, "created": False}
    assert commands == [["kaggle", "models", "instances", "get", INSTANCE_HANDLE]]


def test_ensure_instance_exists_creates_missing_instance(monkeypatch):
    commands: list = []
    created_metadata: dict = {}
    _fake_run_instance_missing(monkeypatch, commands, created_metadata)
    result = kaggle_publisher.ensure_instance_exists(
        INSTANCE_HANDLE, "qlora-sft-v1", overview="o", usage="u"
    )
    assert result == {"instance": INSTANCE_HANDLE, "created": True}
    assert commands[0] == ["kaggle", "models", "instances", "get", INSTANCE_HANDLE]
    assert commands[1][:4] == ["kaggle", "models", "instances", "create"]
    assert commands[1][4] == "-p"
    assert created_metadata["instanceSlug"] == "qlora-sft"
    assert created_metadata["framework"] == "pytorch"
    assert created_metadata["licenseName"] == "Apache 2.0"


def test_ensure_instance_exists_create_failure_raises(monkeypatch):
    def failing_run(argv, **kwargs):
        if argv[1:4] == ["models", "instances", "get"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="not found")
        return SimpleNamespace(returncode=1, stdout="", stderr="create failed")

    monkeypatch.setattr("subprocess.run", failing_run)
    with pytest.raises(RuntimeError, match="could not create model instance"):
        kaggle_publisher.ensure_instance_exists(INSTANCE_HANDLE, "qlora-sft")


def test_upload_variation_uses_instances_versions_create(monkeypatch, tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    monkeypatch.setenv("KAGGLE_TOKEN", "owner:key")
    monkeypatch.setattr(
        kaggle_publisher,
        "ensure_model_public",
        lambda handle: {"handle": handle, "visibility": "public", "repaired": False},
    )
    commands: list = []
    _fake_run_ok(monkeypatch, commands)
    result = upload_variation(
        "qlora-sft-v1",
        model_dir,
        complete_provenance(),
        version_notes="Release 1",
    )
    assert result["method"] == "kaggle-cli"
    # The release label is a version, not an instance: the handle
    # carries the BARE variation, and the instance is checked first.
    assert result["variation"] == "qlora-sft"
    assert commands == [
        ["kaggle", "models", "instances", "get", INSTANCE_HANDLE],
        [
            "kaggle",
            "models",
            "instances",
            "versions",
            "create",
            INSTANCE_HANDLE,
            "-p",
            str(model_dir),
            "-n",
            "Release 1",
        ],
    ]


def test_upload_variation_omits_notes_flag_when_no_notes(monkeypatch, tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    monkeypatch.setenv("KAGGLE_TOKEN", "owner:key")
    monkeypatch.setattr(
        kaggle_publisher,
        "ensure_model_public",
        lambda handle: {"handle": handle, "visibility": "public", "repaired": False},
    )
    commands: list = []
    _fake_run_ok(monkeypatch, commands)
    upload_variation("qlora-sft", model_dir, complete_provenance())
    assert commands == [
        ["kaggle", "models", "instances", "get", f"{KAGGLE_OWNER}/{MODEL_SLUG}/pytorch/qlora-sft"],
        [
            "kaggle",
            "models",
            "instances",
            "versions",
            "create",
            f"{KAGGLE_OWNER}/{MODEL_SLUG}/pytorch/qlora-sft",
            "-p",
            str(model_dir),
        ],
    ]


def test_upload_variation_first_publish_creates_missing_instance(
    monkeypatch, tmp_path: Path
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    monkeypatch.setenv("KAGGLE_TOKEN", "owner:key")
    monkeypatch.setattr(
        kaggle_publisher,
        "ensure_model_public",
        lambda handle: {"handle": handle, "visibility": "public", "repaired": False},
    )
    commands: list = []
    created_metadata: dict = {}
    _fake_run_instance_missing(monkeypatch, commands, created_metadata)
    result = upload_variation(
        "qlora-sft-v1", model_dir, complete_provenance(), version_notes="Release 1"
    )
    assert result["method"] == "kaggle-cli"
    assert result["instance_created"] is True
    assert created_metadata["instanceSlug"] == "qlora-sft"
    assert commands[0] == ["kaggle", "models", "instances", "get", INSTANCE_HANDLE]
    assert commands[1][:4] == ["kaggle", "models", "instances", "create"]
    assert commands[2] == [
        "kaggle",
        "models",
        "instances",
        "versions",
        "create",
        INSTANCE_HANDLE,
        "-p",
        str(model_dir),
        "-n",
        "Release 1",
    ]


def test_upload_variation_existing_instance_no_spurious_create(
    monkeypatch, tmp_path: Path
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    monkeypatch.setenv("KAGGLE_TOKEN", "owner:key")
    monkeypatch.setattr(
        kaggle_publisher,
        "ensure_model_public",
        lambda handle: {"handle": handle, "visibility": "public", "repaired": False},
    )
    commands: list = []
    _fake_run_ok(monkeypatch, commands)
    result = upload_variation("qlora-sft-v1", model_dir, complete_provenance())
    assert result["instance_created"] is False
    assert not [c for c in commands if c[1:4] == ["models", "instances", "create"]]


def test_upload_via_cli_version_create_failure_raises(monkeypatch, tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    def failing_version_run(argv, **kwargs):
        if argv[1:4] == ["models", "instances", "get"]:
            return SimpleNamespace(returncode=0, stdout="exists", stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="version failed")

    monkeypatch.setattr("subprocess.run", failing_version_run)
    with pytest.raises(RuntimeError, match="kaggle CLI upload failed"):
        kaggle_publisher._upload_via_cli(
            f"{KAGGLE_OWNER}/{MODEL_SLUG}", "qlora-sft-v1", model_dir, {}
        )


def test_upload_via_cli_failure_raises(monkeypatch, tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    def failing_run(argv, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="instance not found")

    monkeypatch.setattr("subprocess.run", failing_run)
    # The instance check fails, so the ensure step tries to create
    # the instance and that fails too.
    with pytest.raises(RuntimeError, match="could not create model instance"):
        kaggle_publisher._upload_via_cli(
            f"{KAGGLE_OWNER}/{MODEL_SLUG}", "qlora-sft-v1", model_dir, {}
        )


def test_upload_via_cli_absent_returns_none(monkeypatch, tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    def missing_binary(argv, **kwargs):
        raise FileNotFoundError("kaggle")

    monkeypatch.setattr("subprocess.run", missing_binary)
    assert (
        kaggle_publisher._upload_via_cli(
            f"{KAGGLE_OWNER}/{MODEL_SLUG}", "qlora-sft-v1", model_dir, {}
        )
        is None
    )


def test_upload_variation_falls_back_to_kagglehub(monkeypatch, tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    monkeypatch.setenv("KAGGLE_TOKEN", "owner:key")
    monkeypatch.setattr(kaggle_publisher, "_upload_via_cli", lambda *a, **k: None)
    monkeypatch.setattr(
        kaggle_publisher,
        "_upload_via_kagglehub",
        lambda handle, variation, directory, **_kwargs: {
            "method": "kagglehub",
            "model": handle,
            "variation": variation,
            "directory": str(directory),
        },
    )
    monkeypatch.setattr(
        kaggle_publisher,
        "ensure_model_public",
        lambda handle: {"handle": handle, "visibility": "public", "repaired": False},
    )
    result = upload_variation("qlora-sft-v1", model_dir, complete_provenance())
    assert result["method"] == "kagglehub"
    assert result["visibility"]["visibility"] == "public"


def test_upload_via_kagglehub_uses_four_part_handle(monkeypatch, tmp_path: Path):
    # kagglehub 0.3.12 `parse_model_handle` requires 4 parts
    # (owner/model/framework/variation) and raises ValueError on a
    # 2-part handle — the old code passed `owner/model` and the
    # fallback could never run.
    import sys

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    calls: list = []

    class _FakeKagglehub:
        @staticmethod
        def model_upload(handle, local_dir, version_notes=""):
            calls.append((handle, local_dir, version_notes))

    monkeypatch.setitem(sys.modules, "kagglehub", _FakeKagglehub())
    result = kaggle_publisher._upload_via_kagglehub(
        f"{KAGGLE_OWNER}/{MODEL_SLUG}", "qlora-sft-v1", model_dir,
        version_notes="Release 1",
    )
    assert calls == [
        (f"{KAGGLE_OWNER}/{MODEL_SLUG}/pytorch/qlora-sft", str(model_dir), "Release 1")
    ]
    assert result["method"] == "kagglehub"
    assert result["variation"] == "qlora-sft"


@pytest.mark.skipif(
    not Path("/tmp/kaggle-src/kagglehub-0.3.12/kagglehub/handle.py").exists(),
    reason="pinned kagglehub wheel source only present locally under /tmp/kaggle-src",
)
def test_upload_via_kagglehub_rejects_two_part_handle(monkeypatch, tmp_path: Path):
    # Direct evidence for the pinned behaviour: the real
    # kagglehub 0.3.12 parser rejects a 2-part handle. Loaded from
    # the pinned wheel source by file path so the test never touches
    # sys.path or an installed kagglehub.
    import importlib.util
    import sys
    import types

    # handle.py does `from kagglehub.config import
    # get_kaggle_api_endpoint`; stub just that import so the pinned
    # parser loads standalone.
    pkg = types.ModuleType("kagglehub")
    pkg.__path__ = []
    config = types.ModuleType("kagglehub.config")
    config.get_kaggle_api_endpoint = lambda: ""
    monkeypatch.setitem(sys.modules, "kagglehub", pkg)
    monkeypatch.setitem(sys.modules, "kagglehub.config", config)
    spec = importlib.util.spec_from_file_location(
        "pinned_kagglehub_handle",
        "/tmp/kaggle-src/kagglehub-0.3.12/kagglehub/handle.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="Invalid model handle"):
        module.parse_model_handle(f"{KAGGLE_OWNER}/{MODEL_SLUG}")
    parsed = module.parse_model_handle(
        f"{KAGGLE_OWNER}/{MODEL_SLUG}/pytorch/qlora-sft"
    )
    assert (parsed.owner, parsed.model, parsed.framework, parsed.variation) == (
        KAGGLE_OWNER, MODEL_SLUG, "pytorch", "qlora-sft",
    )


def test_publish_script_version_create_path(monkeypatch, tmp_path: Path):
    import publish_kaggle

    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/kaggle")
    monkeypatch.setattr(publish_kaggle, "_model_exists", lambda: True)
    commands: list = []
    _fake_run_ok(monkeypatch, commands)
    publish_kaggle.upload_with_cli(staging, "1", "qlora-sft-v1")
    assert commands == [
        ["kaggle", "models", "instances", "get", INSTANCE_HANDLE],
        [
            "kaggle",
            "models",
            "instances",
            "versions",
            "create",
            INSTANCE_HANDLE,
            "-p",
            str(staging),
            "-n",
            "Release 1",
        ],
    ]


def test_publish_script_create_new_model_path(monkeypatch, tmp_path: Path):
    import publish_kaggle

    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/kaggle")
    monkeypatch.setattr(publish_kaggle, "_model_exists", lambda: False)
    commands: list = []
    _fake_run_ok(monkeypatch, commands)
    publish_kaggle.upload_with_cli(staging, "1", "qlora-sft-v1")
    # `kaggle models create` makes only the model shell — no
    # instance, no files — so the first publish continues with the
    # instance ensure and the first version create.
    assert commands == [
        ["kaggle", "models", "create", "-p", str(staging)],
        ["kaggle", "models", "instances", "get", INSTANCE_HANDLE],
        [
            "kaggle",
            "models",
            "instances",
            "versions",
            "create",
            INSTANCE_HANDLE,
            "-p",
            str(staging),
            "-n",
            "Release 1",
        ],
    ]


def test_publish_script_first_publish_creates_missing_instance(
    monkeypatch, tmp_path: Path
):
    import publish_kaggle

    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/kaggle")
    monkeypatch.setattr(publish_kaggle, "_model_exists", lambda: True)
    commands: list = []
    created_metadata: dict = {}
    _fake_run_instance_missing(monkeypatch, commands, created_metadata)
    publish_kaggle.upload_with_cli(staging, "1", "qlora-sft-v1")
    assert commands[0] == ["kaggle", "models", "instances", "get", INSTANCE_HANDLE]
    assert commands[1][:4] == ["kaggle", "models", "instances", "create"]
    assert created_metadata["instanceSlug"] == "qlora-sft"
    assert commands[2] == [
        "kaggle",
        "models",
        "instances",
        "versions",
        "create",
        INSTANCE_HANDLE,
        "-p",
        str(staging),
        "-n",
        "Release 1",
    ]


def test_publish_script_kagglehub_upload_uses_four_part_handle(
    monkeypatch, tmp_path: Path
):
    import sys

    import publish_kaggle

    staging = tmp_path / "staging"
    staging.mkdir()
    calls: list = []

    class _FakeKagglehub:
        @staticmethod
        def model_upload(handle, local_dir, version_notes=""):
            calls.append((handle, local_dir, version_notes))

    monkeypatch.setitem(sys.modules, "kagglehub", _FakeKagglehub())
    publish_kaggle.upload_with_kagglehub(staging, "1", "qlora-sft-v1")
    assert calls == [
        (f"{KAGGLE_OWNER}/{MODEL_SLUG}/pytorch/qlora-sft", str(staging), "Release 1")
    ]


def test_publish_script_cli_failure_raises(monkeypatch, tmp_path: Path):
    import publish_kaggle

    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/kaggle")
    monkeypatch.setattr(publish_kaggle, "_model_exists", lambda: True)

    def failing_run(argv, **kwargs):
        if argv[1:4] == ["models", "instances", "get"]:
            return SimpleNamespace(returncode=0, stdout="exists", stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="upload failed")

    monkeypatch.setattr("subprocess.run", failing_run)
    with pytest.raises(RuntimeError, match="kaggle upload failed"):
        publish_kaggle.upload_with_cli(staging, "1", "qlora-sft-v1")
