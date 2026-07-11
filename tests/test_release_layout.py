from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MARKETPLACE_LOCALES = {
    "en_US",
    "zh_Hans",
    "zh_Hant",
    "ja_JP",
    "th_TH",
    "vi_VN",
    "es_ES",
    "ru_RU",
}


def test_local_agent_has_publishable_marketplace_layout() -> None:
    manifest = yaml.safe_load((ROOT / "manifest.yaml").read_text(encoding="utf-8"))
    runner = yaml.safe_load(
        (ROOT / "components" / "agent_runner" / "default.yaml").read_text(encoding="utf-8")
    )
    metadata = manifest["metadata"]
    runner_id = f"plugin:{metadata['author']}/{metadata['name']}/{runner['metadata']['name']}"
    config_fields = [
        item["name"] for item in manifest["spec"].get("config", []) + runner["spec"].get("config", [])
    ]

    assert manifest["apiVersion"] == "v1"
    assert "version" not in manifest["spec"]
    assert metadata["author"] == "langbot-team"
    assert metadata["name"] == "LocalAgent"
    assert re.fullmatch(r"[A-Z][A-Za-z0-9]*", metadata["name"])
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?", metadata["version"])
    assert metadata["repository"] == "https://github.com/langbot-app/langbot-local-agent"
    assert set(metadata["label"]) == MARKETPLACE_LOCALES
    assert set(metadata["description"]) == MARKETPLACE_LOCALES
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert len(root_readme.encode("utf-8")) >= 2_000
    assert any("\u4e00" <= char <= "\u9fff" for char in root_readme)
    assert all(field in root_readme for field in config_fields)
    assert runner_id in root_readme

    expected_readmes = {f"README_{locale}.md" for locale in MARKETPLACE_LOCALES}
    assert {path.name for path in (ROOT / "readme").glob("README_*.md")} == expected_readmes
    for readme_name in expected_readmes:
        localized_readme = (ROOT / "readme" / readme_name).read_text(encoding="utf-8")
        assert len(localized_readme.encode("utf-8")) >= 1_000
        assert all(field in localized_readme for field in config_fields)
        assert runner_id in localized_readme
