import zipfile
from pathlib import Path

from sytra_runner.colibri_bridge import colibri_install_hint
from sytra_runner.colibri_provision import (
    RELEASE,
    colibri_provision_allowed,
    maybe_provision_colibri,
    provision_colibri,
)


def _write_fake_windows_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("coli", "#!/usr/bin/env python3\nprint('coli')\n")
        archive.writestr("colibri.exe", b"MZ")
        archive.writestr("coli_cuda.dll", b"MZ")


def test_provision_colibri_extracts_launcher(tmp_path, monkeypatch):
    archive = tmp_path / "colibri-v1.6.2-windows-x86_64.zip"
    _write_fake_windows_zip(archive)

    def fake_download(url: str, dest: Path) -> None:
        dest.write_bytes(archive.read_bytes())

    monkeypatch.setattr("sytra_runner.colibri_provision.download_file", fake_download)
    monkeypatch.setattr(
        "sytra_runner.colibri_provision.sha256_file",
        lambda path: "12d4cb059a8d3a4f7700eaf16a2cd605de78099e48ddcd756e6b67b1043a1596",
    )

    launcher = provision_colibri(tmp_path, platform="win32")
    dest = tmp_path / ".tools" / "colibri"
    assert (dest / "coli").is_file()
    assert (dest / "colibri.exe").is_file()
    assert (dest / ".sytra-colibri-version").read_text(encoding="utf-8").strip() == RELEASE
    assert launcher[-1] == str((dest / "coli").resolve())
    assert "python" in Path(launcher[0]).name.lower()

    again = provision_colibri(tmp_path, platform="win32")
    assert again == launcher


def test_maybe_provision_skipped_under_pytest():
    assert colibri_provision_allowed() is False
    assert (
        maybe_provision_colibri(
            None,
            requested_backend="colibri",
            colibri_family="glm_moe",
        )
        is None
    )


def test_colibri_install_hint_points_at_provision_script(tmp_path):
    hint = colibri_install_hint(tmp_path)
    assert "provision_colibri.py" in hint
    assert str(tmp_path.resolve()) in hint
    assert "SYTRA_SKIP_COLIBRI_PROVISION" in hint
