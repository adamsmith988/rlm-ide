"""Unit tests for map geo/layers helpers (no live DB)."""

from __future__ import annotations

import os
import sys
import tempfile
import asyncio
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.storage.src.map_geo import (
    _extract_location,
    _haversine_m,
    _in_bounds,
    _tile_bounds,
    _encode_mvt_layer,
    MapGeoService,
)
from services.storage.src.map_layers import MapLayersService


def test_extract_location_from_location_dict():
    loc = _extract_location({"location": {"latitude": 35.7, "longitude": 51.4}})
    assert loc is not None
    assert loc["latitude"] == 35.7
    assert loc["longitude"] == 51.4


def test_extract_location_from_gps_array():
    loc = _extract_location({"gps": [35.7, 51.4]})
    assert loc is not None
    assert loc["source"] == "embed"


def test_haversine_zero_distance():
    assert _haversine_m(35.0, 51.0, 35.0, 51.0) == 0.0


def test_in_bounds():
    b = {"north": 36.0, "south": 35.0, "east": 52.0, "west": 51.0}
    assert _in_bounds(35.5, 51.5, b)
    assert not _in_bounds(34.0, 51.5, b)


def test_pano_row_to_dict_requires_valid_coords():
    svc = MapLayersService(storage=object())
    assert (
        svc._pano_row_to_dict(
            {"filename": "a.jpg", "latitude": None, "longitude": None, "headings": "[]"}
        )
        is None
    )
    d = svc._pano_row_to_dict(
        {
            "filename": "a.jpg",
            "latitude": 35.7,
            "longitude": 51.4,
            "headings": "[]",
            "relative_path": "a.jpg",
            "metadata": "{}",
        }
    )
    assert d is not None
    assert d["lat"] == 35.7
    assert d["lon"] == 51.4


def test_parse_lat_lon():
    from services.storage.src.map_layers import _parse_lat_lon

    assert _parse_lat_lon(35.7, 51.4) == (35.7, 51.4)
    assert _parse_lat_lon(None, 51.4) is None
    assert _parse_lat_lon(91.0, 51.4) is None


def test_streetview_kind_mismatch_when_pmtiles(tmp_path, monkeypatch):
    monkeypatch.setenv("MAP_LAYERS_ROOT", str(tmp_path))
    folder = tmp_path / "layer-vector" / "tiles"
    folder.mkdir(parents=True)
    (folder / "iran.pmtiles").write_bytes(b"PMTiles")
    svc = MapLayersService(storage=object())
    layer = {
        "id": "test-id",
        "layer_kind": "streetview_folder",
        "storage_root": "layer-vector/tiles",
        "source_config": {},
    }
    enriched = svc._enrich_catalog_row(layer)
    assert enriched.get("import_status") == "kind_mismatch"
    assert enriched.get("detected_kind") == "vector_pmtiles"


def test_detect_streetview_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("MAP_LAYERS_ROOT", str(tmp_path))
    sv = tmp_path / "streetview" / "cam1"
    sv.mkdir(parents=True)
    (sv / "pano.jpg").write_bytes(b"fake")
    svc = MapLayersService(storage=object())
    out = svc.detect_path("streetview/cam1")
    assert out.get("ok") is True
    assert out.get("layer_kind") == "streetview_folder"


def test_detect_pmtiles_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MAP_LAYERS_ROOT", str(tmp_path))
    f = tmp_path / "raster" / "tehran.pmtiles"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"PMTiles")
    svc = MapLayersService(storage=object())
    out = svc.detect_path("raster/tehran.pmtiles")
    assert out.get("ok") is True
    assert out.get("layer_kind") == "raster_pmtiles"


def test_detect_pmtiles_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MAP_LAYERS_ROOT", str(tmp_path))
    f = tmp_path / "vector" / "region.pmtiles"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"PMTiles")
    svc = MapLayersService(storage=object())
    layer = {"layer_kind": "vector_pmtiles", "storage_root": "vector/region.pmtiles"}
    found = svc.pmtiles_file_for_layer(layer)
    assert found is not None
    assert found.name == "region.pmtiles"


def test_pmtiles_file_symlink_outside_root(tmp_path, monkeypatch):
    """Symlinked tile dirs (layer-vector/tiles → external mount) must resolve PMTiles."""
    external = tmp_path / "external"
    external.mkdir()
    pm = external / "region.pmtiles"
    pm.write_bytes(b"PMTiles")
    link_parent = tmp_path / "layer-vector"
    link_parent.mkdir()
    (link_parent / "tiles").symlink_to(external, target_is_directory=True)
    monkeypatch.setenv("MAP_LAYERS_ROOT", str(tmp_path))
    svc = MapLayersService(storage=object())
    layer = {"layer_kind": "vector_pmtiles", "storage_root": "layer-vector/tiles"}
    found = svc.pmtiles_file_for_layer(layer)
    assert found is not None
    assert found.name == "region.pmtiles"
    out = svc.detect_path("layer-vector/tiles")
    assert out.get("ok") is True
    assert out.get("data_available") is True


def test_companion_pmtiles_files(tmp_path, monkeypatch):
    tiles = tmp_path / "layer-vector" / "tiles"
    tiles.mkdir(parents=True)
    (tiles / "iran.pmtiles").write_bytes(b"PMTiles")
    (tiles / "world-overview.pmtiles").write_bytes(b"PMTiles")
    (tiles / "world-full-z14.pmtiles").write_bytes(b"PMTiles")
    monkeypatch.setenv("MAP_LAYERS_ROOT", str(tmp_path))
    svc = MapLayersService(storage=object())
    layer = {"layer_kind": "vector_pmtiles", "storage_root": "layer-vector/tiles", "source_config": {"pmtiles_file": "iran.pmtiles"}}
    companions = svc._companion_pmtiles_files(svc.pmtiles_file_for_layer(layer).parent)
    assert companions["world_overview_file"] == "world-overview.pmtiles"
    assert companions["world_full_file"] == "world-full-z14.pmtiles"
    assert svc.pmtiles_file_for_layer(layer, filename="world-overview.pmtiles").name == "world-overview.pmtiles"


def test_read_file_range_partial():
    from services.storage.src.map_api import _parse_byte_range, _read_file_range

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        payload = b"0123456789abcdef"
        tmp.write(payload)
        path = Path(tmp.name)
    try:
        assert _parse_byte_range("bytes=0-3", len(payload)) == (0, 3)
        assert _read_file_range(path, 0, 4) == b"0123"
        assert _read_file_range(path, 10, 6) == b"abcdef"
    finally:
        path.unlink(missing_ok=True)


def test_detect_sas_sqlite_cache(tmp_path, monkeypatch):
    import sqlite3

    from src.sas_sqlite_tiles import sas_sqlite_zoom_range

    monkeypatch.setenv("MAP_LAYERS_ROOT", str(tmp_path))
    db = tmp_path / "sas-sat" / "z10" / "0" / "0" / "0.0.sqlitedb"
    db.parent.mkdir(parents=True)
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE t (x INTEGER NOT NULL, y INTEGER NOT NULL, v INTEGER DEFAULT 0 NOT NULL, "
        "c TEXT, s INTEGER DEFAULT 0 NOT NULL, h INTEGER DEFAULT 0 NOT NULL, d INTEGER NOT NULL, "
        "b BLOB, constraint PK_TB primary key (x,y,v))"
    )
    conn.execute("INSERT INTO t (x, y, v, d, b) VALUES (0, 5, 0, 0, ?)", (b"\x89PNG\x00",))
    conn.commit()
    conn.close()

    db2 = tmp_path / "sas-sat" / "z11" / "0" / "0" / "1.1.sqlitedb"
    db2.parent.mkdir(parents=True)
    conn2 = sqlite3.connect(db2)
    conn2.execute(
        "CREATE TABLE t (x INTEGER NOT NULL, y INTEGER NOT NULL, v INTEGER DEFAULT 0 NOT NULL, "
        "c TEXT, s INTEGER DEFAULT 0 NOT NULL, h INTEGER DEFAULT 0 NOT NULL, d INTEGER NOT NULL, "
        "b BLOB, constraint PK_TB primary key (x,y,v))"
    )
    conn2.execute("INSERT INTO t (x, y, v, d, b) VALUES (256, 312, 0, 0, ?)", (b"\xff\xd8\xffJPEG",))
    conn2.commit()
    conn2.close()

    svc = MapLayersService(storage=object())
    out = svc.detect_path("sas-sat")
    assert out.get("ok") is True
    assert out.get("layer_kind") == "raster_sas_sqlite"

    layer = {"layer_kind": "raster_sas_sqlite", "storage_root": "sas-sat"}
    got = svc.tile_bytes_for_layer(layer, z=9, x=0, y=5)
    assert got is not None
    assert got[0] == b"\x89PNG\x00"
    got2 = svc.tile_bytes_for_layer(layer, z=10, x=256, y=312)
    assert got2 is not None
    assert got2[0] == b"\xff\xd8\xffJPEG"
    assert sas_sqlite_zoom_range(tmp_path / "sas-sat") == (9, 10)


def test_browse_layers_root(tmp_path, monkeypatch):
    monkeypatch.setenv("MAP_LAYERS_ROOT", str(tmp_path))
    (tmp_path / "sas-both").mkdir()
    (tmp_path / "sas-both" / "z10").mkdir()
    (tmp_path / "streetview" / "cam1").mkdir(parents=True)
    svc = MapLayersService(storage=object())
    root_listing = svc._browse_layers_root()
    assert root_listing.get("path") == ""
    names = {i["name"] for i in root_listing.get("items") or []}
    assert "sas-both" in names
    assert "streetview" in names
    sas_row = next(i for i in root_listing["items"] if i["name"] == "sas-both")
    assert sas_row.get("sqlitedb_count") == 1

    nested = svc._browse_layers_root("streetview")
    assert nested.get("path") == "streetview"
    assert nested.get("parent") == ""
    nested_names = {i["name"] for i in nested.get("items") or []}
    assert "cam1" in nested_names


def test_browse_layers_root_symlink_outside_root(tmp_path, monkeypatch):
    """Symlinked catalog folders (sas-both → external mount) must remain browsable."""
    external = tmp_path / "external" / "sat"
    external.mkdir(parents=True)
    (external / "z10").mkdir()
    link = tmp_path / "sas-both"
    link.symlink_to(external, target_is_directory=True)
    monkeypatch.setenv("MAP_LAYERS_ROOT", str(tmp_path))
    svc = MapLayersService(storage=object())
    nested = svc._browse_layers_root("sas-both")
    assert nested.get("path") == "sas-both"
    assert nested.get("parent") == ""
    names = {i["name"] for i in nested.get("items") or []}
    assert "z10" in names


def test_tile_bounds_z0():
    b = _tile_bounds(0, 0, 0)
    assert b["west"] == -180.0
    assert b["east"] == 180.0
    assert b["north"] > 0
    assert b["south"] < 0


def test_encode_mvt_layer_nonempty():
    blob = _encode_mvt_layer(
        "artifact_geo",
        [{"x": 100, "y": 200, "artifact_id": "a1", "count": 1, "cluster": False}],
    )
    assert isinstance(blob, (bytes, bytearray))
    assert len(blob) > 10
    assert b"artifact_geo" in blob


def test_geo_fingerprint_has_no_point_bodies():
    class _Fake:
        postgres = None

        async def artifact_accessible_list(self, **kwargs):
            return {
                "items": [
                    {"id": "a1", "filename": "x.jpg", "mime_type": "image/jpeg"},
                ]
            }

    async def _run():
        svc = MapGeoService(storage=_Fake())

        async def _rows(user_id, **kwargs):
            return [
                {
                    "artifact_id": "a1",
                    "id": "a1",
                    "latitude": 35.0,
                    "longitude": 51.0,
                    "mime_type": "image/jpeg",
                }
            ]

        svc._collect_geo_rows = _rows  # type: ignore[method-assign]
        fp = await svc.geo_fingerprint("u1")
        assert "fingerprint" in fp
        assert "latitude" not in fp
        assert "files" not in fp
        assert "artifact_id" not in fp
        dumped = str(fp)
        assert "35.0" not in dumped
        assert "51.0" not in dumped

    asyncio.run(_run())


def test_geo_tile_rbac_bbox_and_permission_cache_key():
    class _Fake:
        postgres = None

        async def artifact_accessible_list(self, **kwargs):
            return {"items": [{"id": "inside"}, {"id": "outside"}]}

    async def _run():
        import json

        svc = MapGeoService(storage=_Fake())

        async def _rows(user_id, **kwargs):
            return [
                {
                    "artifact_id": "inside",
                    "id": "inside",
                    "latitude": 0.1,
                    "longitude": 0.1,
                },
                {
                    "artifact_id": "outside",
                    "id": "outside",
                    "latitude": 50.0,
                    "longitude": 50.0,
                },
            ]

        async def _fp(*_a, **_k):
            return {"fingerprint": "abc", "permission_key": "perm1"}

        svc._collect_geo_rows = _rows  # type: ignore[method-assign]
        svc.geo_fingerprint = _fp  # type: ignore[method-assign]
        tile = await svc.geo_tile("u1", z=2, x=2, y=1, fmt="geojson")
        assert tile["permission_key"] == "perm1"
        assert tile["content_type"] == "application/geo+json"
        body = json.loads(tile["content"].decode("utf-8"))
        bounds = _tile_bounds(2, 2, 1)
        for f in body.get("features") or []:
            coords = f["geometry"]["coordinates"]
            lon, lat = float(coords[0]), float(coords[1])
            assert _in_bounds(lat, lon, bounds)

    asyncio.run(_run())
