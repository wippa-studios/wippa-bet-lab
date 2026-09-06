from pathlib import Path

import duckdb
import pytest

from betlab.datasets.importer import DatasetImporter, Dataset
from betlab.datasets.validator import DatasetValidator, ValidationReport
from betlab.datasets.versioner import DatasetVersioner
from betlab.datasets.snapshot import DatasetSnapshot

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_CSV = FIXTURES_DIR / "sample_events.csv"


class TestDatasetImporter:
    def test_import_csv_returns_dataset(self):
        importer = DatasetImporter()
        ds = importer.import_csv(SAMPLE_CSV, sport="tennis", name="test_csv")
        assert isinstance(ds, Dataset)
        assert ds.name == "test_csv"
        assert ds.sport == "tennis"
        assert ds.row_count == 5
        assert "event_id" in ds.columns
        assert len(ds.data_hash) == 64
        importer.close()

    def test_import_csv_computes_market_count(self):
        importer = DatasetImporter()
        ds = importer.import_csv(SAMPLE_CSV, sport="tennis", name="test")
        assert ds.market_count == 5
        importer.close()

    def test_import_csv_duckdb_table_exists(self):
        importer = DatasetImporter()
        ds = importer.import_csv(SAMPLE_CSV, sport="tennis", name="test")
        conn = importer._get_conn()
        count = conn.execute(f"SELECT COUNT(*) FROM {ds.table_name}").fetchone()[0]
        assert count == 5
        importer.close()

    def test_import_csv_generates_unique_ids(self):
        importer = DatasetImporter()
        ds1 = importer.import_csv(SAMPLE_CSV, sport="tennis", name="a")
        ds2 = importer.import_csv(SAMPLE_CSV, sport="tennis", name="b")
        assert ds1.id != ds2.id
        importer.close()

    def test_import_json(self, tmp_path):
        import json

        data = [
            {"event_id": "J1", "sport": "tennis", "odds": 1.5},
            {"event_id": "J2", "sport": "tennis", "odds": 2.0},
        ]
        f = tmp_path / "test.json"
        f.write_text(json.dumps(data))

        importer = DatasetImporter()
        ds = importer.import_json(f, sport="tennis", name="json_test")
        assert ds.row_count == 2
        assert "event_id" in ds.columns
        importer.close()


class TestDatasetValidator:
    def test_validate_clean_data(self):
        importer = DatasetImporter()
        ds = importer.import_csv(SAMPLE_CSV, sport="tennis", name="clean")
        validator = DatasetValidator()
        report = validator.validate(ds, importer._get_conn())
        assert report.is_valid
        assert len(report.errors) == 0
        assert report.stats["row_count"] == 5
        importer.close()

    def test_validate_missing_required_columns(self):
        importer = DatasetImporter()
        conn = importer._get_conn()
        conn.execute("CREATE TABLE bad_dataset (col1 VARCHAR, col2 VARCHAR)")
        conn.execute("INSERT INTO bad_dataset VALUES ('a', 'b')")

        class FakeDataset:
            id = "bad"
            table_name = "bad_dataset"
            columns = ["col1", "col2"]
            row_count = 1

        validator = DatasetValidator()
        report = validator.validate(FakeDataset(), conn)
        assert not report.is_valid
        assert any("event_id" in e for e in report.errors)
        importer.close()

    def test_validate_catches_invalid_prices(self):
        importer = DatasetImporter()
        conn = importer._get_conn()
        conn.execute("""
            CREATE TABLE price_bad (
                event_id VARCHAR, start_time VARCHAR,
                p1_back_price VARCHAR, p2_back_price VARCHAR
            )
        """)
        conn.execute("INSERT INTO price_bad VALUES ('E1', '2024-01-01', '0.5', '1500')")

        class FakeDataset:
            id = "price_bad"
            table_name = "price_bad"
            columns = ["event_id", "start_time", "p1_back_price", "p2_back_price"]
            row_count = 1

        validator = DatasetValidator()
        report = validator.validate(FakeDataset(), conn)
        assert not report.is_valid
        assert any("price" in e.lower() for e in report.errors)
        importer.close()


class TestDatasetVersioner:
    def test_create_version(self):
        importer = DatasetImporter()
        ds = importer.import_csv(SAMPLE_CSV, sport="tennis", name="ver_test")
        versioner = DatasetVersioner(":memory:")
        v = versioner.create_version(ds, description="first version")
        assert v.version_number == 1
        assert v.dataset_id == ds.id
        importer.close()

    def test_list_versions(self):
        importer = DatasetImporter()
        ds = importer.import_csv(SAMPLE_CSV, sport="tennis", name="ver_list")
        versioner = DatasetVersioner(":memory:")
        versioner.create_version(ds, description="v1")
        versioner.create_version(ds, description="v2")
        versions = versioner.list_versions(ds.id)
        assert len(versions) == 2
        assert versions[0].version_number == 2
        importer.close()


class TestDatasetSnapshot:
    def test_create_and_get_snapshot(self):
        importer = DatasetImporter()
        ds = importer.import_csv(SAMPLE_CSV, sport="tennis", name="snap_test")
        conn = importer._get_conn()
        snapshotter = DatasetSnapshot(conn)
        meta = snapshotter.create_snapshot(ds)
        assert meta.row_count == 5
        retrieved_meta, records = snapshotter.get_snapshot(meta.snapshot_id)
        assert len(records) == 5
        importer.close()

    def test_snapshot_with_time_range(self):
        importer = DatasetImporter()
        ds = importer.import_csv(SAMPLE_CSV, sport="tennis", name="snap_range")
        conn = importer._get_conn()
        snapshotter = DatasetSnapshot(conn)
        meta = snapshotter.create_snapshot(
            ds, time_range=("2024-01-01", "2024-06-30")
        )
        assert meta.row_count == 3
        importer.close()

    def test_get_missing_snapshot_raises(self):
        conn = duckdb.connect(":memory:")
        snapshotter = DatasetSnapshot(conn)
        with pytest.raises(KeyError):
            snapshotter.get_snapshot("nonexistent")
