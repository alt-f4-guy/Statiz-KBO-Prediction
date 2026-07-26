import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd


class IOUtilsTests(unittest.TestCase):
    def test_atomic_to_csv_preserves_existing_file_on_write_failure(self):
        from io_utils import atomic_to_csv

        with TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.write_text("old\n", encoding="utf-8")
            frame = pd.DataFrame({"value": [1]})
            with patch.object(
                pd.DataFrame,
                "to_csv",
                side_effect=OSError("쓰기 실패"),
            ):
                with self.assertRaisesRegex(OSError, "쓰기 실패"):
                    atomic_to_csv(frame, path)

            self.assertEqual(path.read_text(encoding="utf-8"), "old\n")
            self.assertEqual(list(Path(directory).glob(".data.csv.*.tmp")), [])

    def test_atomic_to_csv_creates_file_in_new_directory(self):
        from io_utils import atomic_to_csv

        with TemporaryDirectory() as directory:
            path = Path(directory) / "subdir" / "data.csv"
            frame = pd.DataFrame({"value": [1, 2]})
            atomic_to_csv(frame, path)

            self.assertTrue(path.exists())
            loaded = pd.read_csv(path)
            self.assertEqual(len(loaded), 2)


if __name__ == "__main__":
    unittest.main()
