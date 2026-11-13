"""Repair recordings cut off by a crash so the pipeline can seek them.

When the recorder is killed (WSL shut down, power loss) the open .mkv files
keep their packets but lack a duration and seek index: OpenCV reports a
nonsense frame count and cannot seek. This remuxes such files
(src.recording.repair_recording, no re-encoding). Healthy files are left
alone. The cut-off originals are moved to `<hour folder>/cut_originals/`
(not deleted) — find_recordings ignores subfolders, so the hour folder is
a clean pipeline input afterwards.

Usage:
    python scripts/repair_recordings.py ~/depot_recordings/2026-11-13/0700
"""

import argparse
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.recording import FILE_SUFFIX, repair_recording


def seekable(path) -> bool:
    import cv2

    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return 0 < n < 10_000_000


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="+", help="Hour folders to check")
    args = parser.parse_args()

    for folder in map(lambda f: Path(f).expanduser(), args.folders):
        for path in sorted(folder.glob(f"*{FILE_SUFFIX}")):
            if seekable(path):
                print(f"ok        {path.name}")
                continue
            tmp = path.with_suffix(".repairing" + FILE_SUFFIX)
            n = repair_recording(path, tmp)
            if not seekable(tmp):
                tmp.unlink(missing_ok=True)
                print(f"FAILED    {path.name} (left as is)")
                continue
            keep = folder / "cut_originals"
            keep.mkdir(exist_ok=True)
            path.rename(keep / path.name)
            tmp.rename(path)
            print(f"repaired  {path.name}: {n} packets ({n / 20 / 60:.1f} min at 20 fps); "
                  f"original in cut_originals/")


if __name__ == "__main__":
    main()
