import argparse
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import pyopenvdb as vdb
except ImportError:
    import openvdb as vdb

HEADER = struct.Struct("<4sIIf3f")
MAGIC = b"MPMG"
NODE_FLOATS = 7
MASS = 3


def read_grid(path):
    data = Path(path).read_bytes()
    magic, n, frame, dx, ox, oy, oz = HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ValueError(f"{path} is not an MPM grid dump")
    nodes = np.frombuffer(data, dtype="<f4", offset=HEADER.size)
    mass = nodes.reshape(n, n, n, NODE_FLOATS)[..., MASS]
    density = np.ascontiguousarray(mass / (dx * dx * dx))
    return frame, dx, (ox, oy, oz), density


def write_vdb(path, density, dx, origin):
    grid = vdb.FloatGrid()
    grid.name = "density"
    grid.gridClass = vdb.GridClass.FOG_VOLUME
    grid.copyFromArray(density)
    ox, oy, oz = origin
    grid.transform = vdb.createLinearTransform(
        [
            [dx, 0.0, 0.0, 0.0],
            [0.0, dx, 0.0, 0.0],
            [0.0, 0.0, dx, 0.0],
            [ox, oy, oz, 1.0],
        ]
    )
    vdb.write(str(path), grids=[grid])


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", nargs="?", default="data", help="dump file or directory")
    parser.add_argument("dst", nargs="?", default="data/vdb", help="output directory")
    args = parser.parse_args(argv)

    src = Path(args.src)
    paths = [src] if src.is_file() else sorted(src.glob("grid_*.bin"))
    if not paths:
        raise SystemExit(f"no grid dumps in {src}")

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    for path in paths:
        frame, dx, origin, density = read_grid(path)
        out = dst / f"grid_{frame:06}.vdb"
        write_vdb(out, density, dx, origin)
        print(out)


if __name__ == "__main__":
    main()
