"""
read_c3d.py – Pure Python C3D file reader (no external dependencies)
Supports little-endian (Intel) C3D files with float or integer data.

Usage:
    python3 read_c3d.py <file.c3d>
"""

import struct
import sys
import os


# ── Helper ────────────────────────────────────────────────────────────────────

def _read_string_array(data, offset, dims):
    """Read a 2-D char array (dims = [str_len, count]) as a list of strings."""
    str_len, count = dims[0], dims[1]
    result = []
    for i in range(count):
        s = data[offset + i * str_len: offset + (i + 1) * str_len]
        result.append(s.decode("ascii", errors="replace").strip().rstrip("\x00"))
    return result


# ── Parameter parser ──────────────────────────────────────────────────────────

def parse_parameters(data, param_block_index):
    """
    Parse the C3D parameter section.

    param_block_index : 1-based block number (from header byte 0).
    Returns (groups, params) where
        groups : {group_id -> group_name}
        params : {(group_id, param_name) -> value}
    """
    start = (param_block_index - 1) * 512
    num_blocks = data[start + 2]
    processor  = data[start + 3]          # 84 = 'T' = Intel/LE in some files
    # Normalise: 1 = Intel LE, 84 is also Intel in many files
    little_endian = processor in (1, 84)

    end    = start + num_blocks * 512
    offset = start + 4                    # skip 4-byte section header

    groups = {}
    params = {}

    while offset < end:
        if offset + 2 > len(data):
            break

        name_len = struct.unpack_from("<b", data, offset)[0]  # signed
        if name_len == 0:
            break
        offset += 1

        group_id = struct.unpack_from("<b", data, offset)[0]  # signed
        offset   += 1

        is_group = group_id < 0
        abs_id   = abs(group_id)
        abs_nlen = abs(name_len)

        if offset + abs_nlen > len(data):
            break

        name   = data[offset: offset + abs_nlen].decode("ascii", errors="replace")
        offset += abs_nlen

        next_offset_bytes = struct.unpack_from("<H", data, offset)[0]
        offset += 2

        if is_group:
            # Group record
            desc_len   = data[offset]
            groups[abs_id] = name
            offset    += 1 + desc_len
        else:
            # Parameter record
            data_type = struct.unpack_from("<b", data, offset)[0]   # signed
            offset   += 1
            num_dims  = data[offset]
            offset   += 1
            dims      = list(data[offset: offset + num_dims])
            offset   += num_dims

            total    = 1
            for d in dims:
                total *= d

            abs_type = abs(data_type)
            value    = None

            if abs_type == 1:   # char / string
                raw = data[offset: offset + total]
                if len(dims) == 2:
                    value = _read_string_array(data, offset, dims)
                else:
                    value = raw.decode("ascii", errors="replace").strip().rstrip("\x00")
            elif abs_type == 2:  # int16
                fmt   = "<h" if little_endian else ">h"
                value = [struct.unpack_from(fmt, data, offset + i * 2)[0]
                         for i in range(total)]
                if len(value) == 1:
                    value = value[0]
            elif abs_type == 4:  # float32
                fmt   = "<f" if little_endian else ">f"
                value = [struct.unpack_from(fmt, data, offset + i * 4)[0]
                         for i in range(total)]
                if len(value) == 1:
                    value = value[0]

            params[(abs_id, name)] = value
            offset += total * abs_type

            # Description field
            if offset < end:
                desc_len = data[offset]
                offset  += 1 + desc_len

    return groups, params


# ── Data reader ───────────────────────────────────────────────────────────────

def read_point_data(data, header, num_frames):
    """
    Read 3-D point (marker) data.

    Returns list of frames; each frame is a list of (x, y, z, residual) tuples.
    NaN is used for occluded/missing markers.
    """
    data_start  = (header["data_start"] - 1) * 512
    num_points  = header["num_points"]
    scale       = header["scale"]
    is_float    = scale < 0                  # negative scale → float data

    frames = []
    offset = data_start

    for _ in range(num_frames):
        frame = []
        for _ in range(num_points):
            if is_float:
                x   = struct.unpack_from("<f", data, offset)[0];     offset += 4
                y   = struct.unpack_from("<f", data, offset)[0];     offset += 4
                z   = struct.unpack_from("<f", data, offset)[0];     offset += 4
                res = struct.unpack_from("<f", data, offset)[0];     offset += 4
                # Missing marker: x == 0 AND residual < 0 is common convention
                if res < 0:
                    frame.append((float("nan"), float("nan"), float("nan"), res))
                else:
                    frame.append((x, y, z, res))
            else:
                # Integer format: multiply by |scale|
                xi  = struct.unpack_from("<h", data, offset)[0];     offset += 2
                yi  = struct.unpack_from("<h", data, offset)[0];     offset += 2
                zi  = struct.unpack_from("<h", data, offset)[0];     offset += 2
                ri  = struct.unpack_from("<H", data, offset)[0];     offset += 2
                s   = abs(scale)
                if xi == 0 and ri & 0xFF == 0xFF:
                    frame.append((float("nan"), float("nan"), float("nan"), -1))
                else:
                    frame.append((xi * s, yi * s, zi * s, ri))
        frames.append(frame)

    return frames


# ── Main reader ───────────────────────────────────────────────────────────────

def read_c3d(filepath):
    """
    Read a C3D file and return a dict with all metadata and point data.
    """
    with open(filepath, "rb") as f:
        data = f.read()

    # ── Header ────────────────────────────────────────────────────────────────
    param_block = data[0]                                    # 1-based
    magic       = data[1]
    assert magic == 0x50, f"Not a valid C3D file (magic={hex(magic)})"

    header = {
        "param_block":  param_block,
        "num_points":   struct.unpack_from("<H", data,  2)[0],
        "num_analog":   struct.unpack_from("<H", data,  4)[0],
        "first_frame":  struct.unpack_from("<H", data,  6)[0],
        "last_frame":   struct.unpack_from("<H", data,  8)[0],
        "max_gap":      struct.unpack_from("<H", data, 10)[0],
        "scale":        struct.unpack_from("<f", data, 12)[0],
        "data_start":   struct.unpack_from("<H", data, 16)[0],
        "analog_samp":  struct.unpack_from("<H", data, 18)[0],
        "frame_rate":   struct.unpack_from("<f", data, 20)[0],
    }
    header["num_frames"] = header["last_frame"] - header["first_frame"] + 1

    # ── Parameters ────────────────────────────────────────────────────────────
    groups, params = parse_parameters(data, header["param_block"])

    # Build a friendly nested dict: params_dict[GROUP_NAME][PARAM_NAME] = value
    params_dict = {}
    for gid, gname in groups.items():
        params_dict[gname] = {}
    for (gid, pname), val in params.items():
        gname = groups.get(gid, f"GROUP_{gid}")
        if gname not in params_dict:
            params_dict[gname] = {}
        params_dict[gname][pname] = val

    # ── Point labels ──────────────────────────────────────────────────────────
    labels = params_dict.get("POINT", {}).get("LABELS", [])
    units  = params_dict.get("POINT", {}).get("UNITS", "unknown").strip("\x00")

    # ── Point data ────────────────────────────────────────────────────────────
    frames = read_point_data(data, header, header["num_frames"])

    return {
        "header":     header,
        "parameters": params_dict,
        "labels":     labels,
        "units":      units,
        "frames":     frames,   # list[frame] of list[marker] of (x,y,z,residual)
    }


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_summary(c3d):
    h   = c3d["header"]
    sep = "─" * 60

    print(sep)
    print("  C3D FILE SUMMARY")
    print(sep)
    print(f"  Frame rate      : {h['frame_rate']} Hz")
    print(f"  Frames          : {h['first_frame']} – {h['last_frame']}  ({h['num_frames']} total)")
    print(f"  Duration        : {h['num_frames'] / h['frame_rate']:.3f} s")
    print(f"  3D markers      : {h['num_points']}")
    print(f"  Analog channels : {h['num_analog']}")
    print(f"  Point units     : {c3d['units']}")
    print(f"  Scale factor    : {h['scale']}  ({'float data' if h['scale'] < 0 else 'integer data'})")

    print()
    print("  MARKER LABELS")
    print(sep)
    for i, lbl in enumerate(c3d["labels"]):
        print(f"  [{i:>3}] {lbl}")

    print()
    print("  FIRST FRAME – MARKER POSITIONS (x, y, z) in", c3d["units"])
    print(sep)
    if c3d["frames"]:
        frame0 = c3d["frames"][0]
        for i, (x, y, z, res) in enumerate(frame0):
            lbl = c3d["labels"][i] if i < len(c3d["labels"]) else f"M{i}"
            if x != x:   # NaN = occluded
                print(f"  {lbl:<12} occluded")
            else:
                print(f"  {lbl:<12}  x={x:10.3f}  y={y:10.3f}  z={z:10.3f}  residual={res:.4f}")

    print(sep)

    # Available parameter groups
    print()
    print("  PARAMETER GROUPS FOUND")
    print(sep)
    for gname, pdict in c3d["parameters"].items():
        keys = ", ".join(pdict.keys())
        print(f"  [{gname}]  {keys}")
    print(sep)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python3 {os.path.basename(__file__)} <file.c3d>")
        sys.exit(1)

    filepath = sys.argv[1]
    print(f"\nReading: {filepath}\n")
    c3d = read_c3d(filepath)
    print_summary(c3d)