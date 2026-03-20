"""
S3O → GLB Converter

Converts a parsed S3OModel into a standard glTF 2.0 Binary (.glb) file.
Each S3O piece becomes a named glTF node with its own mesh, preserving
the piece hierarchy for animation and weapon identification.

Key design decisions:
- Piece names are preserved exactly as glTF node names
- Piece offsets become node translations
- S3O Y-up matches glTF Y-up (no axis conversion needed)
- UV t-coordinate is flipped (1-t) to match glTF convention
- A single default PBR material is created (textures applied later in viewer)
"""

import struct
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from s3o_parser import S3OModel, S3OPiece


def _pad_to_4(data: bytearray) -> bytearray:
    """Pad bytearray to 4-byte alignment."""
    remainder = len(data) % 4
    if remainder > 0:
        data += b'\x00' * (4 - remainder)
    return data


class GLBBuilder:
    """Builds a glTF 2.0 Binary (GLB) file from scratch."""

    def __init__(self):
        self.buffer_data = bytearray()
        self.buffer_views = []
        self.accessors = []
        self.meshes = []
        self.nodes = []
        self.materials = []
        self.scenes = [{"nodes": []}]

    def add_buffer_view(self, data: bytes, target: int = 0) -> int:
        """Add raw data as a buffer view. Returns buffer view index."""
        # Align to 4 bytes
        offset = len(self.buffer_data)
        remainder = offset % 4
        if remainder > 0:
            padding = 4 - remainder
            self.buffer_data += b'\x00' * padding
            offset += padding

        self.buffer_data += data
        bv = {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(data),
        }
        if target:
            bv["target"] = target
        idx = len(self.buffer_views)
        self.buffer_views.append(bv)
        return idx

    def add_accessor(self, buffer_view: int, component_type: int,
                     count: int, accessor_type: str,
                     min_vals=None, max_vals=None) -> int:
        """Add an accessor. Returns accessor index."""
        acc = {
            "bufferView": buffer_view,
            "componentType": component_type,
            "count": count,
            "type": accessor_type,
        }
        if min_vals is not None:
            acc["min"] = min_vals
        if max_vals is not None:
            acc["max"] = max_vals
        idx = len(self.accessors)
        self.accessors.append(acc)
        return idx

    def add_default_material(self) -> int:
        """Add a default PBR metallic-roughness material."""
        mat = {
            "name": "S3O_Default",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.8, 0.8, 0.8, 1.0],
                "metallicFactor": 0.5,
                "roughnessFactor": 0.5,
            },
            "doubleSided": True,
        }
        idx = len(self.materials)
        self.materials.append(mat)
        return idx

    def add_piece_mesh(self, piece: S3OPiece, material_idx: int) -> Optional[int]:
        """Add a mesh for an S3O piece. Returns mesh index or None if empty."""
        tri_indices = piece.triangle_indices()
        if len(piece.vertices) == 0 or len(tri_indices) == 0:
            return None

        # Build vertex data arrays
        positions = np.array(
            [[v.x, v.y, v.z] for v in piece.vertices], dtype=np.float32
        )
        normals = np.array(
            [[v.nx, v.ny, v.nz] for v in piece.vertices], dtype=np.float32
        )
        # Flip UV t-coordinate for glTF convention
        texcoords = np.array(
            [[v.s, 1.0 - v.t] for v in piece.vertices], dtype=np.float32
        )
        indices_arr = np.array(tri_indices, dtype=np.uint32)

        # Compute bounds
        pos_min = positions.min(axis=0).tolist()
        pos_max = positions.max(axis=0).tolist()

        # Add position buffer view (target = ARRAY_BUFFER = 34962)
        pos_bv = self.add_buffer_view(positions.tobytes(), target=34962)
        pos_acc = self.add_accessor(
            pos_bv, 5126, len(positions), "VEC3",
            min_vals=pos_min, max_vals=pos_max
        )

        # Add normal buffer view
        norm_bv = self.add_buffer_view(normals.tobytes(), target=34962)
        norm_acc = self.add_accessor(norm_bv, 5126, len(normals), "VEC3")

        # Add texcoord buffer view
        uv_bv = self.add_buffer_view(texcoords.tobytes(), target=34962)
        uv_acc = self.add_accessor(uv_bv, 5126, len(texcoords), "VEC2")

        # Add index buffer view (target = ELEMENT_ARRAY_BUFFER = 34963)
        idx_bv = self.add_buffer_view(indices_arr.tobytes(), target=34963)
        idx_acc = self.add_accessor(
            idx_bv, 5125, len(indices_arr), "SCALAR",
            min_vals=[int(indices_arr.min())],
            max_vals=[int(indices_arr.max())]
        )

        # Create mesh
        mesh = {
            "name": piece.name,
            "primitives": [{
                "attributes": {
                    "POSITION": pos_acc,
                    "NORMAL": norm_acc,
                    "TEXCOORD_0": uv_acc,
                },
                "indices": idx_acc,
                "material": material_idx,
                "mode": 4,  # TRIANGLES
            }],
        }
        idx = len(self.meshes)
        self.meshes.append(mesh)
        return idx

    def add_piece_node(self, piece: S3OPiece, material_idx: int,
                       parent_node_idx: Optional[int] = None) -> int:
        """Recursively add nodes for a piece and its children.
        Returns the node index of this piece."""
        # Create mesh for this piece (may be None if piece has no geometry)
        mesh_idx = self.add_piece_mesh(piece, material_idx)

        node = {"name": piece.name}

        # Set translation from piece offset
        ox, oy, oz = piece.offset
        if ox != 0 or oy != 0 or oz != 0:
            node["translation"] = [ox, oy, oz]

        if mesh_idx is not None:
            node["mesh"] = mesh_idx

        node_idx = len(self.nodes)
        self.nodes.append(node)

        # Process children
        child_indices = []
        for child in piece.children:
            child_idx = self.add_piece_node(child, material_idx, node_idx)
            child_indices.append(child_idx)

        if child_indices:
            self.nodes[node_idx]["children"] = child_indices

        return node_idx

    def apply_now_rotations(self, now_rots: dict, node_name_to_idx: Dict[str, int],
                            child_offsets: Dict[str, tuple] = None):
        """
        Apply Create() 'turn piece to axis <value> now' rotations as static
        node rotations in the GLB. This sets the rest pose to match what the
        Recoil engine sets up via Create() before any animation plays.
        now_rots: {(piece_lower, axis_int, True): degrees}
        child_offsets: {piece_lower: (cx, cy, cz)} — first child's offset for
                       sideways-mount detection (|cx| > |cz| → sideways piece).
        """
        import math

        child_offsets = child_offsets or {}

        def _is_sideways(piece: str) -> bool:
            """True if piece's child extends primarily along X (not Z)."""
            co = child_offsets.get(piece)
            if co is None:
                return False
            return abs(co[0]) > abs(co[2])

        def _euler_to_quat(rx_deg, ry_deg, rz_deg, sideways: bool = False):
            """Recoil→glTF rotation. Y-axis is negated for handedness correction.
            For sideways-mounted pieces (X-extended child), the handedness
            difference between Recoil (left-handed) and glTF (right-handed)
            already inverts the Y rotation effect, so we cancel the negation."""
            ry_sign = 1.0 if sideways else -1.0
            def aq(axis, deg):
                a = math.radians(deg) / 2
                s, w = math.sin(a), math.cos(a)
                return [s,0,0,w] if axis==0 else ([0,s,0,w] if axis==1 else [0,0,s,w])
            def qm(a, b):
                ax,ay,az,aw = a; bx,by,bz,bw = b
                return [aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx,
                        aw*bz+ax*by-ay*bx+az*bw, aw*bw-ax*bx-ay*by-az*bz]
            return qm(qm(aq(2,rz_deg), aq(0,rx_deg)), aq(1, ry_sign * ry_deg))

        # Group by piece
        by_piece: Dict[str, Dict[int, float]] = {}
        for (piece, axis, _), deg in now_rots.items():
            by_piece.setdefault(piece, {})[axis] = deg

        for piece, axes in by_piece.items():
            node_idx = node_name_to_idx.get(piece)
            if node_idx is None:
                continue
            rx = axes.get(0, 0.0)
            ry = axes.get(1, 0.0)
            rz = axes.get(2, 0.0)
            if rx == 0 and ry == 0 and rz == 0:
                continue
            sideways = _is_sideways(piece)
            self.nodes[node_idx]["rotation"] = _euler_to_quat(rx, ry, rz, sideways)

    def add_animation(self, anim_name: str, tracks: list,
                      node_name_to_idx: Dict[str, int],
                      piece_offsets: Dict[str, tuple] = None,
                      child_offsets: Dict[str, tuple] = None):
        """
        Add a glTF animation from BOS animation tracks.

        tracks       : List[BosTrack] from bos_animator.extract_walk_animation()
        node_name_to_idx : {piece_name_lower: node_index}
        piece_offsets    : {piece_name_lower: (ox, oy, oz)} — S3O rest translations.
                           BOS move values are deltas from this rest position.
        child_offsets    : {piece_name_lower: (cx, cy, cz)} — first child's offset.
                           Used to detect sideways-mounted pieces (|cx| > |cz|).
        """
        import math
        from collections import defaultdict

        piece_offsets = piece_offsets or {}
        child_offsets = child_offsets or {}

        def _is_sideways(piece: str) -> bool:
            """True if piece's child extends primarily along X (not Z)."""
            co = child_offsets.get(piece)
            if co is None:
                return False
            return abs(co[0]) > abs(co[2])

        # Group tracks by piece
        rot_by_piece: Dict[str, Dict[int, object]] = defaultdict(dict)
        trans_by_piece: Dict[str, Dict[int, object]] = defaultdict(dict)

        for track in tracks:
            name_lower = track.piece.lower()
            if name_lower not in node_name_to_idx:
                continue
            if track.is_rotation:
                rot_by_piece[name_lower][track.axis] = track
            else:
                trans_by_piece[name_lower][track.axis] = track

        channels = []
        samplers = []

        def _add_sampler(times: list, values: list, accessor_type: str) -> int:
            t_arr = np.array(times, dtype=np.float32)
            t_bv = self.add_buffer_view(t_arr.tobytes())
            t_acc = self.add_accessor(t_bv, 5126, len(times), "SCALAR",
                                      min_vals=[float(t_arr.min())],
                                      max_vals=[float(t_arr.max())])
            v_arr = np.array(values, dtype=np.float32)
            v_bv = self.add_buffer_view(v_arr.tobytes())
            v_acc = self.add_accessor(v_bv, 5126, len(times), accessor_type)
            idx = len(samplers)
            samplers.append({"input": t_acc, "interpolation": "LINEAR", "output": v_acc})
            return idx

        def _interp(axis_tracks: dict, axis: int, t: float) -> float:
            if axis not in axis_tracks:
                return 0.0
            kfs = axis_tracks[axis].keyframes
            if not kfs:
                return 0.0
            if t <= kfs[0].time:
                return kfs[0].value
            if t >= kfs[-1].time:
                return kfs[-1].value
            for i in range(len(kfs) - 1):
                if kfs[i].time <= t <= kfs[i + 1].time:
                    f = (t - kfs[i].time) / (kfs[i + 1].time - kfs[i].time)
                    return kfs[i].value + f * (kfs[i + 1].value - kfs[i].value)
            return kfs[-1].value

        def _axis_quat(axis: int, deg: float) -> list:
            """Single-axis quaternion: axis 0=X, 1=Y, 2=Z."""
            a = math.radians(deg) / 2
            s = math.sin(a)
            w = math.cos(a)
            if axis == 0: return [s, 0.0, 0.0, w]
            if axis == 1: return [0.0, s, 0.0, w]
            return [0.0, 0.0, s, w]

        def _quat_mul(a: list, b: list) -> list:
            """Quaternion multiply a * b (Hamilton product)."""
            ax, ay, az, aw = a
            bx, by, bz, bw = b
            return [
                aw*bx + ax*bw + ay*bz - az*by,
                aw*by - ax*bz + ay*bw + az*bx,
                aw*bz + ax*by - ay*bx + az*bw,
                aw*bw - ax*bx - ay*by - az*bz,
            ]

        def _spring_to_quat(rx_deg: float, ry_deg: float, rz_deg: float,
                            sideways: bool = False) -> list:
            """Recoil BOS rotation → glTF quaternion.
            Y is negated to account for Recoil→glTF handedness conversion.
            Exception: sideways-mounted pieces (X-extended child) already have
            the Z-movement direction inverted by the handedness difference, so
            we cancel the Y-negation for those pieces.
            Rotation order: R(Z)*R(X)*R(Y) = intrinsic YXZ.
            """
            ry_sign = 1.0 if sideways else -1.0
            qy = _axis_quat(1, ry_sign * ry_deg)
            qx = _axis_quat(0, rx_deg)
            qz = _axis_quat(2, rz_deg)
            return _quat_mul(_quat_mul(qz, qx), qy)

        # --- Rotation channels ---
        for piece, axis_tracks in rot_by_piece.items():
            node_idx = node_name_to_idx[piece]
            sideways = _is_sideways(piece)
            all_times = sorted({kf.time
                                 for tr in axis_tracks.values()
                                 for kf in tr.keyframes})
            quats = []
            for t in all_times:
                rx = _interp(axis_tracks, 0, t)
                ry = _interp(axis_tracks, 1, t)
                rz = _interp(axis_tracks, 2, t)
                quats.extend(_spring_to_quat(rx, ry, rz, sideways))
            s = _add_sampler(all_times, quats, "VEC4")
            channels.append({"sampler": s, "target": {"node": node_idx, "path": "rotation"}})

        # --- Translation channels ---
        for piece, axis_tracks in trans_by_piece.items():
            node_idx = node_name_to_idx[piece]
            rest = piece_offsets.get(piece, (0.0, 0.0, 0.0))
            all_times = sorted({kf.time
                                 for tr in axis_tracks.values()
                                 for kf in tr.keyframes})
            vecs = []
            for t in all_times:
                dx = _interp(axis_tracks, 0, t)
                dy = _interp(axis_tracks, 1, t)
                dz = _interp(axis_tracks, 2, t)
                vecs.extend([rest[0] + dx, rest[1] + dy, rest[2] + dz])
            s = _add_sampler(all_times, vecs, "VEC3")
            channels.append({"sampler": s, "target": {"node": node_idx, "path": "translation"}})

        if channels:
            if not hasattr(self, 'animations'):
                self.animations = []
            self.animations.append({
                "name": anim_name,
                "channels": channels,
                "samplers": samplers,
            })
            print(f"  GLB animation '{anim_name}': {len(channels)} channels")

    def build_glb(self) -> bytes:
        """Build the final GLB binary."""
        # Construct the JSON
        gltf = {
            "asset": {
                "version": "2.0",
                "generator": "BAR-S3O-Converter",
            },
            "scene": 0,
            "scenes": self.scenes,
            "nodes": self.nodes,
            "meshes": self.meshes,
            "accessors": self.accessors,
            "bufferViews": self.buffer_views,
            "buffers": [{
                "byteLength": len(self.buffer_data),
            }],
        }
        if self.materials:
            gltf["materials"] = self.materials
        if hasattr(self, 'animations') and self.animations:
            gltf["animations"] = self.animations

        json_str = json.dumps(gltf, separators=(',', ':'))
        json_bytes = json_str.encode('utf-8')

        # Pad JSON to 4-byte alignment with spaces
        json_pad = (4 - len(json_bytes) % 4) % 4
        json_bytes += b' ' * json_pad

        # Pad binary buffer to 4-byte alignment with zeros
        bin_data = bytes(self.buffer_data)
        bin_pad = (4 - len(bin_data) % 4) % 4
        bin_data += b'\x00' * bin_pad

        # GLB structure:
        # Header: magic(4) + version(4) + length(4) = 12
        # JSON chunk: length(4) + type(4) + data
        # BIN chunk: length(4) + type(4) + data
        total_length = (
            12 +                           # GLB header
            8 + len(json_bytes) +          # JSON chunk header + data
            8 + len(bin_data)              # BIN chunk header + data
        )

        glb = bytearray()
        # GLB header
        glb += struct.pack('<I', 0x46546C67)  # magic: "glTF"
        glb += struct.pack('<I', 2)            # version
        glb += struct.pack('<I', total_length) # total length

        # JSON chunk
        glb += struct.pack('<I', len(json_bytes))
        glb += struct.pack('<I', 0x4E4F534A)   # type: "JSON"
        glb += json_bytes

        # BIN chunk
        glb += struct.pack('<I', len(bin_data))
        glb += struct.pack('<I', 0x004E4942)   # type: "BIN\0"
        glb += bin_data

        return bytes(glb)


def convert_s3o_to_glb(model: S3OModel) -> bytes:
    """Convert a parsed S3OModel to GLB binary data."""
    builder = GLBBuilder()

    # Add default material
    mat_idx = builder.add_default_material()

    if model.root_piece:
        root_node_idx = builder.add_piece_node(model.root_piece, mat_idx)
        builder.scenes[0]["nodes"] = [root_node_idx]

    return builder.build_glb()


def s3o_file_to_glb(input_path: str, output_path: str = None) -> str:
    """Convert an S3O file to GLB. Returns the output path."""
    from s3o_parser import parse_s3o

    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = base + '.glb'

    model = parse_s3o(input_path)
    glb_data = convert_s3o_to_glb(model)

    with open(output_path, 'wb') as f:
        f.write(glb_data)

    return output_path


if __name__ == "__main__":
    import sys
    from s3o_parser import parse_s3o, print_piece_tree

    if len(sys.argv) < 2:
        print("Usage: python s3o_to_glb.py <input.s3o> [output.glb]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Parsing {input_path}...")
    model = parse_s3o(input_path)

    print(f"  Version: {model.version}")
    print(f"  Radius: {model.radius:.2f}, Height: {model.height:.2f}")
    print(f"  Texture 1: {model.texture1}")
    print(f"  Texture 2: {model.texture2}")
    print(f"  Piece tree:")
    if model.root_piece:
        print_piece_tree(model.root_piece, indent=2)

    total_verts = sum(len(p.vertices) for p in model.all_pieces())
    total_tris = sum(len(p.triangle_indices()) // 3 for p in model.all_pieces())
    print(f"  Total: {len(model.all_pieces())} pieces, {total_verts} vertices, {total_tris} triangles")

    out = s3o_file_to_glb(input_path, output_path)
    file_size = os.path.getsize(out)
    print(f"\nWritten: {out} ({file_size:,} bytes)")
