"""
S3O Parser — reads the Spring RTS Engine S3O binary model format.

S3O Binary Layout (all little-endian):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Header (52 bytes):
  magic           : 12 bytes  "Spring3DO\0\0\0"
  version         : uint32
  radius          : float     bounding sphere radius
  height          : float     model height
  midx,midy,midz  : 3×float   midpoint (center of bounding sphere)
  rootPieceOffset : uint32    offset to root piece
  collisionData   : uint32    offset to collision data (unused here)
  tex1Offset      : uint32    offset to texture1 name (null-terminated)
  tex2Offset      : uint32    offset to texture2 name (null-terminated)

Piece (52 bytes):
  nameOffset      : uint32    offset to piece name (null-terminated)
  numChildren     : uint32    number of child pieces
  childrenOffset  : uint32    offset to array of uint32 child piece offsets
  numVertices     : uint32    vertex count
  verticesOffset  : uint32    offset to vertex data
  vertexType      : uint32    0 = standard S3O vertex
  primitiveType   : uint32    0 = triangles, 1 = tri-strips, 2 = quads
  numIndices      : uint32    number of vertex indices
  indicesOffset   : uint32    offset to index data (uint32 each)
  collisionData   : uint32    offset to collision data (unused)
  xoffset         : float     piece origin offset X
  yoffset         : float     piece origin offset Y
  zoffset         : float     piece origin offset Z

Vertex (32 bytes):
  x, y, z         : 3×float   position
  nx, ny, nz      : 3×float   normal
  s, t            : 2×float   UV coordinates
"""

import struct
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class S3OVertex:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    nx: float = 0.0
    ny: float = 1.0
    nz: float = 0.0
    s: float = 0.0
    t: float = 0.0


@dataclass
class S3OPiece:
    name: str = ""
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    vertices: List[S3OVertex] = field(default_factory=list)
    indices: List[int] = field(default_factory=list)
    primitive_type: int = 0  # 0=triangles, 1=tri-strips, 2=quads
    vertex_type: int = 0
    children: List['S3OPiece'] = field(default_factory=list)

    def triangle_indices(self) -> List[int]:
        """Convert any primitive type to triangle indices."""
        if self.primitive_type == 0:
            # Already triangles
            return list(self.indices)
        elif self.primitive_type == 1:
            # Triangle strip → triangles
            tris = []
            for i in range(len(self.indices) - 2):
                if i % 2 == 0:
                    tris.extend([self.indices[i], self.indices[i+1], self.indices[i+2]])
                else:
                    tris.extend([self.indices[i+1], self.indices[i], self.indices[i+2]])
            return tris
        elif self.primitive_type == 2:
            # Quads → triangles
            tris = []
            for i in range(0, len(self.indices) - 3, 4):
                a, b, c, d = self.indices[i], self.indices[i+1], self.indices[i+2], self.indices[i+3]
                tris.extend([a, b, c, a, c, d])
            return tris
        return list(self.indices)

    def all_pieces_flat(self) -> List['S3OPiece']:
        """Return this piece and all descendants in a flat list."""
        result = [self]
        for child in self.children:
            result.extend(child.all_pieces_flat())
        return result


@dataclass
class S3OModel:
    version: int = 0
    radius: float = 0.0
    height: float = 0.0
    midpoint: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    texture1: str = ""
    texture2: str = ""
    root_piece: Optional[S3OPiece] = None

    def all_pieces(self) -> List[S3OPiece]:
        if self.root_piece is None:
            return []
        return self.root_piece.all_pieces_flat()

    def piece_names(self) -> List[str]:
        return [p.name for p in self.all_pieces()]


def _read_string(data: bytes, offset: int) -> str:
    """Read a null-terminated string from the data buffer."""
    if offset == 0 or offset >= len(data):
        return ""
    end = data.index(b'\x00', offset)
    return data[offset:end].decode('ascii', errors='replace')


def _read_piece(data: bytes, offset: int) -> S3OPiece:
    """Recursively read an S3O piece and its children from the data buffer."""
    if offset == 0 or offset + 52 > len(data):
        return S3OPiece(name="<invalid>")

    # Piece header: 10 × uint32 + 3 × float = 52 bytes
    vals = struct.unpack_from('<10I3f', data, offset)
    name_off = vals[0]
    num_children = vals[1]
    children_off = vals[2]
    num_vertices = vals[3]
    vertices_off = vals[4]
    vertex_type = vals[5]
    primitive_type = vals[6]
    num_indices = vals[7]
    indices_off = vals[8]
    # vals[9] = collision data offset (unused)
    xoff, yoff, zoff = vals[10], vals[11], vals[12]

    piece = S3OPiece(
        name=_read_string(data, name_off),
        offset=(xoff, yoff, zoff),
        vertex_type=vertex_type,
        primitive_type=primitive_type,
    )

    # Read vertices (32 bytes each: 3f pos + 3f normal + 2f uv)
    if num_vertices > 0 and vertices_off > 0:
        for i in range(num_vertices):
            voff = vertices_off + i * 32
            if voff + 32 > len(data):
                break
            vx, vy, vz, nx, ny, nz, s, t = struct.unpack_from('<8f', data, voff)
            piece.vertices.append(S3OVertex(vx, vy, vz, nx, ny, nz, s, t))

    # Read indices (4 bytes each, uint32)
    if num_indices > 0 and indices_off > 0:
        for i in range(num_indices):
            ioff = indices_off + i * 4
            if ioff + 4 > len(data):
                break
            (idx,) = struct.unpack_from('<I', data, ioff)
            piece.indices.append(idx)

    # Read children (offsets stored as uint32 array)
    if num_children > 0 and children_off > 0:
        for i in range(num_children):
            coff_pos = children_off + i * 4
            if coff_pos + 4 > len(data):
                break
            (child_off,) = struct.unpack_from('<I', data, coff_pos)
            child = _read_piece(data, child_off)
            piece.children.append(child)

    return piece


def parse_s3o(filepath: str) -> S3OModel:
    """Parse an S3O file and return an S3OModel."""
    with open(filepath, 'rb') as f:
        data = f.read()

    if len(data) < 52:
        raise ValueError(f"File too small to be a valid S3O: {len(data)} bytes")

    # Header
    magic = data[0:12]
    valid_magics = [b'Spring3DO\x00', b'Spring unit\x00']
    if not any(magic.startswith(m) for m in valid_magics):
        raise ValueError(f"Invalid S3O magic: {magic!r} (expected 'Spring3DO' or 'Spring unit')")

    (version,) = struct.unpack_from('<I', data, 12)
    (radius,) = struct.unpack_from('<f', data, 16)
    (height,) = struct.unpack_from('<f', data, 20)
    midx, midy, midz = struct.unpack_from('<3f', data, 24)
    (root_off,) = struct.unpack_from('<I', data, 36)
    # collision data offset at 40 (unused)
    (tex1_off,) = struct.unpack_from('<I', data, 44)
    (tex2_off,) = struct.unpack_from('<I', data, 48)

    model = S3OModel(
        version=version,
        radius=radius,
        height=height,
        midpoint=(midx, midy, midz),
        texture1=_read_string(data, tex1_off),
        texture2=_read_string(data, tex2_off),
    )

    if root_off > 0:
        model.root_piece = _read_piece(data, root_off)

    return model


def print_piece_tree(piece: S3OPiece, indent: int = 0):
    """Print the piece hierarchy for debugging."""
    prefix = "  " * indent
    n_tris = len(piece.triangle_indices()) // 3
    print(f"{prefix}├── {piece.name}  "
          f"(verts={len(piece.vertices)}, tris={n_tris}, "
          f"offset=({piece.offset[0]:.1f}, {piece.offset[1]:.1f}, {piece.offset[2]:.1f}))")
    for child in piece.children:
        print_piece_tree(child, indent + 1)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python s3o_parser.py <file.s3o>")
        sys.exit(1)

    model = parse_s3o(sys.argv[1])
    print(f"S3O Model v{model.version}")
    print(f"  Radius: {model.radius:.2f}, Height: {model.height:.2f}")
    print(f"  Midpoint: ({model.midpoint[0]:.2f}, {model.midpoint[1]:.2f}, {model.midpoint[2]:.2f})")
    print(f"  Texture 1: {model.texture1}")
    print(f"  Texture 2: {model.texture2}")
    print(f"  Pieces:")
    if model.root_piece:
        print_piece_tree(model.root_piece)
