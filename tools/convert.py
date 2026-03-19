"""
BAR S3O → GLB Batch Converter with Weapon Metadata

Converts S3O models to GLB and embeds weapon-to-piece mappings
as glTF extras metadata on each node.

Usage:
  # Convert a single unit
  python convert.py --s3o objects3d/corjugg.s3o --script scripts/Units/corjugg.bos

  # Batch convert a BAR game directory
  python convert.py --bar-dir /path/to/Beyond-All-Reason --output-dir ./glb_output

  # Just parse and show info (no conversion)
  python convert.py --s3o objects3d/corjugg.s3o --info-only

The GLB output includes:
  - Full piece hierarchy with correct names and offsets
  - Weapon metadata in glTF node extras: {"weapons": [1, 2], "weapon_role": "fire_point"}
  - Model metadata in root node extras: {"texture1": "...", "texture2": "...", ...}

This metadata can be read in Three.js via:
  node.userData.weapons   → array of weapon numbers
  node.userData.weapon_role → "fire_point" | "aim_from" | "aim_piece"
"""

import os
import sys
import json
import struct
import argparse
import numpy as np
from typing import Dict, List, Optional, Tuple

from s3o_parser import parse_s3o, S3OModel, S3OPiece, print_piece_tree
from s3o_to_glb import GLBBuilder, convert_s3o_to_glb
from bos_parser import parse_unit_script, BOSParseResult, WeaponPieceMapping


def convert_with_weapons(
    model: S3OModel,
    weapon_info: Optional[BOSParseResult] = None
) -> bytes:
    """Convert S3O to GLB with weapon metadata embedded in glTF extras."""
    builder = GLBBuilder()
    mat_idx = builder.add_default_material()

    # Build weapon lookup: piece_name → weapon info
    weapon_lookup: Dict[str, dict] = {}
    if weapon_info:
        for wnum, wmap in weapon_info.weapons.items():
            if wmap.query_piece:
                key = wmap.query_piece.lower()
                if key not in weapon_lookup:
                    weapon_lookup[key] = {"weapons": [], "roles": []}
                weapon_lookup[key]["weapons"].append(wnum)
                if "fire_point" not in weapon_lookup[key]["roles"]:
                    weapon_lookup[key]["roles"].append("fire_point")

            if wmap.aim_from_piece:
                key = wmap.aim_from_piece.lower()
                if key not in weapon_lookup:
                    weapon_lookup[key] = {"weapons": [], "roles": []}
                if wnum not in weapon_lookup[key]["weapons"]:
                    weapon_lookup[key]["weapons"].append(wnum)
                if "aim_from" not in weapon_lookup[key]["roles"]:
                    weapon_lookup[key]["roles"].append("aim_from")

            for ap in wmap.aim_pieces:
                key = ap.lower()
                if key not in weapon_lookup:
                    weapon_lookup[key] = {"weapons": [], "roles": []}
                if wnum not in weapon_lookup[key]["weapons"]:
                    weapon_lookup[key]["weapons"].append(wnum)
                if "aim_piece" not in weapon_lookup[key]["roles"]:
                    weapon_lookup[key]["roles"].append("aim_piece")

    def add_piece_with_extras(piece: S3OPiece, parent_idx=None) -> int:
        """Add a piece node with weapon extras metadata."""
        mesh_idx = builder.add_piece_mesh(piece, mat_idx)

        node = {"name": piece.name}
        ox, oy, oz = piece.offset
        if ox != 0 or oy != 0 or oz != 0:
            node["translation"] = [ox, oy, oz]
        if mesh_idx is not None:
            node["mesh"] = mesh_idx

        # Add weapon extras
        piece_key = piece.name.lower()
        extras = {}
        if piece_key in weapon_lookup:
            winfo = weapon_lookup[piece_key]
            extras["weapons"] = sorted(winfo["weapons"])
            extras["weapon_roles"] = winfo["roles"]

        if extras:
            node["extras"] = extras

        node_idx = len(builder.nodes)
        builder.nodes.append(node)

        child_indices = []
        for child in piece.children:
            child_idx = add_piece_with_extras(child, node_idx)
            child_indices.append(child_idx)
        if child_indices:
            builder.nodes[node_idx]["children"] = child_indices

        return node_idx

    if model.root_piece:
        root_idx = add_piece_with_extras(model.root_piece)

        # Add model-level metadata to root node extras
        root_extras = builder.nodes[root_idx].get("extras", {})
        root_extras["s3o_texture1"] = model.texture1
        root_extras["s3o_texture2"] = model.texture2
        root_extras["s3o_radius"] = model.radius
        root_extras["s3o_height"] = model.height
        root_extras["s3o_midpoint"] = list(model.midpoint)
        if weapon_info and weapon_info.weapons:
            root_extras["weapon_count"] = len(weapon_info.weapons)
            root_extras["weapon_summary"] = {
                str(wnum): {
                    "fire_point": wmap.query_piece,
                    "aim_from": wmap.aim_from_piece,
                    "aim_pieces": wmap.aim_pieces,
                }
                for wnum, wmap in weapon_info.weapons.items()
            }
        builder.nodes[root_idx]["extras"] = root_extras

        builder.scenes[0]["nodes"] = [root_idx]

    return builder.build_glb()


def find_script_for_unit(bar_dir: str, unit_name: str) -> Optional[str]:
    """Find the BOS or Lua script for a given unit in the BAR directory."""
    scripts_dir = os.path.join(bar_dir, 'scripts', 'Units')
    if not os.path.isdir(scripts_dir):
        scripts_dir = os.path.join(bar_dir, 'scripts')

    # Try .bos first, then .lua
    for ext in ['.bos', '.lua']:
        path = os.path.join(scripts_dir, unit_name + ext)
        if os.path.isfile(path):
            return path

    # Some units use a different script name via unitdef
    # We could parse the unitdef, but for now just check common patterns
    return None


def convert_single(s3o_path: str, script_path: Optional[str] = None,
                   output_path: Optional[str] = None,
                   info_only: bool = False) -> Optional[str]:
    """Convert a single S3O file to GLB."""
    model = parse_s3o(s3o_path)
    unit_name = os.path.splitext(os.path.basename(s3o_path))[0]

    print(f"\n{'='*60}")
    print(f"Unit: {unit_name}")
    print(f"{'='*60}")
    print(f"  S3O Version: {model.version}")
    print(f"  Radius: {model.radius:.2f}, Height: {model.height:.2f}")
    print(f"  Texture 1: {model.texture1}")
    print(f"  Texture 2: {model.texture2}")
    print(f"  Piece tree:")
    if model.root_piece:
        print_piece_tree(model.root_piece, indent=2)

    total_verts = sum(len(p.vertices) for p in model.all_pieces())
    total_tris = sum(len(p.triangle_indices()) // 3 for p in model.all_pieces())
    print(f"  Total: {len(model.all_pieces())} pieces, {total_verts} verts, {total_tris} tris")

    weapon_info = None
    if script_path and os.path.isfile(script_path):
        print(f"\n  Script: {script_path}")
        weapon_info = parse_unit_script(script_path)
        weapon_info.print_summary()

    if info_only:
        return None

    if output_path is None:
        output_path = os.path.splitext(s3o_path)[0] + '.glb'

    glb_data = convert_with_weapons(model, weapon_info)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(glb_data)

    print(f"\n  → GLB written: {output_path} ({len(glb_data):,} bytes)")
    return output_path


def batch_convert(bar_dir: str, output_dir: str, unit_filter: str = None):
    """Batch convert all S3O files in a BAR game directory."""
    objects_dir = os.path.join(bar_dir, 'objects3d')
    if not os.path.isdir(objects_dir):
        print(f"Error: objects3d directory not found at {objects_dir}")
        return

    s3o_files = sorted([
        f for f in os.listdir(objects_dir)
        if f.endswith('.s3o')
    ])

    if unit_filter:
        s3o_files = [f for f in s3o_files if unit_filter.lower() in f.lower()]

    print(f"Found {len(s3o_files)} S3O files to convert")
    os.makedirs(output_dir, exist_ok=True)

    success, failed = 0, 0
    for filename in s3o_files:
        unit_name = os.path.splitext(filename)[0]
        s3o_path = os.path.join(objects_dir, filename)
        glb_path = os.path.join(output_dir, unit_name + '.glb')
        script_path = find_script_for_unit(bar_dir, unit_name)

        try:
            convert_single(s3o_path, script_path, glb_path)
            success += 1
        except Exception as e:
            print(f"  ERROR converting {filename}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Batch conversion complete: {success} success, {failed} failed")


def main():
    parser = argparse.ArgumentParser(
        description="BAR S3O → GLB Converter with Weapon Metadata"
    )
    parser.add_argument('--s3o', help='Path to a single .s3o file')
    parser.add_argument('--script', help='Path to the .bos or .lua script file')
    parser.add_argument('--output', '-o', help='Output .glb path')
    parser.add_argument('--bar-dir', help='BAR game directory for batch conversion')
    parser.add_argument('--output-dir', default='./glb_output',
                        help='Output directory for batch conversion')
    parser.add_argument('--filter', help='Unit name filter for batch mode')
    parser.add_argument('--info-only', action='store_true',
                        help='Only show info, do not convert')

    args = parser.parse_args()

    if args.bar_dir:
        batch_convert(args.bar_dir, args.output_dir, args.filter)
    elif args.s3o:
        convert_single(args.s3o, args.script, args.output, args.info_only)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
