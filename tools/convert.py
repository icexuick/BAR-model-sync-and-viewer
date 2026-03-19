"""
BAR S3O → GLB Batch Converter with Weapon Metadata

Converts S3O models to GLB and embeds weapon-to-piece mappings
as glTF extras metadata on each node.

Usage:
  # Convert by unit name — fetches files automatically from GitHub
  python convert.py --unit corjugg

  # Convert a single unit from local files
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
import re
import sys
import json
import struct
import argparse
import tempfile
import base64
import hashlib
import urllib.request
import numpy as np
from typing import Dict, List, Optional, Tuple

from s3o_parser import parse_s3o, S3OModel, S3OPiece, print_piece_tree
from s3o_to_glb import GLBBuilder, convert_s3o_to_glb
from bos_parser import parse_unit_script, BOSParseResult, WeaponPieceMapping
from bos_animator import extract_walk_animation


def convert_with_weapons(
    model: S3OModel,
    weapon_info: Optional[BOSParseResult] = None,
    script_path: Optional[str] = None,
) -> bytes:
    """Convert S3O to GLB with weapon metadata and walk animation."""
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

    # Maps piece_name.lower() → glTF node index (built while adding pieces)
    node_name_to_idx: Dict[str, int] = {}
    # Maps piece_name.lower() → S3O rest offset (x, y, z)
    piece_offsets: Dict[str, tuple] = {}

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

        # Track name → index and rest offset for animation
        node_name_to_idx[piece_key] = node_idx
        piece_offsets[piece_key] = (ox, oy, oz)

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

    # --- Animation ---
    if script_path and os.path.isfile(script_path):
        try:
            with open(script_path, 'r', errors='replace') as f:
                bos_content = f.read()
            result = extract_walk_animation(bos_content)
            if result:
                anim_name, tracks = result
                builder.add_animation(anim_name, tracks, node_name_to_idx, piece_offsets)
        except Exception as e:
            print(f"  Warning: animation extraction failed: {e}")

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

    glb_data = convert_with_weapons(model, weapon_info, script_path)
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


BAR_RAW = "https://github.com/beyond-all-reason/Beyond-All-Reason/raw/refs/heads/master"
BAR_API = "https://api.github.com/repos/beyond-all-reason/Beyond-All-Reason"

# Load .env file from the same directory as this script (if it exists)
_ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.isfile(_ENV_FILE):
    with open(_ENV_FILE) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))


def _github_headers() -> dict:
    """Build GitHub API request headers, including token if GITHUB_TOKEN is set."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "BAR-modelviewer",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _github_get(url: str) -> dict:
    """GET a GitHub API URL and return parsed JSON."""
    req = urllib.request.Request(url, headers=_github_headers())
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 403:
            print("\nGitHub API rate limit exceeded (60 req/hour without auth).")
            print("Fix: create a free token at https://github.com/settings/tokens")
            print("Then run:  set GITHUB_TOKEN=your_token_here  (Windows)")
            print("       or: export GITHUB_TOKEN=your_token_here  (Linux/Mac)\n")
        raise


def _download(url: str, dest: str):
    """Download a URL to dest, raising on HTTP errors."""
    url = url.replace(" ", "%20")
    req = urllib.request.Request(url, headers={"User-Agent": "BAR-modelviewer"})
    with urllib.request.urlopen(req) as resp:
        with open(dest, 'wb') as f:
            f.write(resp.read())


def _find_unit_lua_path(unit_name: str) -> Optional[str]:
    """
    Find the path of {unit_name}.lua in the BAR units/ directory tree.
    Uses the git trees API to list all files under units/ in 3 requests (no auth needed).
    """
    target = f"{unit_name.lower()}.lua"

    # 1. Get root tree SHA from latest master commit
    commit = _github_get(f"{BAR_API}/commits/master")
    root_tree_sha = commit["commit"]["tree"]["sha"]

    # 2. Find the 'units' subtree SHA in the root tree
    root_tree = _github_get(f"{BAR_API}/git/trees/{root_tree_sha}")
    units_entry = next((e for e in root_tree["tree"] if e["path"] == "units"), None)
    if not units_entry:
        print("  Error: 'units' directory not found in repo root tree")
        return None

    # 3. Recursively list all files under units/
    units_tree = _github_get(f"{BAR_API}/git/trees/{units_entry['sha']}?recursive=1")
    for entry in units_tree["tree"]:
        if entry["type"] == "blob" and entry["path"].lower().endswith("/" + target):
            return f"units/{entry['path']}"
        if entry["type"] == "blob" and entry["path"].lower() == target:
            return f"units/{entry['path']}"

    return None


def fetch_unit_from_github(unit_name: str, output_path: Optional[str] = None,
                            info_only: bool = False,
                            push: bool = False) -> Optional[str]:
    """
    Look up a BAR unit by name in the GitHub repo, download its S3O and script,
    and convert to GLB.

    Steps:
      1. Browse units/ via Contents API → find {unit_name}.lua
      2. Parse lua for objectName (S3O) and script (BOS/Lua)
      3. Download S3O from objects3d/ and script from scripts/Units/
      4. Convert with weapon metadata
    """
    print(f"Searching GitHub for unit: {unit_name}")

    # 1. Find the unit .lua file by browsing units/ subdirectories
    unit_lua_path = _find_unit_lua_path(unit_name)
    if not unit_lua_path:
        print(f"Error: could not find {unit_name}.lua in the BAR units/ directory")
        return None

    print(f"  Found unit def: {unit_lua_path}")

    # 2. Download and parse the unit lua file
    lua_url = f"{BAR_RAW}/{unit_lua_path}".replace(" ", "%20")
    req = urllib.request.Request(lua_url, headers={"User-Agent": "BAR-modelviewer"})
    with urllib.request.urlopen(req) as resp:
        lua_content = resp.read().decode("utf-8", errors="replace")

    obj_match = re.search(r'objectName\s*=\s*["\']([^"\']+\.s3o)["\']', lua_content, re.IGNORECASE)
    script_match = re.search(r'\bscript\s*=\s*["\']([^"\']+)["\']', lua_content, re.IGNORECASE)

    # objectName may contain a subpath like "Units/CORJUGG.s3o"
    # Keep directory casing as-is, only lowercase the filename
    s3o_raw = obj_match.group(1) if obj_match else f"{unit_name}.s3o"
    s3o_parts = s3o_raw.replace("\\", "/").split("/")
    s3o_parts[-1] = s3o_parts[-1].lower()
    s3o_subpath = "/".join(s3o_parts)                  # e.g. "Units/corjugg.s3o"
    s3o_name = s3o_parts[-1]                           # e.g. "corjugg.s3o"

    script_raw = script_match.group(1) if script_match else f"{unit_name}.bos"
    script_parts = script_raw.replace("\\", "/").split("/")
    script_parts[-1] = script_parts[-1].lower()
    script_base = script_parts[-1]
    # .cob is compiled; replace with .bos source extension
    script_base = re.sub(r'\.cob$', '.bos', script_base)

    print(f"  S3O model : {s3o_subpath}  (parsed: {s3o_raw!r})")
    print(f"  Script    : {script_base}  (parsed: {script_raw!r})")

    # 3. Download files to a temp directory, then convert
    with tempfile.TemporaryDirectory() as tmpdir:
        s3o_local = os.path.join(tmpdir, s3o_name)
        script_local = os.path.join(tmpdir, script_base)

        s3o_url = f"{BAR_RAW}/objects3d/{s3o_subpath}"
        print(f"  Downloading {s3o_url} ...")
        _download(s3o_url, s3o_local)

        script_ok = False
        # Try scripts/Units/ first, then scripts/
        for script_subpath_try in [f"scripts/Units/{script_base}", f"scripts/{script_base}"]:
            try:
                script_url = f"{BAR_RAW}/{script_subpath_try}"
                print(f"  Downloading {script_url} ...")
                _download(script_url, script_local)
                script_ok = True
                break
            except Exception:
                pass
        if not script_ok:
            print("  Warning: script not found, converting without weapon metadata")
            script_local = None

        # Output inside the temp dir so it never lands in the repo locally.
        # push_glb_to_repo() uploads it to GitHub; after that it's cleaned up.
        # Use --local to save to disk instead.
        if output_path is None:
            output_path = os.path.join(tmpdir, s3o_name.replace(".s3o", ".glb"))

        glb_path = convert_single(s3o_local, script_local, output_path, info_only)
        if glb_path and push and not info_only:
            push_glb_to_repo(glb_path)
        return glb_path


VIEWER_REPO = "icexuick/BAR-modelviewer"


def push_glb_to_repo(glb_path: str):
    """Upload or overwrite a GLB file in the BAR-modelviewer GitHub repo."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("  Warning: GITHUB_TOKEN not set, skipping push to repo")
        return

    filename = os.path.basename(glb_path)
    repo_path = filename  # GLBs live in the repo root

    with open(glb_path, "rb") as f:
        raw = f.read()

    api_url = f"https://api.github.com/repos/{VIEWER_REPO}/contents/{repo_path}"

    # Check if file already exists (need its SHA to overwrite)
    existing_sha = None
    try:
        existing = _github_get(api_url)
        existing_sha = existing["sha"]
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise

    # Compare git blob SHA to avoid empty commits when content is unchanged
    blob_sha = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if existing_sha and blob_sha == existing_sha:
        print(f"  → No changes, skipping push ({filename} already up to date)")
        return

    content_b64 = base64.b64encode(raw).decode("ascii")
    body = {
        "message": f"{'Update' if existing_sha else 'Add'} {filename}",
        "content": content_b64,
    }
    if existing_sha:
        body["sha"] = existing_sha

    req = urllib.request.Request(
        api_url,
        data=json.dumps(body).encode("utf-8"),
        headers={**_github_headers(), "Content-Type": "application/json"},
        method="PUT",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"\n  Push failed (404). Your GITHUB_TOKEN likely needs 'public_repo' scope.")
            print(f"  Regenerate it at https://github.com/settings/tokens and tick 'public_repo'.")
        elif e.code == 422:
            print(f"\n  Push failed (422 Unprocessable). The SHA for the existing file may be stale — try again.")
        else:
            print(f"\n  Push failed (HTTP {e.code}): {e.reason}")
        return

    action = "Updated" if existing_sha else "Created"
    print(f"  → {action} in repo: https://github.com/{VIEWER_REPO}/blob/main/{repo_path}")


def main():
    parser = argparse.ArgumentParser(
        description="BAR S3O → GLB Converter with Weapon Metadata"
    )
    parser.add_argument('--unit', help='Unit name (fetches files automatically from GitHub)')
    parser.add_argument('--s3o', help='Path to a single .s3o file')
    parser.add_argument('--script', help='Path to the .bos or .lua script file')
    parser.add_argument('--output', '-o', help='Output .glb path')
    parser.add_argument('--bar-dir', help='BAR game directory for batch conversion')
    parser.add_argument('--output-dir', default='./glb_output',
                        help='Output directory for batch conversion')
    parser.add_argument('--filter', help='Unit name filter for batch mode')
    parser.add_argument('--info-only', action='store_true',
                        help='Only show info, do not convert')
    parser.add_argument('--local', action='store_true',
                        help='Write GLB locally only, do not push to GitHub repo')

    args = parser.parse_args()

    if args.unit:
        if args.local:
            # --local: save GLB to repo root so the user can inspect it
            if args.output is None:
                repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                args.output = os.path.join(repo_root, f"{args.unit}.glb")
        fetch_unit_from_github(
            args.unit, args.output, args.info_only,
            push=not args.local and not args.info_only,
        )
    elif args.bar_dir:
        batch_convert(args.bar_dir, args.output_dir, args.filter)
    elif args.s3o:
        convert_single(args.s3o, args.script, args.output, args.info_only)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
