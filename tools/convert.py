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
from bos_animator import extract_walk_animation, extract_spin_animation, parse_create_now_rotations, parse_create_hide_pieces, extract_stopwalking_pose, extract_activate_loop_animation, extract_toggle_animations




def parse_lua_weapon_defs(lua_content: str) -> Dict[int, str]:
    """
    Extract weapon def names from a unit Lua file.
    Returns {weapon_num: def_name_lowercase}, e.g. {1: "corkorg_fire", 2: "corkorg_laser"}.
    Parses: weapons = { [1] = { def = "NAME", ... }, [2] = { ... } }
    """
    result: Dict[int, str] = {}
    # Strip Lua line comments (-- to end of line) so commented-out weapon slots
    # (e.g. --[2] = { def = "..." }) are not mistakenly parsed as active weapons.
    lua_content = re.sub(r'--[^\n]*', '', lua_content)
    # Find the weapons = { ... } block
    m = re.search(r'\bweapons\s*=\s*\{', lua_content, re.IGNORECASE)
    if not m:
        return result
    # Walk braces to find the full block
    start = m.end() - 1
    depth = 0
    pos = start
    while pos < len(lua_content):
        if lua_content[pos] == '{':
            depth += 1
        elif lua_content[pos] == '}':
            depth -= 1
            if depth == 0:
                break
        pos += 1
    weapons_block = lua_content[start:pos + 1]
    # Find each [N] = { def = "NAME" }
    for entry in re.finditer(r'\[(\d+)\]\s*=\s*\{([^}]*)\}', weapons_block, re.DOTALL):
        wnum = int(entry.group(1))
        body = entry.group(2)
        def_m = re.search(r'\bdef\s*=\s*["\']?(\w+)["\']?', body, re.IGNORECASE)
        if def_m:
            result[wnum] = def_m.group(1).lower()
    return result


def parse_lua_unit_role(lua_content: str) -> Optional[str]:
    """
    Detect the unit's role from unitdef Lua fields.
    Returns one of: 'RADAR', 'JAMMER', 'SONAR', 'RADAR_JAMMER', 'MEX', or None.
    """
    has_radar  = bool(re.search(r'\bradardistance\s*=\s*[1-9]', lua_content, re.IGNORECASE))
    has_jammer = bool(re.search(r'\bradardistancejam\s*=\s*[1-9]', lua_content, re.IGNORECASE))
    has_sonar  = bool(re.search(r'\bsonardistance\s*=\s*[1-9]', lua_content, re.IGNORECASE))
    has_mex    = bool(re.search(r'\bextractsmetal\s*=\s*[0-9]*\.[0-9]*[1-9]', lua_content, re.IGNORECASE))
    if has_radar and has_jammer:
        return 'RADAR_JAMMER'
    if has_jammer:
        return 'JAMMER'
    if has_radar:
        return 'RADAR'
    if has_sonar:
        return 'SONAR'
    if has_mex:
        return 'MEX'
    return None


def _build_piece_maps(root_piece: 'S3OPiece') -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Build piece parent and children maps from S3O hierarchy.
    Returns:
      parent_map:   {piece_name.lower() → parent_name.lower() or None}
      children_map: {piece_name.lower() → [child_name.lower(), ...]}
    """
    parent_map: Dict[str, Optional[str]] = {}
    children_map: Dict[str, List[str]] = {}

    def walk(piece, parent_key):
        key = piece.name.lower()
        parent_map[key] = parent_key
        children_map[key] = [c.name.lower() for c in piece.children]
        for child in piece.children:
            walk(child, key)

    walk(root_piece, None)
    return parent_map, children_map


def _collect_subtree(piece_key: str, children_map: Dict[str, List[str]]) -> List[str]:
    """Return all piece keys in the subtree rooted at piece_key (inclusive)."""
    result = []
    stack = [piece_key]
    while stack:
        cur = stack.pop()
        result.append(cur)
        stack.extend(children_map.get(cur, []))
    return result


# Per-unit pieces whose translation tracks should be stripped from animations.
# Useful for units where body sway/sliding looks wrong in the viewer.
_STRIP_ANIM_TRANSLATION: Dict[str, set] = {
    'corsktl': {'base'},
}

# Per-unit pieces whose rotation tracks should be stripped from animations.
# Useful for units where body twisting/spinning looks wrong in the viewer.
_STRIP_ANIM_ROTATION: Dict[str, set] = {
    'corsktl': {'base'},
}

# Per-unit target duration (seconds) for walk animations.
# All keyframe times are scaled proportionally so the loop matches this duration.
# Use this when the BOS sleep values give an unrealistic playback speed.
_ANIM_DURATION_OVERRIDE: Dict[str, float] = {
    'cortermite': 0.6,
}


def convert_with_weapons(
    model: S3OModel,
    weapon_info: Optional[BOSParseResult] = None,
    script_path: Optional[str] = None,
    weapon_defs: Optional[Dict[int, str]] = None,
    hide_pieces: Optional[set] = None,
    unit_role: Optional[str] = None,
    unit_name: str = '',
) -> bytes:
    """Convert S3O to GLB with weapon metadata, walk/spin animation, and unit role."""
    builder = GLBBuilder()
    mat_idx = builder.add_default_material()

    # Build piece hierarchy maps for visual weapon root detection
    parent_map: Dict[str, Optional[str]] = {}
    children_map: Dict[str, List[str]] = {}
    piece_vert_count: Dict[str, int] = {}  # piece_key → number of vertices (own mesh only)
    if model.root_piece:
        parent_map, children_map = _build_piece_maps(model.root_piece)
        for p in model.all_pieces():
            piece_vert_count[p.name.lower()] = len(p.vertices)

    # Build weapon lookup: piece_name → weapon info
    # Strategy:
    #   - aim_pieces: tagged as "aim_piece" (structural, used for animation targeting)
    #   - fire_point: tagged as "fire_point"
    #   - aim_from: tagged as "aim_from"
    #   - visual weapon subtree: for each weapon, walk from fire_point up the hierarchy
    #     to find the nearest aim_piece ancestor → mark its non-aim-piece descendants
    #     as "visual" so the viewer can highlight the actual gun geometry.
    weapon_lookup: Dict[str, dict] = {}

    def _add_to_lookup(key: str, wnum: int, role: str):
        if key not in weapon_lookup:
            weapon_lookup[key] = {"weapons": [], "roles": []}
        if wnum not in weapon_lookup[key]["weapons"]:
            weapon_lookup[key]["weapons"].append(wnum)
        if role not in weapon_lookup[key]["roles"]:
            weapon_lookup[key]["roles"].append(role)

    if weapon_info:
        # If weapon_defs is available (from unitdef Lua), filter BOS weapons to only
        # those that actually exist in the unitdef. BOS scripts sometimes use higher
        # weapon slots internally (e.g. QueryWeapon2 for a flare when only weapon 1
        # exists in the unitdef). Drop any BOS wnum not present in weapon_defs.
        # Weapon def names that indicate a fake/internal weapon — no geometry to highlight.
        # These are used by the engine for targeting or special logic, not actual guns.
        _DUMMY_DEF_KEYWORDS = {'targeting', 'dummy', 'aim_weapon', 'aimweapon', 'scanner', 'aimhull', 'bogus'}

        def _is_dummy_def(def_name: str) -> bool:
            if not def_name:
                return False
            low = def_name.lower()
            return any(kw in low for kw in _DUMMY_DEF_KEYWORDS)

        # Piece name keywords that indicate targeting/dummy geometry — never visual gun parts.
        _DUMMY_PIECE_KEYWORDS = {'targeting', 'target', 'dummy', 'scanner'}

        def _is_dummy_piece(piece_key: str) -> bool:
            low = piece_key.lower()
            return any(kw in low for kw in _DUMMY_PIECE_KEYWORDS)

        if weapon_defs:
            # Drop any BOS weapon number not present in the unitdef weapons table.
            weapon_info.weapons = {wn: wm for wn, wm in weapon_info.weapons.items()
                                   if wn in weapon_defs}
            # Also drop weapons whose def is a known dummy/targeting type.
            weapon_info.weapons = {wn: wm for wn, wm in weapon_info.weapons.items()
                                   if not _is_dummy_def(weapon_defs.get(wn, ''))}
            # Add empty entries for real weapons present in unitdef but absent in BOS
            # (e.g. anti-nuke launchers that have no QueryWeapon function).
            from bos_parser import WeaponPieceMapping
            for wn, def_name in weapon_defs.items():
                if wn not in weapon_info.weapons and not _is_dummy_def(def_name):
                    weapon_info.weapons[wn] = WeaponPieceMapping(weapon_num=wn)

        # Collect all aim_piece keys across all weapons (for exclusion logic below)
        all_aim_pieces: set = set()
        for wmap in weapon_info.weapons.values():
            for ap in wmap.aim_pieces:
                all_aim_pieces.add(ap.lower())

        for wnum, wmap in weapon_info.weapons.items():
            if wmap.query_piece:
                _add_to_lookup(wmap.query_piece.lower(), wnum, "fire_point")

            if wmap.aim_from_piece:
                _add_to_lookup(wmap.aim_from_piece.lower(), wnum, "aim_from")

            for ap in wmap.aim_pieces:
                _add_to_lookup(ap.lower(), wnum, "aim_piece")

            # Find the visual weapon root: highest aim_piece ancestor of the fire_point.
            # "Highest" = furthest from the fire_point but still in aim_set — this captures
            # the full gun mount (e.g. lshoulder) rather than just an inner pivot (lsleeve).
            # If no aim_pieces, use aim_from_piece or direct parent of fire_point.
            # Fallback for no fire_point but has aim_pieces: use highest aim_piece ≤30%.
            if not wmap.query_piece and wmap.aim_pieces:
                aim_set = {ap.lower() for ap in wmap.aim_pieces}
                total_pieces = len(parent_map)
                other_aim_pieces = {
                    ap.lower()
                    for wn2, wm2 in weapon_info.weapons.items()
                    if wn2 != wnum
                    for ap in wm2.aim_pieces
                }
                # Tag ALL aim_pieces whose subtree is ≤30% of the model as visual roots.
                # This handles dual-barrel units (sleeveTop + sleeveBottom both highlight).
                # Special case: l/r mirror pairs (gunl/gunr, finl/finr) where each piece
                # individually exceeds 30% on small models — if both mirrors are in aim_set,
                # accept them together (their combined subtree is the weapon assembly).
                def _lr_mirror(n):
                    if n.endswith('l'): return n[:-1] + 'r'
                    if n.endswith('r'): return n[:-1] + 'l'
                    if n.endswith('1'): return n[:-1] + '2'
                    if n.endswith('2'): return n[:-1] + '1'
                    if n.startswith('l'): return 'r' + n[1:]
                    if n.startswith('r'): return 'l' + n[1:]
                    return n

                visual_roots = []
                for ap_key in sorted(aim_set):  # sorted for determinism
                    sub = _collect_subtree(ap_key, children_map)
                    qualifies = (total_pieces == 0 or len(sub) <= total_pieces * 0.30)
                    if not qualifies:
                        # l/r mirror pair (gunl/gunr): accept if mirror is also in aim_set
                        # and neither piece individually is a large structural assembly (>50%).
                        mirror = _lr_mirror(ap_key)
                        if mirror != ap_key and mirror in aim_set:
                            qualifies = (total_pieces == 0 or len(sub) <= total_pieces * 0.50)
                    if qualifies:
                        visual_roots.append((ap_key, sub))
                # Fallback: try aim_from if no aim_piece qualifies
                if not visual_roots and wmap.aim_from_piece:
                    af_key = wmap.aim_from_piece.lower()
                    af_sub = _collect_subtree(af_key, children_map)
                    if total_pieces == 0 or len(af_sub) <= total_pieces * 0.50:
                        # Prefer aim_from only if it has real geometry — otherwise
                        # pick the smallest aim_piece (covers e.g. aimpoint on corsala)
                        af_verts = piece_vert_count.get(af_key, 0)
                        if af_verts > 10:
                            visual_roots.append((af_key, af_sub))
                        else:
                            # aim_from is a dummy/near-dummy — use smallest aim_piece
                            best_ap = min(aim_set, key=lambda k: len(_collect_subtree(k, children_map)))
                            best_sub = _collect_subtree(best_ap, children_map)
                            visual_roots.append((best_ap, best_sub))
                    else:
                        # aim_from subtree too large — use smallest aim_piece as last resort
                        best_ap = min(aim_set, key=lambda k: len(_collect_subtree(k, children_map)))
                        best_sub = _collect_subtree(best_ap, children_map)
                        visual_roots.append((best_ap, best_sub))
                visual_root_keys = {vr for vr, _ in visual_roots}
                for visual_root, subtree in visual_roots:
                    for piece_key in subtree:
                        if not _is_dummy_piece(piece_key) and (piece_key == visual_root or piece_key in visual_root_keys or piece_key not in other_aim_pieces):
                            _add_to_lookup(piece_key, wnum, "visual")
                if visual_roots:
                    roots_str = ', '.join(r for r, _ in visual_roots)
                    total_tagged = sum(len(s) for _, s in visual_roots)
                    print(f"  Weapon {wnum}: visual roots = [{roots_str}] (no fire_point), "
                          f"total tagged = {total_tagged}")

            if wmap.query_piece:
                aim_set = {ap.lower() for ap in wmap.aim_pieces}
                other_aim_pieces = {
                    ap.lower()
                    for wn2, wm2 in weapon_info.weapons.items()
                    if wn2 != wnum
                    for ap in wm2.aim_pieces
                }
                # Pieces that belong to OTHER weapons — used to stop ancestor-walk from
                # accidentally tagging shared pivots (e.g. aimx on legkark).
                # Include: aim_pieces of other weapons (clear ownership)
                #          query_pieces/aim_from ONLY if not shared with current weapon
                #          (prevents blocking shared turret bases like on legnavydestro)
                own_pieces_lower = {p.lower() for p in wmap.all_pieces}
                other_weapon_pieces = {
                    ap.lower()
                    for wn2, wm2 in weapon_info.weapons.items()
                    if wn2 != wnum
                    for ap in list(wm2.aim_pieces)  # aim_pieces are clearly owned
                } | {
                    ap.lower()
                    for wn2, wm2 in weapon_info.weapons.items()
                    if wn2 != wnum
                    for ap in ([wm2.query_piece] if wm2.query_piece else [])
                                + ([wm2.aim_from_piece] if wm2.aim_from_piece else [])
                    if ap.lower() not in own_pieces_lower  # only if not shared with us
                }

                def _mirror(name: str) -> str:
                    """Swap leading l/r prefix, trailing l/r suffix, or trailing 1/2 suffix."""
                    if name.startswith('l'): return 'r' + name[1:]
                    if name.startswith('r'): return 'l' + name[1:]
                    # Trailing l/r suffix: gunl ↔ gunr, finl ↔ finr, etc.
                    if name.endswith('l'): return name[:-1] + 'r'
                    if name.endswith('r'): return name[:-1] + 'l'
                    # Numbered siblings: barrel1 ↔ barrel2, gun1 ↔ gun2, etc.
                    if name.endswith('1'): return name[:-1] + '2'
                    if name.endswith('2'): return name[:-1] + '1'
                    return name

                # Piece-name fragments that mark structural limb joints on bipedal/
                # quadrupedal bots. A visual root must NOT be one of these pieces or
                # an ancestor of these pieces — selecting e.g. "ruparm" would pull in
                # the entire arm + torso as a weapon highlight.
                _LIMB_JOINT_KEYWORDS = ('uparm', 'upleg', 'thigh', 'shoulder',
                                        'torso', 'cockpit', 'body', 'hull', 'pelvis',
                                        'hip', 'chest', 'neck', 'head')

                def _is_limb_joint(name: str) -> bool:
                    low = name.lower()
                    return any(kw in low for kw in _LIMB_JOINT_KEYWORDS)

                def _subtree_has_limb_joint(root_key: str) -> bool:
                    """True if any piece in the subtree is a limb joint (uparm, torso, etc.)."""
                    return any(_is_limb_joint(k) for k in _collect_subtree(root_key, children_map)
                               if k != root_key)

                _STRUCTURAL_KEYWORDS = {'wing', 'leg', 'track', 'wheel', 'foot',
                                        'thruster', 'thrust', 'engine', 'body', 'hull',
                                        'chassis', 'torso', 'hip', 'armor',
                                        'plate', 'wake', 'bow', 'stern',
                                        'uparm', 'shoulder', 'thigh',
                                        'pelvis', 'chest', 'neck'}
                _STRUCTURAL_EXACT = {'base', 'pelvis', 'body', 'hull'}

                def _is_structural(name: str) -> bool:
                    low = name.lower()
                    if low in _STRUCTURAL_EXACT:
                        return True
                    return any(kw in low for kw in _STRUCTURAL_KEYWORDS)

                def _find_visual_root(fp_key: str) -> Optional[str]:
                    """Find the visual root for a given fire_point key."""
                    visual_root = None
                    if aim_set:
                        total_pieces = len(parent_map)
                        cur = parent_map.get(fp_key)
                        while cur is not None:
                            if cur in aim_set and not _is_dummy_piece(cur) and not _is_limb_joint(cur):
                                sub = _collect_subtree(cur, children_map)
                                # Reject if subtree contains structural body parts (e.g. aimx1
                                # on a biped whose subtree includes uparm/torso/etc.)
                                if not _subtree_has_limb_joint(cur):
                                    if total_pieces == 0 or len(sub) <= total_pieces * 0.50:
                                        visual_root = cur
                            cur = parent_map.get(cur)

                    if visual_root is None:
                        if wmap.aim_from_piece:
                            aim_from_key = wmap.aim_from_piece.lower()
                            # Check if aim_from is an ancestor of fp, OR a sibling of fp
                            fp_parent = parent_map.get(fp_key)
                            aim_from_is_candidate = False
                            cur = fp_parent
                            while cur is not None:
                                if cur == aim_from_key:
                                    aim_from_is_candidate = True
                                    break
                                cur = parent_map.get(cur)
                            # Also accept if aim_from is a sibling (same parent as fp)
                            if not aim_from_is_candidate and fp_parent:
                                if aim_from_key in children_map.get(fp_parent, []):
                                    aim_from_is_candidate = True
                            if aim_from_is_candidate:
                                if not _is_limb_joint(aim_from_key) and not _subtree_has_limb_joint(aim_from_key):
                                    candidate_subtree = _collect_subtree(aim_from_key, children_map)
                                    total_pieces = len(parent_map)
                                    # Sibling aim_from (e.g. spindle next to flare) may be larger
                                    # than ancestor aim_from — allow up to 70% for sibling case.
                                    limit = 0.70 if aim_from_key in children_map.get(fp_parent, []) else 0.50
                                    if total_pieces == 0 or len(candidate_subtree) <= total_pieces * limit:
                                        visual_root = aim_from_key
                        if visual_root is None:
                            # Walk up from fire_point, taking the highest ancestor whose
                            # subtree is ≤30% of the model, ≤10 pieces total, and does
                            # not contain structural limb joint pieces (uparm, torso, etc.)
                            cur = parent_map.get(fp_key)
                            best = cur
                            total_pieces = len(parent_map)
                            while cur is not None:
                                # Stop at limb joints or structural pieces.
                                # If best is still the direct parent (= the structural piece itself),
                                # there is no dedicated weapon visual — return None.
                                if _is_limb_joint(cur) or _is_structural(cur):
                                    if best == cur:
                                        best = None
                                    break
                                sub = _collect_subtree(cur, children_map)
                                # Stop if subtree contains limb joint pieces
                                if _subtree_has_limb_joint(cur):
                                    break
                                # Stop if this ancestor's subtree overlaps other weapons' pieces
                                if any(p in other_weapon_pieces for p in sub):
                                    break
                                # Accept ancestor if subtree ≤ 50% AND ≤ 10 pieces.
                                if len(sub) <= 10 and (total_pieces == 0 or len(sub) <= total_pieces * 0.50):
                                    best = cur
                                else:
                                    break
                                cur = parent_map.get(cur)
                            visual_root = best
                    return visual_root

                def _subtree_verts(root_key: str) -> int:
                    """Total vertex count across all pieces in subtree."""
                    return sum(piece_vert_count.get(k, 0) for k in _collect_subtree(root_key, children_map))

                # Process all fire_points (multi-barrel weapons have flare1..N in query_pieces)
                # Filter out "aim reference" pieces used as camera targets when unit is not
                # deployed (e.g. aimFlare on legapopupdef). These have "aim" in their name
                # and are not descended from any weapon aim_piece — they sit far from the barrel.
                # Keep them only if they're the sole fire_point (no better option).
                raw_fp_keys = list(dict.fromkeys(
                    fp.lower() for fp in (wmap.query_pieces if wmap.query_pieces else [wmap.query_piece])
                ))
                aim_set_lower = {ap.lower() for ap in wmap.aim_pieces}
                def _is_aim_reference(fp_k: str) -> bool:
                    """True if fp looks like a camera/aim reference, not a barrel flare."""
                    if 'aim' not in fp_k:
                        return False
                    # Check if any aim_piece is an ancestor of this fire_point
                    cur = parent_map.get(fp_k)
                    while cur is not None:
                        if cur in aim_set_lower:
                            return False  # fp is inside the weapon assembly
                        cur = parent_map.get(cur)
                    return True  # aim* piece not under any aim_piece = camera reference
                real_fps = [k for k in raw_fp_keys if not _is_aim_reference(k)]
                all_fp_keys = real_fps if real_fps else raw_fp_keys
                seen_roots: set = set()

                for fp_key in all_fp_keys:
                    visual_root = _find_visual_root(fp_key)
                    if not visual_root or visual_root in seen_roots:
                        continue

                    subtree = _collect_subtree(visual_root, children_map)

                    # If the visual root's subtree has no renderable geometry, walk UP the
                    # ancestor chain and collect small-subtree sibling pieces that have
                    # geometry and names suggesting weapon parts (not structural names).
                    # This handles units like legphoenix where ring1/2/3 are weapon geometry
                    # siblings of the fire_point ancestor, not connected via BOS aim_pieces.

                    if _subtree_verts(visual_root) <= 3:
                        cur_anc = parent_map.get(visual_root)
                        while cur_anc is not None:
                            geo_siblings = [k for k in children_map.get(cur_anc, [])
                                            if k != visual_root
                                            and not _is_dummy_piece(k)
                                            and not _is_structural(k)
                                            and _subtree_verts(k) > 0
                                            and len(_collect_subtree(k, children_map)) <= 8
                                            and k not in other_weapon_pieces
                                            and k not in seen_roots]
                            if geo_siblings:
                                # Prefer cur_anc as root if it is itself a small, non-structural
                                # piece whose subtree wraps the geo siblings (e.g. lgun containing
                                # lbarrel+lflare). This lets the mirror logic pick up rgun as well.
                                anc_sub = _collect_subtree(cur_anc, children_map)
                                total_pieces = len(parent_map)
                                if (not _is_structural(cur_anc)
                                        and not _is_limb_joint(cur_anc)
                                        and not _subtree_has_limb_joint(cur_anc)
                                        and (total_pieces == 0 or len(anc_sub) <= total_pieces * 0.50)
                                        and len(anc_sub) <= 10):
                                    visual_root = cur_anc
                                    subtree = list(anc_sub)
                                else:
                                    visual_root = geo_siblings[0]
                                    subtree = []
                                    for sib in geo_siblings:
                                        subtree.extend(_collect_subtree(sib, children_map))
                                break
                            cur_anc = parent_map.get(cur_anc)
                    total_pieces = len(parent_map)
                    fp_is_aim = fp_key in {ap.lower() for ap in wmap.aim_pieces}
                    # Skip only when the fire_point itself has real geometry (it's an aim_pivot
                    # masquerading as a fire_point) AND its subtree is very large.
                    # Zero-vert fire_points (flares, dummies) are real barrel tips — never skip.
                    fp_has_verts = piece_vert_count.get(fp_key, 0) > 0
                    is_dummy = fp_is_aim and fp_has_verts and total_pieces > 0 and len(subtree) > total_pieces * 0.30
                    if is_dummy:
                        print(f"  Weapon {wnum}: visual root = {visual_root}, "
                              f"subtree size = {len(subtree)} (skipped — fire_point is aim_piece and subtree too large)")
                        continue

                    seen_roots.add(visual_root)
                    tagged_subtrees = [subtree]
                    # Mirror sibling (l↔r or 1↔2) — only for bilateral symmetric pairs.
                    # Skip mirror when there are >2 fire_points (radial multi-barrel like
                    # legstarfall with 7 sleeves) to avoid claiming the mirror's own fp root.
                    mirror_root = _mirror(visual_root)
                    vr_parent = parent_map.get(visual_root)
                    mirror_parent = parent_map.get(mirror_root)
                    # Mirror is valid when mirror_root is either:
                    # (a) a sibling of visual_root (same parent), OR
                    # (b) a child of the mirrored parent (e.g. rarm under ruparm ↔ larm under luparm)
                    _mirror_parent_ok = (
                        vr_parent is not None and (
                            mirror_root in children_map.get(vr_parent, []) or
                            (mirror_parent is not None and _mirror(vr_parent) == mirror_parent)
                        )
                    )
                    if (mirror_root != visual_root
                            and mirror_root in children_map
                            and mirror_root not in other_weapon_pieces
                            and mirror_root not in seen_roots
                            and len(all_fp_keys) <= 2
                            and _mirror_parent_ok):
                        mirror_subtree = _collect_subtree(mirror_root, children_map)
                        tagged_subtrees.append(mirror_subtree)
                        seen_roots.add(mirror_root)

                    # Named-type siblings: pieces that share a "type word" with the visual
                    # root but aren't caught by l↔r mirror (e.g. leftBarrel/rightBarrel/topBarrel).
                    # Extract the longest lowercase word in the visual root name (e.g. "barrel"
                    # from "topBarrel") and tag all same-parent siblings that also contain it.
                    _BARREL_WORDS = {'barrel', 'gun', 'cannon', 'turret', 'sleeve',
                                     'launcher', 'missile', 'rocket', 'pod', 'tube'}
                    vr_lower = visual_root.lower()
                    vr_type_word = next((w for w in _BARREL_WORDS if w in vr_lower), None)
                    if vr_type_word:
                        vr_parent = parent_map.get(visual_root)
                        if vr_parent is not None:
                            for sib in children_map.get(vr_parent, []):
                                if (sib != visual_root
                                        and sib not in seen_roots
                                        and sib not in other_weapon_pieces
                                        and vr_type_word in sib.lower()
                                        and not _is_dummy_piece(sib)
                                        and _subtree_verts(sib) > 0):
                                    sib_subtree = _collect_subtree(sib, children_map)
                                    tagged_subtrees.append(sib_subtree)
                                    seen_roots.add(sib)

                    for s in tagged_subtrees:
                        for piece_key in s:
                            if (not _is_dummy_piece(piece_key)
                                    and (piece_key == visual_root or not _is_limb_joint(piece_key))
                                    and (piece_key == visual_root or piece_key not in other_aim_pieces)):
                                _add_to_lookup(piece_key, wnum, "visual")

                    # Also tag ancestors of the visual root that are part of the weapon mount:
                    # - same-weapon aim_pieces (e.g. sleeve/turret containing the barrels)
                    # - other small non-structural ancestors (e.g. strut/housing leading to base)
                    # These are pieces the raycaster can hit that must highlight too.
                    _total_anc = len(parent_map)
                    cur = parent_map.get(visual_root)
                    while cur is not None:
                        cur_sub = _collect_subtree(cur, children_map)
                        # Stop when ancestor is a shared pivot for OTHER weapons (e.g. aimx on legkark)
                        if cur in other_weapon_pieces:
                            break
                        # Stop when ancestor subtree contains pieces from OTHER weapons
                        if any(p in other_weapon_pieces for p in cur_sub):
                            break
                        # Stop at structural limb joints (uparm, torso, etc.) — never tag these
                        # as weapon visual, even if they are small or in the aim_set.
                        if _is_limb_joint(cur):
                            break
                        is_big = _total_anc > 0 and len(cur_sub) > _total_anc * 0.30
                        is_in_aim = cur in aim_set or cur == wmap.aim_from_piece
                        # Stop when ancestor is large AND structural (body/hull/base)
                        if is_big and _is_structural(cur):
                            break
                        # Tag aim_pieces as visual (even if big) — they are weapon geometry
                        # Also tag small non-structural ancestors (weapon mount struts/housings)
                        if not _is_dummy_piece(cur) and (is_in_aim or not _is_structural(cur)):
                            _add_to_lookup(cur, wnum, "visual")
                        # Stop after a large non-aim ancestor to avoid tagging whole model.
                        # Exception: if it has its own mesh geometry it's a rendered weapon
                        # housing (e.g. turretBaseHeadingPivot on legnavydestro) — tag and stop.
                        if is_big and not is_in_aim:
                            if piece_vert_count.get(cur, 0) > 0 and not _is_structural(cur):
                                _add_to_lookup(cur, wnum, "visual")
                            break
                        cur = parent_map.get(cur)

                    print(f"  Weapon {wnum}: visual root = {visual_root}, "
                          f"subtree size = {len(subtree)}")

    hide_pieces = hide_pieces or set()

    # Maps piece_name.lower() → glTF node index (built while adding pieces)
    node_name_to_idx: Dict[str, int] = {}
    # Maps piece_name.lower() → S3O rest offset (x, y, z)
    piece_offsets: Dict[str, tuple] = {}

    def add_piece_with_extras(piece: S3OPiece, parent_idx=None) -> int:
        """Add a piece node with weapon extras metadata."""
        piece_key = piece.name.lower()
        # Skip mesh geometry for hidden pieces so they don't affect bounding box
        if piece_key in hide_pieces:
            mesh_idx = None
        else:
            mesh_idx = builder.add_piece_mesh(piece, mat_idx)

        node = {"name": piece.name}
        ox, oy, oz = piece.offset
        if ox != 0 or oy != 0 or oz != 0:
            node["translation"] = [ox, oy, oz]
        if mesh_idx is not None:
            node["mesh"] = mesh_idx

        # Add weapon extras
        extras = {}
        if piece_key in weapon_lookup:
            winfo = weapon_lookup[piece_key]
            extras["weapons"] = sorted(winfo["weapons"])
            extras["weapon_roles"] = winfo["roles"]
        if piece_key in hide_pieces:
            extras["hide"] = True

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
                    "def": (weapon_defs or {}).get(wnum),
                    "fire_point": wmap.query_piece,
                    "aim_from": wmap.aim_from_piece,
                    "aim_pieces": wmap.aim_pieces,
                }
                for wnum, wmap in weapon_info.weapons.items()
            }
        if unit_role:
            root_extras["unit_role"] = unit_role
        builder.nodes[root_idx]["extras"] = root_extras
        builder.scenes[0]["nodes"] = [root_idx]

    # --- Animation ---
    if script_path and os.path.isfile(script_path):
        try:
            with open(script_path, 'r', errors='replace') as f:
                bos_content = f.read()
            now_rots = {}
            result = extract_walk_animation(bos_content)
            if result:
                anim_name, tracks, now_rots = result
                # Strip translation/rotation tracks for units where body sway looks wrong
                _strip_trans = _STRIP_ANIM_TRANSLATION.get(unit_name.lower(), set())
                _strip_rot = _STRIP_ANIM_ROTATION.get(unit_name.lower(), set())
                if _strip_trans:
                    tracks = [t for t in tracks if not (not t.is_rotation and t.piece.lower() in _strip_trans)]
                if _strip_rot:
                    tracks = [t for t in tracks if not (t.is_rotation and t.piece.lower() in _strip_rot)]
                target_dur = _ANIM_DURATION_OVERRIDE.get(unit_name.lower())
                if target_dur:
                    current_dur = max(kf.time for t in tracks for kf in t.keyframes)
                    if current_dur > 0:
                        scale = target_dur / current_dur
                        for t in tracks:
                            for kf in t.keyframes:
                                kf.time *= scale
                builder.apply_now_rotations(now_rots, node_name_to_idx)
                builder.add_animation(anim_name, tracks, node_name_to_idx, piece_offsets)
                # StopWalking pose — exported as a second clip so the viewer can
                # crossfade to the neutral stance when the movement toggle is off.
                stop_tracks = extract_stopwalking_pose(bos_content)
                if stop_tracks:
                    builder.add_animation('StopWalking', stop_tracks, node_name_to_idx, piece_offsets)
            else:
                # No walk animation — collect rest-pose rotations (Create() now + fly pose).
                # Apply them as static node rotations so aircraft show in fly pose.
                # They are also baked into spin animation keyframes below.
                now_rots = parse_create_now_rotations(bos_content)
                if now_rots:
                    builder.apply_now_rotations(now_rots, node_name_to_idx)
                    # Hide pieces that move underground after Activate (factory doors).
                    # Check: if Activate moves a piece to a negative y-position, mark hidden.
                    for (piece, axis, is_rot), val in now_rots.items():
                        if not is_rot and axis == 1:  # y-axis move
                            base_y = piece_offsets.get(piece, (0, 0, 0))[1]
                            final_y = base_y + val
                            if final_y < -2.0:
                                node_idx = node_name_to_idx.get(piece)
                                if node_idx is not None:
                                    builder.nodes[node_idx].setdefault('extras', {})['hide'] = True
                                    print(f"  Hiding {piece}: moved underground (y={final_y:.1f})")

            # Always try spin animation — some units have BOTH walk and spin
            # (e.g. factories with a dish + opening animation).
            # Only include spin clips for pieces that are visually meaningful in the viewer:
            # radar/sonar/jammer dishes, or any unit with an explicit unit_role.
            _SPIN_INTERESTING_NAMES = (
                'dish', 'radar', 'sonar', 'strut', 'turret', 'tower', 'spinner',
                'fork', 'jam', 'antenna', 'array',
                'fan', 'blade', 'turbine', 'collar', 'ball', 'blades', 'prop',
                'wheel', 'cradle', 'rotor', 'ring',
                'arm', 'stand', 'drill', 'sphere',
            )
            spin_clips = extract_spin_animation(bos_content)
            if spin_clips:
                # Keep only clips whose piece name is interesting, unless unit has a role
                if unit_role:
                    filtered_clips = spin_clips
                else:
                    filtered_clips = [
                        (cn, ct) for cn, ct in spin_clips
                        if any(frag in t.piece.lower() for t in ct
                               for frag in _SPIN_INTERESTING_NAMES)
                    ]
                if filtered_clips:
                    spin_pieces = []
                    # Group tracks by BOS event (e.g. "Activate_turbinef" → "Activate")
                    # so all spinners from the same event end up in one GLB animation clip.
                    # The viewer only plays the first animation, so a single merged clip
                    # ensures all pieces spin simultaneously.
                    from collections import defaultdict
                    grouped: dict = defaultdict(list)
                    for clip_name, clip_tracks in filtered_clips:
                        event = clip_name.split('_')[0] if '_' in clip_name else clip_name
                        grouped[event].extend(clip_tracks)
                    for event_name, all_tracks in grouped.items():
                        builder.add_spin_animation(event_name, all_tracks, node_name_to_idx,
                                                   now_rots or None)
                        spin_pieces.extend(t.piece for t in all_tracks)
                    # Store spin_pieces in root extras so viewer can target tooltip and animations
                    if model.root_piece:
                        root_idx = builder.scenes[0]["nodes"][0]
                        builder.nodes[root_idx].setdefault("extras", {})["spin_pieces"] = spin_pieces
            # Activate-loop animations (e.g. armaser spinarms — while(TRUE) + turn-to + sleep)
            if not spin_clips:
                loop_clips = extract_activate_loop_animation(bos_content)
                if loop_clips:
                    _strip_trans = _STRIP_ANIM_TRANSLATION.get(unit_name.lower(), set())
                    _strip_rot = _STRIP_ANIM_ROTATION.get(unit_name.lower(), set())
                    for clip_name, clip_tracks in loop_clips:
                        if _strip_trans:
                            clip_tracks = [t for t in clip_tracks if not (not t.is_rotation and t.piece.lower() in _strip_trans)]
                        if _strip_rot:
                            clip_tracks = [t for t in clip_tracks if not (t.is_rotation and t.piece.lower() in _strip_rot)]
                        builder.add_animation(clip_name, clip_tracks, node_name_to_idx,
                                              piece_offsets)
                    loop_pieces = [t.piece for _, ct in loop_clips for t in ct]
                    loop_clip_names = [cn for cn, _ in loop_clips]
                    if model.root_piece:
                        root_idx = builder.scenes[0]["nodes"][0]
                        builder.nodes[root_idx].setdefault("extras", {})["spin_pieces"] = loop_pieces + loop_clip_names

            # Toggle animations (Open/Close or MMStatus) — always check, independent of spin
            toggle_clips = extract_toggle_animations(bos_content)
            if toggle_clips:
                for clip_name, clip_tracks in toggle_clips:
                    builder.add_animation(clip_name, clip_tracks, node_name_to_idx,
                                          piece_offsets)
                if model.root_piece:
                    root_idx = builder.scenes[0]["nodes"][0]
                    extras = builder.nodes[root_idx].setdefault("extras", {})
                    extras["toggleable"] = True
                    if unit_name.lower().endswith('solar'):
                        extras["autoplay_open"] = True
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
                   info_only: bool = False,
                   weapon_defs: Optional[Dict[int, str]] = None) -> Optional[str]:
    """Convert a single S3O file to GLB."""
    model = parse_s3o(s3o_path)
    unit_name = os.path.splitext(os.path.basename(s3o_path))[0]

    # Piece name fragments that are always hidden in the viewer regardless of unit.
    # These are cosmetic/award pieces shown by in-game Lua widgets, not BOS Create().
    _GLOBAL_HIDE_FRAGMENTS = ('crown', 'medal')

    # Per-unit pieces to hide in the viewer (incorrectly positioned in rest pose).
    _UNIT_HIDE_PIECES: Dict[str, set] = {
        'legacluster': {'door1', 'door2', 'door3', 'door4', 'door5', 'door6',
                        'door1pivot', 'door2pivot', 'door3pivot',
                        'door4pivot', 'door5pivot', 'door6pivot'},
    }

    hide_pieces = set(_UNIT_HIDE_PIECES.get(unit_name.lower(), set()))
    # Add any piece whose name contains a globally-hidden fragment.
    for piece in model.all_pieces():
        if any(frag in piece.name.lower() for frag in _GLOBAL_HIDE_FRAGMENTS):
            hide_pieces.add(piece.name.lower())

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
        # Auto-hide pieces that BOS Create() hides at game start (medals, effects).
        # Only apply to mesh-less pieces (verts=0) — structural geometry that is
        # temporarily hidden in Create() but shown later by animations should stay
        # visible in the static viewer.
        with open(script_path, 'r', errors='replace') as _f:
            _bos = _f.read()
        bos_hides = parse_create_hide_pieces(_bos)
        if bos_hides:
            piece_verts = {p.name.lower(): len(p.vertices) for p in model.all_pieces()}
            hide_pieces |= {p for p in bos_hides if piece_verts.get(p, 0) == 0}

    # Expand hide_pieces to include all descendants of fragment-matched pieces only
    # (e.g. crown subtree). BOS-hidden aim-pivots (aimy1, aimx*) have important
    # children that must stay visible — do NOT expand those recursively.
    fragment_hidden = {p.name.lower() for p in model.all_pieces()
                       if any(frag in p.name.lower() for frag in _GLOBAL_HIDE_FRAGMENTS)}
    if fragment_hidden and model.root_piece:
        def _collect_subtree(piece, collecting):
            if collecting or piece.name.lower() in fragment_hidden:
                hide_pieces.add(piece.name.lower())
                collecting = True
            for child in piece.children:
                _collect_subtree(child, collecting)
        _collect_subtree(model.root_piece, False)

    # Per-unit weapon merges: map source weapon numbers → target weapon number.
    # Used when the BOS defines redundant separate weapons that should be linked
    # visually (e.g. legapopupdef w2+w3 are both miniguns of the same weapon def).
    _UNIT_WEAPON_MERGE: Dict[str, Dict[int, int]] = {
        'legapopupdef': {3: 2},  # minigunL (w3) → same weapon as minigunR (w2)
    }
    merge_map = _UNIT_WEAPON_MERGE.get(unit_name.lower(), {})
    if merge_map and weapon_info:
        for src_wnum, dst_wnum in merge_map.items():
            if src_wnum in weapon_info.weapons and dst_wnum in weapon_info.weapons:
                src = weapon_info.weapons.pop(src_wnum)
                dst = weapon_info.weapons[dst_wnum]
                # Merge all pieces from src into dst
                merged_query_pieces = list(dict.fromkeys(dst.query_pieces + src.query_pieces))
                merged_aim_pieces = sorted(set(dst.aim_pieces) | set(src.aim_pieces))
                dst.query_pieces = merged_query_pieces
                if not dst.query_piece and src.query_piece:
                    dst.query_piece = src.query_piece
                dst.aim_pieces = merged_aim_pieces
                dst._update_all()
                print(f"  Merged weapon {src_wnum} -> weapon {dst_wnum} (all_pieces now: {sorted(dst.all_pieces)})")

    # If weapon_defs not provided, try to find the unitdef .lua locally.
    # Also extract unit_role (RADAR/JAMMER/SONAR) from the same file.
    # Search for {unit_name}.lua in the same BAR install tree as the script/s3o.
    unit_role: Optional[str] = None
    if weapon_defs is None:
        def _native(p: str) -> str:
            """Convert MSYS/bash /c/... paths to Windows C:/... paths."""
            import re as _re
            return _re.sub(r'^/([a-zA-Z])/', lambda m: m.group(1).upper() + ':/', p)

        candidate_dirs = []
        for ref_path in [script_path, s3o_path]:
            if ref_path:
                # Walk up to find a 'units' sibling directory
                d = os.path.dirname(_native(ref_path))
                for _ in range(6):
                    units_dir = os.path.join(d, 'units')
                    if os.path.isdir(units_dir):
                        # Only accept if this units/ dir contains .lua files
                        # (rejects scripts/units/ which only has .bos/.cob)
                        has_lua = any(
                            fname.endswith('.lua')
                            for fname in os.listdir(units_dir)
                            if os.path.isfile(os.path.join(units_dir, fname))
                        )
                        if not has_lua:
                            # Check one level of subdirs for .lua files
                            has_lua = any(
                                fname.endswith('.lua')
                                for sub in os.listdir(units_dir)
                                for fname in (os.listdir(os.path.join(units_dir, sub))
                                              if os.path.isdir(os.path.join(units_dir, sub)) else [])
                            )
                        if has_lua:
                            candidate_dirs.append(units_dir)
                            break
                    d = os.path.dirname(d)
        for units_dir in candidate_dirs:
            for root, _, files in os.walk(units_dir):
                for fname in files:
                    if fname.lower() == f'{unit_name.lower()}.lua':
                        lua_path = os.path.join(root, fname)
                        try:
                            with open(lua_path, 'r', errors='replace') as f:
                                lua_content = f.read()
                            weapon_defs = parse_lua_weapon_defs(lua_content)
                            if weapon_defs:
                                print(f"  Weapon defs from: {lua_path}")
                            unit_role = parse_lua_unit_role(lua_content)
                            if unit_role:
                                print(f"  Unit role: {unit_role}")
                            break
                        except Exception:
                            pass
                if weapon_defs or unit_role:
                    break
            if weapon_defs or unit_role:
                break

    if info_only:
        return None

    if output_path is None:
        output_path = os.path.splitext(s3o_path)[0] + '.glb'

    glb_data = convert_with_weapons(model, weapon_info, script_path, weapon_defs, hide_pieces, unit_role, unit_name)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(glb_data)

    print(f"\n  GLB written: {output_path} ({len(glb_data):,} bytes)")
    return output_path


def batch_convert(bar_dir: str, output_dir: str, unit_filter: str = None):
    """Batch convert all S3O files in a BAR game directory (including subdirs)."""
    import fnmatch
    objects_dir = os.path.join(bar_dir, 'objects3d')
    if not os.path.isdir(objects_dir):
        print(f"Error: objects3d directory not found at {objects_dir}")
        return

    # Walk all subdirectories to find .s3o files.
    # Skip variant subdirs (event/, aprilfools/, scavboss/, etc.) — only use
    # files directly under objects3d/ or objects3d/Units/.
    _SKIP_DIRS = {'event', 'aprilfools', 'scavboss', 'lups', 'test'}
    s3o_paths = []
    for root, dirs, files in os.walk(objects_dir):
        # Prune skipped subdirs in-place so os.walk won't descend into them
        dirs[:] = [d for d in dirs if d.lower() not in _SKIP_DIRS]
        for f in files:
            if f.lower().endswith('.s3o'):
                s3o_paths.append(os.path.join(root, f))
    s3o_paths.sort()

    # Always exclude dead/wreck/debris models and commander units with custom animations
    _EXCLUDE = ('_dead', 'wreck', 'debris')
    _EXCLUDE_EXACT = {'armcom', 'corcom'}
    s3o_paths = [p for p in s3o_paths
                 if not any(x in os.path.basename(p).lower() for x in _EXCLUDE)
                 and os.path.splitext(os.path.basename(p))[0].lower() not in _EXCLUDE_EXACT]

    if unit_filter:
        # Support glob patterns like "arm*"
        s3o_paths = [p for p in s3o_paths
                     if fnmatch.fnmatch(os.path.splitext(os.path.basename(p))[0].lower(),
                                        unit_filter.lower())]

    print(f"Found {len(s3o_paths)} S3O files to convert")
    os.makedirs(output_dir, exist_ok=True)

    success, failed = 0, 0
    for s3o_path in s3o_paths:
        filename = os.path.basename(s3o_path)
        unit_name = os.path.splitext(filename)[0]
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


_units_tree_cache: Optional[list] = None


def _get_units_tree() -> list:
    """Fetch and cache the full units/ tree from GitHub (3 API calls, cached)."""
    global _units_tree_cache
    if _units_tree_cache is not None:
        return _units_tree_cache

    commit = _github_get(f"{BAR_API}/commits/master")
    root_tree_sha = commit["commit"]["tree"]["sha"]
    root_tree = _github_get(f"{BAR_API}/git/trees/{root_tree_sha}")
    units_entry = next((e for e in root_tree["tree"] if e["path"] == "units"), None)
    if not units_entry:
        raise RuntimeError("'units' directory not found in repo root tree")
    units_tree = _github_get(f"{BAR_API}/git/trees/{units_entry['sha']}?recursive=1")
    _units_tree_cache = units_tree["tree"]
    return _units_tree_cache


def _find_unit_lua_path(unit_name: str) -> Optional[str]:
    """
    Find the path of {unit_name}.lua in the BAR units/ directory tree.
    Uses the git trees API to list all files under units/ in 3 requests (cached).
    """
    target = f"{unit_name.lower()}.lua"
    for entry in _get_units_tree():
        if entry["type"] == "blob" and entry["path"].lower().endswith("/" + target):
            return f"units/{entry['path']}"
        if entry["type"] == "blob" and entry["path"].lower() == target:
            return f"units/{entry['path']}"
    return None


def _find_units_with_prefix(prefix: str) -> List[str]:
    """Return all unit names whose .lua filename starts with the given prefix."""
    prefix_lower = prefix.lower()
    names = []
    for entry in _get_units_tree():
        if entry["type"] != "blob":
            continue
        filename = entry["path"].split("/")[-1]
        if not filename.endswith(".lua"):
            continue
        name = filename[:-4]  # strip .lua
        if name.lower().startswith(prefix_lower):
            names.append(name)
    return sorted(names)


def fetch_unit_from_github(unit_name: str, output_path: Optional[str] = None,
                            info_only: bool = False,
                            push: bool = False,
                            force: bool = False) -> Optional[str]:
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

        weapon_defs = parse_lua_weapon_defs(lua_content)
        glb_path = convert_single(s3o_local, script_local, output_path, info_only, weapon_defs)
        if glb_path and push and not info_only:
            push_glb_to_repo(glb_path, force=force)
        return glb_path


VIEWER_REPO = "icexuick/BAR-modelviewer"


def push_glb_to_repo(glb_path: str, force: bool = False):
    """Upload or overwrite a GLB file in the BAR-modelviewer GitHub repo."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("  Warning: GITHUB_TOKEN not set, skipping push to repo")
        return

    filename = os.path.basename(glb_path)
    repo_path = f"glb/{filename}"  # GLBs live in the glb/ subfolder

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
    if existing_sha and blob_sha == existing_sha and not force:
        print(f"  No changes, skipping push ({filename} already up to date)")
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
    print(f"  {action} in repo: https://github.com/{VIEWER_REPO}/blob/main/{repo_path}")


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
    parser.add_argument('--force', action='store_true',
                        help='Force push to GitHub even if file is unchanged')
    parser.add_argument('--prefix', help='Convert all units whose name starts with this prefix (e.g. "leg")')

    args = parser.parse_args()

    if args.prefix:
        unit_names = _find_units_with_prefix(args.prefix)
        if not unit_names:
            print(f"No units found with prefix '{args.prefix}'")
            return
        print(f"Found {len(unit_names)} units with prefix '{args.prefix}': {unit_names}")
        ok, skipped, failed = 0, 0, []
        for i, unit_name in enumerate(unit_names, 1):
            print(f"\n[{i}/{len(unit_names)}] {unit_name}")
            try:
                result = fetch_unit_from_github(
                    unit_name, None, False,
                    push=not args.local,
                    force=args.force,
                )
                if result:
                    ok += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"  ERROR: {e}")
                failed.append(unit_name)
        print(f"\n=== Batch complete: {ok} converted, {skipped} skipped, {len(failed)} failed ===")
        if failed:
            print(f"Failed: {failed}")
        return

    if args.unit:
        if args.local:
            # --local: save GLB to glb/ subfolder so the user can inspect it
            if args.output is None:
                repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                glb_dir = os.path.join(repo_root, "glb")
                os.makedirs(glb_dir, exist_ok=True)
                args.output = os.path.join(glb_dir, f"{args.unit}.glb")
        fetch_unit_from_github(
            args.unit, args.output, args.info_only,
            push=not args.local and not args.info_only,
            force=args.force,
        )
    elif args.bar_dir:
        batch_convert(args.bar_dir, args.output_dir, args.filter)
    elif args.s3o:
        convert_single(args.s3o, args.script, args.output, args.info_only)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
