"""
BOS Script Parser — extracts weapon-to-piece mappings from BAR animation scripts.

In the Spring/BAR engine, BOS (Block of Script) files define unit animations.
Weapons are mapped to model pieces through specific functions:

  - QueryWeaponN(...)   → returns the piece that fires weapon N (the barrel/flare)
  - AimFromWeaponN(...) → returns the piece used for aiming weapon N (the turret)
  - AimWeaponN(...)     → contains turn commands showing which pieces rotate for aiming

The compiled .cob files are binary and hard to parse directly.
The .bos SOURCE files are human-readable and contain this information.

This parser reads .bos files and extracts:
  1. The piece declarations (piece list)
  2. Weapon → piece mappings from QueryWeapon/AimFromWeapon functions
  3. Aiming pieces from turn commands inside AimWeapon functions

BAR-specific notes:
  - BOS files are in: scripts/Units/<unitname>.bos
  - The unitdef references scripts via: script = "Units/<unitname>.cob"
  - Piece names in BOS must match piece names in the S3O model
  - BAR uses Lua scripts for some newer units (scripts/Units/<unitname>.lua)
"""

import re
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set


@dataclass
class WeaponPieceMapping:
    """Maps a weapon number to its associated S3O model pieces."""
    weapon_num: int                    # 1-based weapon number
    query_piece: Optional[str] = None  # piece returned by QueryWeapon (fire point)
    aim_from_piece: Optional[str] = None  # piece returned by AimFromWeapon (aim origin)
    aim_pieces: List[str] = field(default_factory=list)  # pieces turned in AimWeapon
    all_pieces: Set[str] = field(default_factory=set)  # union of all weapon pieces

    def __post_init__(self):
        self._update_all()

    def _update_all(self):
        self.all_pieces = set()
        if self.query_piece:
            self.all_pieces.add(self.query_piece)
        if self.aim_from_piece:
            self.all_pieces.add(self.aim_from_piece)
        self.all_pieces.update(self.aim_pieces)


@dataclass
class BOSParseResult:
    """Result of parsing a BOS file."""
    pieces: List[str] = field(default_factory=list)
    weapons: Dict[int, WeaponPieceMapping] = field(default_factory=dict)
    static_vars: List[str] = field(default_factory=list)

    def weapon_pieces(self) -> Dict[str, List[int]]:
        """Returns piece_name → [weapon_numbers] mapping (inverse lookup)."""
        result: Dict[str, List[int]] = {}
        for wnum, wmap in self.weapons.items():
            for piece_name in wmap.all_pieces:
                if piece_name not in result:
                    result[piece_name] = []
                result[piece_name].append(wnum)
        return result

    def print_summary(self):
        print(f"Pieces ({len(self.pieces)}): {', '.join(self.pieces)}")
        print(f"Weapons found: {len(self.weapons)}")
        for wnum in sorted(self.weapons.keys()):
            w = self.weapons[wnum]
            print(f"  Weapon {wnum}:")
            if w.query_piece:
                print(f"    Fire point (QueryWeapon): {w.query_piece}")
            if w.aim_from_piece:
                print(f"    Aim from (AimFromWeapon): {w.aim_from_piece}")
            if w.aim_pieces:
                print(f"    Aiming pieces (turn in AimWeapon): {', '.join(w.aim_pieces)}")
            print(f"    All associated pieces: {', '.join(sorted(w.all_pieces))}")

        # Inverse mapping
        wp = self.weapon_pieces()
        if wp:
            print(f"\nPiece -> Weapon mapping:")
            for piece in sorted(wp.keys()):
                wnums = sorted(wp[piece])
                print(f"  {piece} -> weapon(s) {', '.join(str(w) for w in wnums)}")


def parse_bos(filepath: str) -> BOSParseResult:
    """Parse a BOS file and extract weapon-piece mappings."""
    with open(filepath, 'r', errors='replace') as f:
        content = f.read()

    result = BOSParseResult()

    # 1. Extract piece declarations
    # Format: "piece base, turret, barrel, flare, ..."
    piece_match = re.search(
        r'piece\s+([\w,\s]+?)\s*;',
        content,
        re.IGNORECASE
    )
    if piece_match:
        pieces_str = piece_match.group(1)
        result.pieces = [p.strip().lower() for p in pieces_str.split(',') if p.strip()]

    # Also catch multi-line piece declarations
    if not result.pieces:
        # Sometimes pieces are declared on multiple lines
        all_pieces = re.findall(r'piece\s+([\w,\s]+)', content, re.IGNORECASE)
        for pm in all_pieces:
            for p in pm.split(','):
                p = p.strip().lower()
                if p and p not in result.pieces:
                    result.pieces.append(p)

    # Remove comments for cleaner parsing
    # BOS uses // for line comments and /* */ for block comments
    clean = re.sub(r'//[^\n]*', '', content)
    clean = re.sub(r'/\*.*?\*/', '', clean, flags=re.DOTALL)

    # Old-style BOS uses Primary/Secondary/Tertiary/Quaternary instead of Weapon1/2/3/4
    _LEGACY_WEAPON_MAP = {
        'primary': 1, 'secondary': 2, 'tertiary': 3, 'quaternary': 4,
    }

    def _legacy_wnum(name_group: str, num_group: str) -> int:
        if num_group:
            return int(num_group)
        return _LEGACY_WEAPON_MAP.get((name_group or '').lower(), 1)

    _LEGACY_NAMES = 'Primary|Secondary|Tertiary|Quaternary'

    # 2. Extract QueryWeaponN / QueryPrimary / QuerySecondary / ... functions
    for match in re.finditer(
        rf'Query(Weapon(\d+)|({_LEGACY_NAMES}))\s*\([^)]*\)\s*\{{([^}}]+)\}}',
        clean, re.DOTALL | re.IGNORECASE
    ):
        wnum = _legacy_wnum(match.group(3), match.group(2))
        body = match.group(4)
        piece_ref = _extract_piece_from_function(body, result.pieces)
        if piece_ref:
            if wnum not in result.weapons:
                result.weapons[wnum] = WeaponPieceMapping(weapon_num=wnum)
            result.weapons[wnum].query_piece = piece_ref
            result.weapons[wnum]._update_all()

    # 3. Extract AimFromWeaponN / AimFromPrimary / ... functions
    for match in re.finditer(
        rf'AimFrom(Weapon(\d+)|({_LEGACY_NAMES}))\s*\([^)]*\)\s*\{{([^}}]+)\}}',
        clean, re.DOTALL | re.IGNORECASE
    ):
        wnum = _legacy_wnum(match.group(3), match.group(2))
        body = match.group(4)
        piece_ref = _extract_piece_from_function(body, result.pieces)
        if piece_ref:
            if wnum not in result.weapons:
                result.weapons[wnum] = WeaponPieceMapping(weapon_num=wnum)
            result.weapons[wnum].aim_from_piece = piece_ref
            result.weapons[wnum]._update_all()

    # 4. Extract AimWeaponN / AimPrimary / ... — look for turn commands
    for match in re.finditer(
        rf'Aim(Weapon(\d+)|({_LEGACY_NAMES}))\s*\([^)]*\)\s*\{{([^}}]+)\}}',
        clean, re.DOTALL | re.IGNORECASE
    ):
        wnum = _legacy_wnum(match.group(3), match.group(2))
        body = match.group(4)

        aim_pieces = set()
        for turn_match in re.finditer(r'turn\s+(\w+)\s+to\s+[xyz]-axis', body, re.IGNORECASE):
            piece_name = turn_match.group(1).lower()
            if piece_name in [p.lower() for p in result.pieces]:
                aim_pieces.add(piece_name)

        if aim_pieces:
            if wnum not in result.weapons:
                result.weapons[wnum] = WeaponPieceMapping(weapon_num=wnum)
            result.weapons[wnum].aim_pieces = sorted(aim_pieces)
            result.weapons[wnum]._update_all()

    return result


def _extract_piece_from_function(body: str, known_pieces: List[str]) -> Optional[str]:
    """Extract a piece name from a QueryWeapon/AimFromWeapon function body."""
    # Pattern 1: "piecenum = <piece>;" — most common in BOS
    # The piece variable is typically used directly
    for piece in known_pieces:
        # Check if piece name appears in a return-like context
        if re.search(rf'\b{re.escape(piece)}\b', body, re.IGNORECASE):
            return piece.lower()

    return None


def parse_lua_script(filepath: str) -> BOSParseResult:
    """Parse a Lua unit script for weapon-piece mappings.

    BAR Lua scripts use a different pattern:
      function script.QueryWeapon1() return piece_name end
      function script.AimWeapon1(heading, pitch) ... end
    """
    with open(filepath, 'r', errors='replace') as f:
        content = f.read()

    result = BOSParseResult()

    # Extract piece declarations from "local piece_name = piece('name')" pattern
    for match in re.finditer(r"local\s+(\w+)\s*=\s*piece\s*['\"](\w+)['\"]", content):
        var_name = match.group(1)
        piece_name = match.group(2).lower()
        result.pieces.append(piece_name)

    # Also from: "local pieces = { base = piece('base'), ... }"
    for match in re.finditer(r"piece\s*\(\s*['\"](\w+)['\"]\s*\)", content):
        piece_name = match.group(1).lower()
        if piece_name not in result.pieces:
            result.pieces.append(piece_name)

    # QueryWeaponN
    for match in re.finditer(
        r'function\s+script\.QueryWeapon(\d+)\s*\([^)]*\)(.*?)(?=\nfunction|\nend\s*$|\Z)',
        content, re.DOTALL
    ):
        wnum = int(match.group(1))
        body = match.group(2)
        # Look for "return <piece>" pattern
        ret_match = re.search(r'return\s+(\w+)', body)
        if ret_match:
            piece_var = ret_match.group(1)
            if wnum not in result.weapons:
                result.weapons[wnum] = WeaponPieceMapping(weapon_num=wnum)
            result.weapons[wnum].query_piece = piece_var.lower()
            result.weapons[wnum]._update_all()

    # AimFromWeaponN
    for match in re.finditer(
        r'function\s+script\.AimFromWeapon(\d+)\s*\([^)]*\)(.*?)(?=\nfunction|\nend\s*$|\Z)',
        content, re.DOTALL
    ):
        wnum = int(match.group(1))
        body = match.group(2)
        ret_match = re.search(r'return\s+(\w+)', body)
        if ret_match:
            piece_var = ret_match.group(1)
            if wnum not in result.weapons:
                result.weapons[wnum] = WeaponPieceMapping(weapon_num=wnum)
            result.weapons[wnum].aim_from_piece = piece_var.lower()
            result.weapons[wnum]._update_all()

    # AimWeaponN — look for Turn() calls
    for match in re.finditer(
        r'function\s+script\.AimWeapon(\d+)\s*\([^)]*\)(.*?)(?=\nfunction|\nend\s*$|\Z)',
        content, re.DOTALL
    ):
        wnum = int(match.group(1))
        body = match.group(2)
        aim_pieces = set()
        for turn_match in re.finditer(r'Turn\s*\(\s*(\w+)', body):
            piece_var = turn_match.group(1).lower()
            aim_pieces.add(piece_var)
        if aim_pieces:
            if wnum not in result.weapons:
                result.weapons[wnum] = WeaponPieceMapping(weapon_num=wnum)
            result.weapons[wnum].aim_pieces = sorted(aim_pieces)
            result.weapons[wnum]._update_all()

    return result


def parse_unit_script(filepath: str) -> BOSParseResult:
    """Auto-detect BOS vs Lua script and parse accordingly."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.lua':
        return parse_lua_script(filepath)
    elif ext == '.bos':
        return parse_bos(filepath)
    else:
        # Try BOS first, fall back to Lua
        try:
            result = parse_bos(filepath)
            if result.pieces:
                return result
        except Exception:
            pass
        return parse_lua_script(filepath)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python bos_parser.py <script.bos|script.lua>")
        sys.exit(1)

    result = parse_unit_script(sys.argv[1])
    result.print_summary()
