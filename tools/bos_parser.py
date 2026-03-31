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
    query_piece: Optional[str] = None  # primary fire point piece (first found)
    query_pieces: List[str] = field(default_factory=list)  # ALL fire point pieces (multi-barrel)
    aim_from_piece: Optional[str] = None  # piece returned by AimFromWeapon (aim origin)
    aim_pieces: List[str] = field(default_factory=list)  # pieces turned in AimWeapon
    aim_disabled: bool = False  # True when AimWeapon returns 0 (e.g. drone controller)
    all_pieces: Set[str] = field(default_factory=set)  # union of all weapon pieces

    def __post_init__(self):
        self._update_all()

    def _update_all(self):
        self.all_pieces = set()
        if self.query_piece:
            self.all_pieces.add(self.query_piece)
        self.all_pieces.update(self.query_pieces)
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

    # Remove comments for cleaner parsing first — piece declarations may have
    # inline // comments (e.g. "railTopRight, //4\n railBotRight,") that break parsing.
    # BOS uses // for line comments and /* */ for block comments
    clean = re.sub(r'//[^\n]*', '', content)
    clean = re.sub(r'/\*.*?\*/', '', clean, flags=re.DOTALL)

    # 1. Extract piece declarations from comment-stripped source
    # Format: "piece base, turret, barrel, flare, ..."
    piece_match = re.search(
        r'piece\s+([\w,\s]+?)\s*;',
        clean,
        re.IGNORECASE
    )
    if piece_match:
        pieces_str = piece_match.group(1)
        result.pieces = [p.strip().lower() for p in pieces_str.split(',') if p.strip()]

    # Also catch multi-line piece declarations without semicolon
    if not result.pieces:
        all_pieces = re.findall(r'piece\s+([\w,\s]+)', clean, re.IGNORECASE)
        for pm in all_pieces:
            for p in pm.split(','):
                p = p.strip().lower()
                if p and p not in result.pieces:
                    result.pieces.append(p)

    # Old-style BOS uses Primary/Secondary/Tertiary/Quaternary instead of Weapon1/2/3/4
    _LEGACY_WEAPON_MAP = {
        'primary': 1, 'secondary': 2, 'tertiary': 3, 'quaternary': 4,
    }

    def _legacy_wnum(name_group: str, num_group: str) -> int:
        if num_group:
            return int(num_group)
        return _LEGACY_WEAPON_MAP.get((name_group or '').lower(), 1)

    _LEGACY_NAMES = 'Primary|Secondary|Tertiary|Quaternary'

    def _extract_brace_body(text: str, func_pattern: str) -> List[tuple]:
        """Find all occurrences of func_pattern and return (group1, body) pairs
        using proper brace matching — handles nested if/while blocks."""
        results = []
        for m in re.finditer(func_pattern, text, re.IGNORECASE):
            open_pos = text.find('{', m.end())
            if open_pos == -1:
                continue
            depth = 0
            pos = open_pos
            while pos < len(text):
                if text[pos] == '{':
                    depth += 1
                elif text[pos] == '}':
                    depth -= 1
                    if depth == 0:
                        results.append((m.group(1), text[open_pos + 1:pos]))
                        break
                pos += 1
        return results

    # 2. Extract QueryWeaponN / QueryPrimary / QuerySecondary / ... functions
    # Use brace-matching so if/else chains (multi-barrel cycling) are parsed fully.
    _QUERY_PAT = rf'Query(Weapon(\d+)|({_LEGACY_NAMES}))\s*\([^)]*\)\s*(?=\{{)'
    for suffix, body in _extract_brace_body(clean, _QUERY_PAT):
        num_m = re.match(r'Weapon(\d+)', suffix, re.IGNORECASE)
        leg_m = re.match(rf'({_LEGACY_NAMES})', suffix, re.IGNORECASE)
        wnum = int(num_m.group(1)) if num_m else _LEGACY_WEAPON_MAP.get((leg_m.group(1) if leg_m else '').lower(), 1)
        all_refs = _extract_all_pieces_from_function(body, result.pieces)
        # Detect BOS barrel-alternating: "pieceIndex = <piece> + gun_N"
        # gun_N toggles 0/1 each shot, so piece and its adjacent piece (by index) both fire.
        # We also look for the mirror piece by name (l↔r prefix/suffix swap).
        gun_alt = re.search(r'pieceIndex\s*=\s*(\w+)\s*\+\s*gun_\d+', body, re.IGNORECASE)
        if gun_alt:
            base_piece = gun_alt.group(1).lower()
            if base_piece in result.pieces:
                base_idx = result.pieces.index(base_piece)
                # Try name-mirror first: swap l↔r in prefix or suffix
                def _mirror(name):
                    if name.startswith('l'): return 'r' + name[1:]
                    if name.startswith('r'): return 'l' + name[1:]
                    if name.endswith('l'): return name[:-1] + 'r'
                    if name.endswith('r'): return name[:-1] + 'l'
                    return None
                mirror = _mirror(base_piece)
                if mirror and mirror in result.pieces:
                    alt_piece = mirror
                elif base_idx + 1 < len(result.pieces):
                    alt_piece = result.pieces[base_idx + 1]
                elif base_idx - 1 >= 0:
                    alt_piece = result.pieces[base_idx - 1]
                else:
                    alt_piece = None
                if alt_piece and alt_piece not in all_refs:
                    all_refs = [base_piece, alt_piece]
        if all_refs:
            if wnum not in result.weapons:
                result.weapons[wnum] = WeaponPieceMapping(weapon_num=wnum)
            result.weapons[wnum].query_piece = all_refs[0]
            result.weapons[wnum].query_pieces = all_refs
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

    # 4. Extract AimWeaponN / AimPrimary / ... — use brace-matching to handle
    # nested if/while blocks that trip up simple [^}]+ regex.
    _AIM_PAT = rf'Aim(Weapon(\d+)|({_LEGACY_NAMES}))\s*\([^)]*\)\s*(?=\{{)'
    for suffix, body in _extract_brace_body(clean, _AIM_PAT):
        num_m = re.match(r'Weapon(\d+)', suffix, re.IGNORECASE)
        leg_m = re.match(rf'({_LEGACY_NAMES})', suffix, re.IGNORECASE)
        wnum = int(num_m.group(1)) if num_m else _LEGACY_WEAPON_MAP.get((leg_m.group(1) if leg_m else '').lower(), 1)

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

        # Detect disabled weapons: AimWeapon returns 0 with no turn/move commands
        # and no call-script (indirect aiming) and no return(1) (conditional aiming).
        # e.g. drone controllers that don't actually aim
        stripped = re.sub(r'//[^\n]*', '', body)  # strip comments
        has_return_0 = bool(re.search(r'\breturn\s*\(\s*0\s*\)', stripped, re.IGNORECASE))
        has_return_1 = bool(re.search(r'\breturn\s*\(\s*1\s*\)', stripped, re.IGNORECASE))
        has_turn_move = bool(re.search(r'\b(?:turn|move)\s+\w+\s+to\s+[xyz]-axis', stripped, re.IGNORECASE))
        has_call_script = bool(re.search(r'\bcall-script\b', stripped, re.IGNORECASE))
        if has_return_0 and not has_turn_move and not has_call_script and not has_return_1:
            if wnum not in result.weapons:
                result.weapons[wnum] = WeaponPieceMapping(weapon_num=wnum)
            result.weapons[wnum].aim_disabled = True

    return result


def _extract_piece_from_function(body: str, known_pieces: List[str]) -> Optional[str]:
    """Extract the first piece name from a QueryWeapon/AimFromWeapon function body."""
    for piece in known_pieces:
        if re.search(rf'\b{re.escape(piece)}\b', body, re.IGNORECASE):
            return piece.lower()
    return None


def _extract_all_pieces_from_function(body: str, known_pieces: List[str]) -> List[str]:
    """Extract ALL piece names from a QueryWeapon body (for multi-barrel weapons).

    Some units have QueryWeapon functions with if/else chains that assign different
    fire point pieces per shot (e.g. 6-tube missile launcher cycling through flare1..6).
    This returns all distinct pieces mentioned in the body, in declaration order.
    """
    found = []
    for piece in known_pieces:
        if re.search(rf'\b{re.escape(piece)}\b', body, re.IGNORECASE):
            found.append(piece.lower())
    return found


def parse_lua_script(filepath: str) -> BOSParseResult:
    """Parse a Lua unit script for weapon-piece mappings.

    Supports two calling conventions:

    1. Numbered variant (simple scripts):
       function script.QueryWeapon1() return piece_name end
       function script.AimWeapon1(heading, pitch) ... end

    2. Combined variant with weapons table (commander scripts):
       weapons = { [1] = "laser", [2] = "uwlaser", [3] = "dgun" }
       function script.QueryWeapon(weapon)
           if weapons[weapon] == "laser" then return laserflare end
       end
       function script.AimWeapon(weapon, heading, pitch)
           if weapons[weapon] == "laser" then
               Turn(aimy1, 2, heading, ...)
           end
       end
    """
    with open(filepath, 'r', errors='replace') as f:
        content = f.read()

    result = BOSParseResult()

    # --- Piece declarations ---
    # Multi-piece: local a, b, c = piece("a", "b", "c")
    piece_map: Dict[str, str] = {}  # variable name → piece string name
    for match in re.finditer(r'local\s+([\w\s,]+?)\s*=\s*piece\s*\((.*?)\)', content, re.DOTALL):
        vars_str = match.group(1)
        pieces_str = match.group(2)
        var_names = [v.strip() for v in vars_str.split(',') if v.strip()]
        piece_names = re.findall(r'["\'](\w+)["\']', pieces_str)
        for var, pname in zip(var_names, piece_names):
            piece_map[var] = pname.lower()
            if pname.lower() not in result.pieces:
                result.pieces.append(pname.lower())

    # Single-piece: local turret = piece("turret") or piece 'turret'
    for match in re.finditer(r"local\s+(\w+)\s*=\s*piece\s*['\"](\w+)['\"]", content):
        var_name = match.group(1)
        piece_name = match.group(2).lower()
        if var_name not in piece_map:
            piece_map[var_name] = piece_name
        if piece_name not in result.pieces:
            result.pieces.append(piece_name)

    # Also from: piece("name") anywhere (catch-all)
    for match in re.finditer(r"piece\s*\(\s*['\"](\w+)['\"]\s*\)", content):
        piece_name = match.group(1).lower()
        if piece_name not in result.pieces:
            result.pieces.append(piece_name)

    def _resolve_piece(var_name: str) -> str:
        """Resolve a variable name to its piece string name."""
        return piece_map.get(var_name, var_name).lower()

    # --- Weapons table (combined variant) ---
    # weapons = { [1] = "laser", [2] = "uwlaser", [3] = "dgun" }
    weapon_types: Dict[int, str] = {}
    wt_match = re.search(r'\bweapons\s*=\s*\{(.*?)\}', content, re.DOTALL)
    if wt_match:
        for entry in re.finditer(r'\[(\d+)\]\s*=\s*["\'](\w+)["\']', wt_match.group(1)):
            weapon_types[int(entry.group(1))] = entry.group(2)

    # --- Numbered variant: script.QueryWeaponN ---
    for match in re.finditer(
        r'function\s+script\.QueryWeapon(\d+)\s*\([^)]*\)(.*?)(?=\nfunction\s|\Z)',
        content, re.DOTALL
    ):
        wnum = int(match.group(1))
        body = match.group(2)
        ret_match = re.search(r'return\s+(\w+)', body)
        if ret_match:
            if wnum not in result.weapons:
                result.weapons[wnum] = WeaponPieceMapping(weapon_num=wnum)
            result.weapons[wnum].query_piece = _resolve_piece(ret_match.group(1))
            result.weapons[wnum]._update_all()

    # --- Numbered variant: script.AimFromWeaponN ---
    for match in re.finditer(
        r'function\s+script\.AimFromWeapon(\d+)\s*\([^)]*\)(.*?)(?=\nfunction\s|\Z)',
        content, re.DOTALL
    ):
        wnum = int(match.group(1))
        body = match.group(2)
        ret_match = re.search(r'return\s+(\w+)', body)
        if ret_match:
            if wnum not in result.weapons:
                result.weapons[wnum] = WeaponPieceMapping(weapon_num=wnum)
            result.weapons[wnum].aim_from_piece = _resolve_piece(ret_match.group(1))
            result.weapons[wnum]._update_all()

    # --- Numbered variant: script.AimWeaponN ---
    for match in re.finditer(
        r'function\s+script\.AimWeapon(\d+)\s*\([^)]*\)(.*?)(?=\nfunction\s|\Z)',
        content, re.DOTALL
    ):
        wnum = int(match.group(1))
        body = match.group(2)
        aim_pieces = set()
        for turn_match in re.finditer(r'\b[Tt]urn\s*\(\s*(\w+)', body):
            aim_pieces.add(_resolve_piece(turn_match.group(1)))
        if aim_pieces:
            if wnum not in result.weapons:
                result.weapons[wnum] = WeaponPieceMapping(weapon_num=wnum)
            result.weapons[wnum].aim_pieces = sorted(aim_pieces)
            result.weapons[wnum]._update_all()

    # --- Combined variant: script.QueryWeapon(weapon) ---
    # Dispatch via: if weapons[weapon] == "type" then return piece end
    qw_match = re.search(
        r'function\s+script\.QueryWeapon\s*\(\s*weapon\s*\)(.*?)(?=\nfunction\s|\Z)',
        content, re.DOTALL
    )
    if qw_match and weapon_types:
        body = qw_match.group(1)
        # Parse each branch: weapons[weapon] == "type" then ... return piece
        for branch in re.finditer(
            r'weapons\[weapon\]\s*==\s*["\'](\w+)["\']\s*then\s*\n?\s*return\s+(\w+)',
            body
        ):
            wtype = branch.group(1)
            piece_var = branch.group(2)
            # Map all weapon numbers with this type
            for wnum, wt in weapon_types.items():
                if wt == wtype:
                    if wnum not in result.weapons:
                        result.weapons[wnum] = WeaponPieceMapping(weapon_num=wnum)
                    result.weapons[wnum].query_piece = _resolve_piece(piece_var)
                    result.weapons[wnum]._update_all()

    # --- Combined variant: script.AimFromWeapon(weapon) ---
    af_match = re.search(
        r'function\s+script\.AimFromWeapon\s*\(\s*weapon\s*\)(.*?)(?=\nfunction\s|\Z)',
        content, re.DOTALL
    )
    if af_match and weapon_types:
        body = af_match.group(1)
        for branch in re.finditer(
            r'weapons\[weapon\]\s*==\s*["\'](\w+)["\']\s*then\s*\n?\s*return\s+(\w+)',
            body
        ):
            wtype = branch.group(1)
            piece_var = branch.group(2)
            for wnum, wt in weapon_types.items():
                if wt == wtype:
                    if wnum not in result.weapons:
                        result.weapons[wnum] = WeaponPieceMapping(weapon_num=wnum)
                    # "return 0" means aim is disabled (e.g. dgun uses unit center)
                    if piece_var == '0' or piece_var == 'nil':
                        result.weapons[wnum].aim_disabled = True
                    else:
                        result.weapons[wnum].aim_from_piece = _resolve_piece(piece_var)
                    result.weapons[wnum]._update_all()

    # --- Combined variant: script.AimWeapon(weapon, heading, pitch) ---
    aw_match = re.search(
        r'function\s+script\.AimWeapon\s*\(\s*weapon\s*,\s*heading\s*,\s*pitch\s*\)(.*?)(?=\nfunction\s|\Z)',
        content, re.DOTALL
    )
    if aw_match and weapon_types:
        body = aw_match.group(1)
        # Split by weapon type checks — each section from one weapons[weapon]=="type"
        # to the next contains all Turn commands for that weapon type, including
        # nested if/else blocks.
        branches = list(re.finditer(
            r'weapons\[weapon\]\s*==\s*["\'](\w+)["\']',
            body
        ))
        for i, branch in enumerate(branches):
            wtype = branch.group(1)
            start = branch.end()
            # Section ends at the next weapons[weapon] check or end of body
            end = branches[i + 1].start() if i + 1 < len(branches) else len(body)
            branch_body = body[start:end]
            # Strip Lua line comments to avoid matching Turn() in comments
            branch_clean = re.sub(r'--[^\n]*', '', branch_body)
            aim_pieces = set()
            # Match both Turn() and turn() for aim pieces
            for turn_match in re.finditer(r'\b[Tt]urn\s*\(\s*(\w+)', branch_clean):
                piece_var = turn_match.group(1)
                # Skip non-piece identifiers
                if piece_var.lower() in ('heading', 'pitch', 'math', 'rad',
                                          'true', 'false', 'nil', 'self'):
                    continue
                aim_pieces.add(_resolve_piece(piece_var))
            if aim_pieces:
                for wnum, wt in weapon_types.items():
                    if wt == wtype:
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
