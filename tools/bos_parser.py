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

    # 1b. Pre-extract emit-sfx pieces from ShotN/FireWeaponN functions.
    # These explicitly name the pieces that emit muzzle flash effects and are the
    # most reliable indicator of actual fire points — more so than piece-index
    # arithmetic in QueryWeaponN which can break if the piece declaration order
    # doesn't match the intended barrel sequence (e.g. armbrawl).
    _shot_emit_pieces: Dict[int, List[str]] = {}
    _SHOT_PAT = rf'(?:Shot|FireWeapon)(\d+|(?:{_LEGACY_NAMES}))\s*\([^)]*\)\s*(?=\{{)'
    for _shot_suffix, _shot_body in _extract_brace_body(clean, _SHOT_PAT):
        _shot_num_m = re.match(r'\d+', _shot_suffix)
        _shot_leg_m = re.match(rf'({_LEGACY_NAMES})', _shot_suffix, re.IGNORECASE)
        if _shot_num_m:
            _swnum = int(_shot_num_m.group())
        elif _shot_leg_m:
            _swnum = _LEGACY_WEAPON_MAP.get(_shot_leg_m.group(1).lower(), 1)
        else:
            continue
        _emit_refs = re.findall(r'emit-sfx\s+[^;]*?\s+from\s+(\w+)', _shot_body, re.IGNORECASE)
        if _emit_refs:
            _seen: set = set()
            _unique: list = []
            for _p in _emit_refs:
                _pl = _p.lower()
                if _pl in [pp.lower() for pp in result.pieces] and _pl not in _seen:
                    _seen.add(_pl)
                    _unique.append(_pl)
            if _unique:
                _shot_emit_pieces[_swnum] = _unique

    # 2. Extract QueryWeaponN / QueryPrimary / QuerySecondary / ... functions
    # Use brace-matching so if/else chains (multi-barrel cycling) are parsed fully.
    _QUERY_PAT = rf'Query(Weapon(\d+)|({_LEGACY_NAMES}))\s*\([^)]*\)\s*(?=\{{)'
    for suffix, body in _extract_brace_body(clean, _QUERY_PAT):
        num_m = re.match(r'Weapon(\d+)', suffix, re.IGNORECASE)
        leg_m = re.match(rf'({_LEGACY_NAMES})', suffix, re.IGNORECASE)
        wnum = int(num_m.group(1)) if num_m else _LEGACY_WEAPON_MAP.get((leg_m.group(1) if leg_m else '').lower(), 1)
        all_refs = _extract_all_pieces_from_function(body, result.pieces)
        # Popup-defense pattern: QueryWeapon checks is_open and returns a dummy
        # piece (e.g. aimFlare) when closed, real fire piece when open.
        # The viewer always shows the open state, so discard the closed-branch pieces.
        if len(all_refs) > 1 and re.search(r'\b(?:is_open|isOpen|GunOpen|gun_open)\b', body, re.IGNORECASE):
            open_refs = _extract_open_state_pieces(body, result.pieces)
            if open_refs:
                all_refs = open_refs
        # Detect BOS barrel-alternating: "pieceIndex = <piece_or_number> + <variable>"
        # variable cycles 0..N-1 each shot, so piece and N-1 consecutive pieces fire.
        # Also handles "piecenum = <number> + <variable>" (e.g. cormadsam: piecenum = 3 + barrel).
        # Detect cycle count from "if (variable == N) variable = 0;" or
        # "variable = !variable" (toggle → 2).  Fallback: mirror piece l↔r.
        gun_alt = re.search(r'(?:piecenum|pieceIndex)\s*=\s*(\w+)\s*\+\s*(\w+)\s*;', body, re.IGNORECASE)
        if gun_alt and gun_alt.group(2).lower() in result.pieces:
            gun_alt = None  # second token is a piece name, not a variable — skip
        if gun_alt:
            base_token = gun_alt.group(1).lower()
            var_name = gun_alt.group(2).lower()
            base_idx = None
            if base_token in result.pieces:
                base_idx = result.pieces.index(base_token)
            elif base_token.isdigit():
                base_idx = int(base_token)
            if base_idx is not None and base_idx < len(result.pieces):
                # Determine cycle count from the variable's wrap logic in the full script.
                # Pattern: "if (var == N) var = 0;" → N is the count.
                # Find all "var == <num>" comparisons and take the max as the wrap limit.
                # Also check "var > N" patterns (e.g. "barrel > 11" with base 3 → count = 11-3+1 = 9).
                cycle_count = 2  # default: toggle between 2
                # Check toggle first: "var = !var" always means 2
                has_toggle = bool(re.search(
                    rf'{re.escape(var_name)}\s*=\s*!\s*{re.escape(var_name)}', clean, re.IGNORECASE
                ))
                all_limits = [int(m2.group(1)) for m2 in re.finditer(
                    rf'{re.escape(var_name)}\s*==\s*(\d+)', clean, re.IGNORECASE
                )]
                if has_toggle:
                    cycle_count = 2
                elif all_limits:
                    # Check "var > N" pattern: variable resets when > N, so it ranges
                    # from some start value up to N, giving (N - base_idx + 1) pieces
                    # when the base is a numeric offset.
                    gt_limits = [int(m2.group(1)) for m2 in re.finditer(
                        rf'{re.escape(var_name)}\s*>\s*(\d+)', clean, re.IGNORECASE
                    )]
                    if gt_limits and base_token.isdigit():
                        # e.g. base=3, barrel>11 → pieces at indices 3+1..3+8 = 4..10
                        # The variable starts at some min and wraps at >max.
                        # Find the initial assignment to determine start value.
                        init_match = re.search(
                            rf'{re.escape(var_name)}\s*=\s*(\d+)\s*;', clean, re.IGNORECASE
                        )
                        var_start = int(init_match.group(1)) if init_match else 0
                        var_max = max(gt_limits)
                        cycle_count = var_max - var_start + 1
                    elif gt_limits:
                        cycle_count = max(gt_limits)
                    else:
                        has_toggle = bool(re.search(
                            rf'{re.escape(var_name)}\s*=\s*!\s*{re.escape(var_name)}', clean, re.IGNORECASE
                        ))
                        if has_toggle:
                            cycle_count = 2
                # For numeric base with variable offset, the actual piece indices are
                # base_idx + var_start .. base_idx + var_start + cycle_count - 1
                if base_token.isdigit():
                    init_match = re.search(
                        rf'{re.escape(var_name)}\s*=\s*(\d+)\s*;', clean, re.IGNORECASE
                    )
                    var_start = int(init_match.group(1)) if init_match else 0
                    start_idx = base_idx + var_start
                    all_refs = [result.pieces[i] for i in range(start_idx, min(start_idx + cycle_count, len(result.pieces)))]
                else:
                    # Collect consecutive pieces from base_idx
                    all_refs = [result.pieces[i] for i in range(base_idx, min(base_idx + cycle_count, len(result.pieces)))]
                # Sanity check: piece-index arithmetic relies on the BOS piece
                # declaration order matching consecutive barrel pieces.  Some scripts
                # (e.g. armbrawl) have flare1,flare2 at indices 0-1 but flare3,flare4
                # at indices 8-9, so "flare1 + gun_1" overflows into unrelated pieces
                # like "base".  When this happens, prefer the emit-sfx pieces from the
                # corresponding ShotN/FireWeaponN function which explicitly name the
                # real fire-effect pieces.
                _FIRE_PREFIXES = ('flare', 'fire', 'emit', 'muzzle', 'barrel', 'rflare', 'lflare')
                _has_suspect = any(
                    not any(p.startswith(pfx) for pfx in _FIRE_PREFIXES)
                    for p in all_refs
                )
                if _has_suspect and wnum in _shot_emit_pieces:
                    emit_refs = _shot_emit_pieces[wnum]
                    if len(emit_refs) >= len(all_refs):
                        all_refs = emit_refs
        # Fallback: "piecenum = <variable>;" or "pieceIndex = <variable>;" where
        # variable is NOT a piece name. The variable acts as a piece index, cycled
        # in FireWeaponN. Resolve by finding all assignments to that variable.
        # BOS uses "piecenum" (the function argument), some scripts use "pieceIndex".
        if not all_refs:
            var_assign = re.search(r'(?:piecenum|pieceIndex)\s*=\s*(\w+)\s*;', body, re.IGNORECASE)
            if var_assign:
                var_name = var_assign.group(1).lower()
                if var_name not in result.pieces and var_name not in ('piecenum', 'pieceindex'):
                    # Find all assignments: var = <piece_or_number>
                    # Use ordered list to preserve BOS assignment order
                    assigned_pieces_ordered = []
                    assigned_pieces = set()
                    for asgn in re.finditer(
                        rf'\b{re.escape(var_name)}\s*=\s*(\w+)\s*;', clean, re.IGNORECASE
                    ):
                        val = asgn.group(1).lower()
                        if val in result.pieces and val not in assigned_pieces:
                            assigned_pieces.add(val)
                            assigned_pieces_ordered.append(val)
                    if assigned_pieces:
                        # Variable is assigned piece names (e.g. gun_1 = flare1)
                        # Check if there's a ++var pattern → cycling through consecutive pieces
                        has_increment = bool(re.search(
                            rf'\+\+\s*{re.escape(var_name)}', clean, re.IGNORECASE
                        ))
                        has_toggle = bool(re.search(
                            rf'{re.escape(var_name)}\s*=\s*!\s*{re.escape(var_name)}', clean, re.IGNORECASE
                        ))
                        if has_increment and len(assigned_pieces) == 1:
                            # Cycling from initial piece upward — find the limit
                            start_piece = list(assigned_pieces)[0]
                            start_idx = result.pieces.index(start_piece)
                            # Find the maximum comparison value (the reset threshold)
                            limit_vals = [int(m.group(1)) for m in re.finditer(
                                rf'{re.escape(var_name)}\s*==\s*(\d+)', clean, re.IGNORECASE
                            )]
                            count = max(limit_vals) if limit_vals else 2
                            all_refs = [result.pieces[i] for i in range(start_idx, min(start_idx + count, len(result.pieces)))
                                        if result.pieces[i].startswith('flare') or result.pieces[i].startswith('fire')]
                            if not all_refs:
                                all_refs = [result.pieces[i] for i in range(start_idx, min(start_idx + count, len(result.pieces)))]
                        elif has_toggle and len(assigned_pieces) == 1:
                            # Toggle pattern: var = piece; ... var = !var;
                            # BOS !var flips 0↔1, so the variable alternates between
                            # the initial piece (index N) and the next piece (index N+1).
                            start_piece = list(assigned_pieces)[0]
                            start_idx = result.pieces.index(start_piece)
                            all_refs = [result.pieces[i] for i in range(start_idx, min(start_idx + 2, len(result.pieces)))]
                        else:
                            all_refs = assigned_pieces_ordered
                    else:
                        # Variable assigned numeric values (BOS 1-based piece indices).
                        # e.g. guncount = 7; guncount = 10; guncount = 13;
                        # These are 1-based indices into the piece declaration list.
                        has_toggle = bool(re.search(
                            rf'{re.escape(var_name)}\s*=\s*!\s*{re.escape(var_name)}', clean, re.IGNORECASE
                        ))
                        numeric_vals = set()
                        for asgn in re.finditer(
                            rf'\b{re.escape(var_name)}\s*=\s*(\d+)\s*;', clean, re.IGNORECASE
                        ):
                            numeric_vals.add(int(asgn.group(1)))
                        if has_toggle:
                            numeric_vals.update({0, 1})
                        if numeric_vals:
                            # BOS piece indices are 1-based: convert to 0-based for list lookup
                            # First try 1-based (standard BOS convention)
                            refs_1based = [result.pieces[v - 1] for v in sorted(numeric_vals)
                                           if 1 <= v <= len(result.pieces)]
                            # Also try 0-based as fallback
                            refs_0based = [result.pieces[v] for v in sorted(numeric_vals)
                                           if 0 <= v < len(result.pieces)]
                            # Prefer whichever resolves to flare/fire pieces
                            def _has_fire_pieces(refs):
                                return any(p.startswith('flare') or p.startswith('fire') for p in refs)
                            if refs_1based and _has_fire_pieces(refs_1based):
                                all_refs = refs_1based
                            elif refs_0based and _has_fire_pieces(refs_0based):
                                all_refs = refs_0based
                            elif refs_1based:
                                all_refs = refs_1based
                            elif refs_0based:
                                all_refs = refs_0based
        # Final fallback: use emit-sfx fire pieces from FireWeapon/Shot functions
        if not all_refs and wnum in _shot_emit_pieces:
            all_refs = _shot_emit_pieces[wnum]
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


def _extract_open_state_pieces(body: str, known_pieces: List[str]) -> List[str]:
    """For popup-defense QueryWeapon bodies with is_open checks, extract only the
    pieces from the 'open' branch.

    Patterns handled:
      if (is_open == 0) { pieceIndex = DUMMY; } else { pieceIndex = REAL; }
      if (is_open == 1) { pieceIndex = REAL; } else { pieceIndex = DUMMY; }
    """
    _OPEN_VARS = r'(?:is_open|isOpen|GunOpen|gun_open)'
    # Find if-else block testing is_open / GunOpen
    # Pattern: if (var == 0) { ... } else { ... }
    m = re.search(
        r'if\s*\(\s*' + _OPEN_VARS + r'\s*==\s*(\d)\s*\)\s*\{([^}]*)\}\s*else\s*\{([^}]*)\}',
        body, re.IGNORECASE
    )
    if not m:
        # Try: if (!var) { ... } else { ... }
        m = re.search(
            r'if\s*\(\s*!\s*' + _OPEN_VARS + r'\s*\)\s*\{([^}]*)\}\s*else\s*\{([^}]*)\}',
            body, re.IGNORECASE
        )
        if not m:
            # Try: if (var) { ... } else { ... }  (truthy = open)
            m = re.search(
                r'if\s*\(\s*' + _OPEN_VARS + r'\s*\)\s*\{([^}]*)\}\s*else\s*\{([^}]*)\}',
                body, re.IGNORECASE
            )
            if m:
                # var truthy = open state is group 1, closed is group 2
                open_body = m.group(1)
            else:
                return []
        else:
            # !var = closed state is group 1, open state is group 2
            open_body = m.group(2)
    else:
        val = int(m.group(1))
        # is_open == 0 → group 2 is closed, group 3 is open
        # is_open == 1 → group 2 is open, group 3 is closed
        open_body = m.group(3) if val == 0 else m.group(2)
    return _extract_all_pieces_from_function(open_body, known_pieces)


def _extract_all_pieces_from_function(body: str, known_pieces: List[str]) -> List[str]:
    """Extract ALL piece names from a QueryWeapon body (for multi-barrel weapons).

    Some units have QueryWeapon functions with if/else chains that assign different
    fire point pieces per shot (e.g. 6-tube missile launcher cycling through flare1..6).
    This returns all distinct pieces in BOS assignment order (the order they appear
    in the function body), which matches the barrel cycling order used by animations.
    """
    known_lower = {p.lower() for p in known_pieces}
    found_set = set()
    found = []
    # Find all "pieceIndex = <piece>;" or "piecenum = <piece>;" assignments in order
    for m in re.finditer(r'(?:pieceIndex|piecenum)\s*=\s*(\w+)\s*;', body, re.IGNORECASE):
        p = m.group(1).lower()
        if p in known_lower and p not in found_set:
            found_set.add(p)
            found.append(p)
    if found:
        return found
    # Fallback: scan for any piece names in body order
    for m in re.finditer(r'\b(\w+)\b', body):
        p = m.group(1).lower()
        if p in known_lower and p not in found_set:
            found_set.add(p)
            found.append(p)
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
