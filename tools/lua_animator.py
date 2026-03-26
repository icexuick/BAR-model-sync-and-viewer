"""
LUA/LUS → glTF Animation Extractor

Extracts keyframe animation data from BAR/Spring LUS (Lua Unit Script) files.
LUS scripts are the Lua equivalent of BOS scripts, used primarily for commander
units (armcom, corcom) and some newer units.

LUS Walk() structure (Skeletor_S3O export format):
  function walk()
      if (bMoving) then --Frame:4
          turn(piece, axis, goal, speed/animSpeed)
          Sleep( (33*animSpeed) -1)
      end
      while(bMoving) do
          if (bMoving) then --Frame:8
              ...
              Sleep( (33*animSpeed) -1)
          end
      end
  end

Key differences from BOS:
- Two calling conventions:
  1. lowercase turn(piece, axis, goal, speed) — custom wrapper that converts
     degrees to radians and negates z-axis (axis 3). Values are in DEGREES.
  2. uppercase Turn(piece, axis_const, goal, speed) — native Spring API,
     values are already in RADIANS.
- Piece variables declared via: local head, ... = piece("head", ...)
- Axis constants: x_axis=1, y_axis=2, z_axis=3 (1-based, vs BOS x/y/z-axis)
- Sleep() instead of sleep()
- bMoving instead of isMoving

Output: BosTrack/BosKeyframe objects compatible with the existing GLB builder.
"""

import re
import math
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field

# Re-use the same data structures as bos_animator
from bos_animator import BosTrack, BosKeyframe

AXIS_MAP_NUMERIC = {1: 0, 2: 1, 3: 2}   # LUS 1-based → 0-based
AXIS_MAP_NAMED = {'x_axis': 0, 'y_axis': 1, 'z_axis': 2}
FPS = 30.0


# ---------------------------------------------------------------------------
# Regex patterns for LUS commands
# ---------------------------------------------------------------------------

# Frame markers: --Frame:N or -- Frame: N or //Frame:N
_FRAME_RE = re.compile(r'(?://|--)\s*Frame\s*:?\s*(\d+)', re.IGNORECASE)

# lowercase turn(piece, axis, goal, speed) — custom wrapper, values in degrees
# Examples:
#   turn(head, 1, -2.620635, 39.654598/animSpeed)
#   turn(lthigh, 3,  10.085981, 605.673206/animSpeed)
#   if (leftArm) then turn(biggun, 1, -48.215180, 113.735764/animSpeed) end
_TURN_LOWER_RE = re.compile(
    r'\bturn\s*\(\s*(\w+)\s*,\s*(\d)\s*,\s*([-\d.]+)',
    re.IGNORECASE
)

# uppercase Turn(piece, axis_const, goal, speed) — native API, values in radians
# Examples:
#   Turn(head, y_axis, 0.104720, 3.141593 * speedMult)
#   Turn(lfoot, x_axis, -0.034375, 2.583373 * speedMult)
_TURN_UPPER_RE = re.compile(
    r'\bTurn\s*\(\s*(\w+)\s*,\s*([xyz]_axis|\d)\s*,\s*([-\d.]+)',
)

# lowercase move(piece, axis, goal, speed) — wrapper, engine units
# Examples:
#   move(pelvis, 2, -1.000000, 6.944444)
#   move (pelvis, 2,  -2.000000 , 25.555551 /animSpeed)
_MOVE_LOWER_RE = re.compile(
    r'\bmove\s*\(\s*(\w+)\s*,\s*(\d)\s*,\s*([-\d.]+)',
    re.IGNORECASE
)

# uppercase Move(piece, axis_const, goal, speed) — native API, engine units
# Examples:
#   Move(pelvis, x_axis, - 2.217917, 66.537516 * speedMult)
#   Move(pelvis, y_axis, -2.880190, 56.405711 * speedMult)
_MOVE_UPPER_RE = re.compile(
    r'\bMove\s*\(\s*(\w+)\s*,\s*([xyz]_axis|\d)\s*,\s*([-\d.e]+)',
)

# Spin(piece, axis_const, speed) — in script.Create()
# Example: Spin(dish, 2, 2.5)  or  Spin(dish, y_axis, 2.5)
_SPIN_RE = re.compile(
    r'\bSpin\s*\(\s*(\w+)\s*,\s*([xyz]_axis|\d)\s*,\s*([-\d.]+)',
)

# Hide(piece) — in script.Create()
_HIDE_RE = re.compile(r'\bHide\s*\(\s*(\w+)\s*\)', re.IGNORECASE)

# Sleep((33*animSpeed)-1) or Sleep(sleepTime) or Sleep(N)
_SLEEP_RE = re.compile(r'\bSleep\s*\(\s*(?:\(\s*(\d+)\s*\*\s*animSpeed\s*\)\s*-\s*1|(\d+))\s*\)',
                        re.IGNORECASE)

# piece() declarations: local a, b, c = piece("a", "b", "c")
_PIECE_DECL_RE = re.compile(
    r'local\s+([\w\s,]+?)\s*=\s*piece\s*\((.*?)\)',
    re.DOTALL
)

# Weapon table: weapons = { [1] = "laser", ... }
_WEAPON_TABLE_RE = re.compile(
    r'\bweapons\s*=\s*\{(.*?)\}',
    re.DOTALL
)
_WEAPON_ENTRY_RE = re.compile(r'\[(\d+)\]\s*=\s*["\'](\w+)["\']')

# script.QueryWeapon — return piece for weapon
_QUERY_WEAPON_RE = re.compile(
    r'function\s+script\.QueryWeapon\s*\(\s*weapon\s*\)(.*?)(?=\nfunction\s|\Z)',
    re.DOTALL
)

# script.AimFromWeapon — return piece for weapon
_AIM_FROM_RE = re.compile(
    r'function\s+script\.AimFromWeapon\s*\(\s*weapon\s*\)(.*?)(?=\nfunction\s|\Z)',
    re.DOTALL
)


def _resolve_axis(axis_str: str) -> int:
    """Convert axis string to 0-based index."""
    if axis_str in AXIS_MAP_NAMED:
        return AXIS_MAP_NAMED[axis_str]
    try:
        return AXIS_MAP_NUMERIC[int(axis_str)]
    except (ValueError, KeyError):
        return 0


def _parse_piece_names(lua_content: str) -> Dict[str, str]:
    """
    Parse piece() declarations to map variable names to actual piece names.
    Returns {variable_name: piece_string_name}.
    """
    result = {}
    for m in _PIECE_DECL_RE.finditer(lua_content):
        vars_str = m.group(1)
        pieces_str = m.group(2)
        var_names = [v.strip() for v in vars_str.split(',') if v.strip()]
        piece_names = re.findall(r'["\'](\w+)["\']', pieces_str)
        for var, pname in zip(var_names, piece_names):
            result[var] = pname
    return result


def _detect_turn_convention(lua_content: str) -> str:
    """
    Detect whether the walk function uses the lowercase turn() wrapper
    (degrees, custom z-negate) or uppercase Turn() (radians, native).

    Returns 'lower' or 'upper'.
    """
    # Look at the walk function specifically
    walk_body = _extract_lua_function_body(lua_content, 'walk')
    if not walk_body:
        walk_body = lua_content

    lower_count = len(_TURN_LOWER_RE.findall(walk_body))
    upper_count = len(_TURN_UPPER_RE.findall(walk_body))

    return 'upper' if upper_count > lower_count else 'lower'


def _extract_lua_function_body(content: str, func_name: str) -> Optional[str]:
    """
    Extract the body of a Lua function.
    Handles both:
      function funcname() ... end
      local function funcname() ... end
      function script.FuncName() ... end
    """
    # Try various patterns
    patterns = [
        rf'(?:local\s+)?function\s+(?:script\.)?{re.escape(func_name)}\s*\([^)]*\)',
    ]

    for pat in patterns:
        match = re.search(pat, content, re.IGNORECASE)
        if not match:
            continue

        # Find the body by counting end/function nesting
        start = match.end()
        depth = 1
        pos = start

        # Simple Lua block tracking: function/if/while/for/do increase depth,
        # 'end' decreases it.
        # We need to be careful with string literals and comments.
        while pos < len(content) and depth > 0:
            # Skip comments
            if content[pos:pos+2] == '--':
                if content[pos:pos+4] == '--[[':
                    # Block comment
                    end_comment = content.find(']]', pos + 4)
                    pos = end_comment + 2 if end_comment != -1 else len(content)
                    continue
                else:
                    # Line comment
                    end_line = content.find('\n', pos)
                    pos = end_line + 1 if end_line != -1 else len(content)
                    continue

            # Skip string literals
            if content[pos] in ('"', "'"):
                quote = content[pos]
                pos += 1
                while pos < len(content) and content[pos] != quote:
                    if content[pos] == '\\':
                        pos += 1  # skip escaped char
                    pos += 1
                pos += 1  # skip closing quote
                continue

            # Check for keywords that increase nesting
            # Need to match whole words only
            rest = content[pos:]
            for kw in ('function', 'if', 'while', 'for', 'do'):
                if rest[:len(kw)] == kw and (pos == 0 or not content[pos-1].isalnum()) \
                        and (pos + len(kw) >= len(content) or not content[pos + len(kw)].isalnum()):
                    # Special: 'do' after 'while ... do' or 'for ... do' is already counted
                    # by the while/for keyword, so skip standalone 'do'
                    if kw == 'do':
                        # Only count 'do' if it's not preceded by while/for on the same line
                        line_start = content.rfind('\n', 0, pos) + 1
                        line_before = content[line_start:pos].strip()
                        if re.search(r'\b(?:while|for)\b', line_before):
                            pos += len(kw)
                            break
                    # 'if' only counts if it's followed by 'then'
                    # Actually in Lua, all blocks end with 'end', so we count:
                    # function, if...then, while...do, for...do, do...end
                    if kw in ('function', 'if', 'while', 'for'):
                        depth += 1
                    elif kw == 'do':
                        depth += 1
                    pos += len(kw)
                    break
            else:
                if rest[:3] == 'end' and (pos == 0 or not content[pos-1].isalnum()) \
                        and (pos + 3 >= len(content) or not content[pos + 3].isalnum()):
                    depth -= 1
                    if depth == 0:
                        return content[start:pos]
                    pos += 3
                else:
                    pos += 1

        if depth == 0:
            return content[start:pos - 3]  # Already returned above

    return None


def _parse_lua_frame_blocks(text: str, convention: str = 'lower',
                            piece_map: Dict[str, str] = None) -> List[Tuple[int, Dict[tuple, float]]]:
    """
    Split text on Frame:N markers and parse turn/move commands per block.

    convention: 'lower' = custom wrapper (degrees), 'upper' = native Turn (radians)
    piece_map: variable name → piece string name mapping

    Returns list of (frame_number, {(piece_name, axis, is_rotation): value_in_degrees}).
    All values are converted to degrees for compatibility with the existing GLB builder.
    """
    if piece_map is None:
        piece_map = {}

    frame_positions = [(m.start(), int(m.group(1))) for m in _FRAME_RE.finditer(text)]
    if not frame_positions:
        return []

    blocks = []
    for i, (pos, frame_num) in enumerate(frame_positions):
        end_pos = frame_positions[i + 1][0] if i + 1 < len(frame_positions) else len(text)
        block_text = text[pos:end_pos]

        commands: Dict[tuple, float] = {}

        if convention == 'lower':
            # lowercase turn(piece, axis_num, goal_degrees, speed)
            for m in _TURN_LOWER_RE.finditer(block_text):
                var_name = m.group(1)
                piece_name = piece_map.get(var_name, var_name).lower()
                axis = _resolve_axis(m.group(2))
                value_deg = float(m.group(3))
                # The custom wrapper negates z-axis (axis 3 = index 2)
                # Since BOS also negates z (COBWTF), and the GLB builder already
                # handles the BOS convention, we need to UN-negate z here so the
                # builder's negate produces the correct result.
                # Actually: the LUS wrapper does: Turn(piece, axis, -rad(goal)) for z
                # And the GLB builder does: rz = -math.radians(rz_deg) (COBWTF negate)
                # So if LUS already negated z, and GLB builder will negate again,
                # we get double-negation = positive. We need to pass the ORIGINAL
                # degree value as-is — the wrapper's negate and the builder's negate cancel.
                # So: just pass the raw degree value for ALL axes.
                commands[(piece_name, axis, True)] = value_deg

            # lowercase move(piece, axis_num, goal, speed)
            for m in _MOVE_LOWER_RE.finditer(block_text):
                var_name = m.group(1)
                piece_name = piece_map.get(var_name, var_name).lower()
                axis = _resolve_axis(m.group(2))
                value = float(m.group(3))
                commands[(piece_name, axis, False)] = value

        else:
            # uppercase Turn(piece, axis_const, goal_radians, speed)
            for m in _TURN_UPPER_RE.finditer(block_text):
                var_name = m.group(1)
                piece_name = piece_map.get(var_name, var_name).lower()
                axis = _resolve_axis(m.group(2))
                value_rad = float(m.group(3))
                value_deg = math.degrees(value_rad)
                # Native Turn() does NOT negate z. The GLB builder DOES negate z
                # (COBWTF convention). So for native Turn, we need to negate z
                # ourselves so the builder's negate produces the correct result.
                if axis == 2:
                    value_deg = -value_deg
                commands[(piece_name, axis, True)] = value_deg

            # uppercase Move(piece, axis_const, goal, speed)
            for m in _MOVE_UPPER_RE.finditer(block_text):
                var_name = m.group(1)
                piece_name = piece_map.get(var_name, var_name).lower()
                axis = _resolve_axis(m.group(2))
                value = float(m.group(3))
                commands[(piece_name, axis, False)] = value

        if commands:
            blocks.append((frame_num, commands))

    return blocks


def _extract_while_body_lua(body: str) -> Tuple[str, str]:
    """
    Split a walk() body into (pre_while_text, while_loop_body).
    Handles Lua while...do...end syntax.
    """
    # Find 'while' keyword
    match = re.search(r'\bwhile\s*\(', body)
    if not match:
        match = re.search(r'\bwhile\s+', body)
    if not match:
        return body, ''

    while_start = match.start()

    # Find the 'do' keyword after while condition
    do_match = re.search(r'\bdo\b', body[match.end():])
    if not do_match:
        return body, ''

    do_pos = match.end() + do_match.end()

    # Now find the matching 'end' for this while block
    depth = 1
    pos = do_pos
    while pos < len(body) and depth > 0:
        # Skip comments
        if body[pos:pos+2] == '--':
            end_line = body.find('\n', pos)
            pos = end_line + 1 if end_line != -1 else len(body)
            continue
        # Skip strings
        if body[pos] in ('"', "'"):
            quote = body[pos]
            pos += 1
            while pos < len(body) and body[pos] != quote:
                if body[pos] == '\\':
                    pos += 1
                pos += 1
            pos += 1
            continue

        rest = body[pos:]
        # Check block openers
        for kw in ('function', 'if', 'while', 'for'):
            if rest[:len(kw)] == kw and (pos == 0 or not body[pos-1].isalnum()) \
                    and (pos + len(kw) >= len(body) or not body[pos + len(kw)].isalnum()):
                depth += 1
                pos += len(kw)
                break
        else:
            if rest[:3] == 'end' and (pos == 0 or not body[pos-1].isalnum()) \
                    and (pos + 3 >= len(body) or not body[pos + 3].isalnum()):
                depth -= 1
                if depth == 0:
                    pre = body[:while_start]
                    loop_body = body[do_pos:pos]
                    return pre, loop_body
                pos += 3
            # Check for 'do' after while/for (already counted by while/for)
            elif rest[:2] == 'do' and (pos == 0 or not body[pos-1].isalnum()) \
                    and (pos + 2 >= len(body) or not body[pos + 2].isalnum()):
                # Only count standalone 'do' blocks, not while/for...do
                line_start = body.rfind('\n', 0, pos) + 1
                line_before = body[line_start:pos].strip()
                if not re.search(r'\b(?:while|for)\b', line_before):
                    depth += 1
                pos += 2
            else:
                pos += 1

    return body, ''


# ---------------------------------------------------------------------------
# Public API — mirrors bos_animator interface
# ---------------------------------------------------------------------------

def extract_lua_walk_animation(lua_content: str) -> Optional[Tuple[str, List[BosTrack], Dict]]:
    """
    Extract walk animation tracks from a LUS script.

    Returns (animation_name, List[BosTrack], now_rots) or None if not found.
    Compatible with bos_animator.extract_walk_animation() output format.
    """
    piece_map = _parse_piece_names(lua_content)
    convention = _detect_turn_convention(lua_content)

    # Try extracting the walk function
    walk_body = _extract_lua_function_body(lua_content, 'walk')
    if not walk_body:
        return None

    pre_body, while_body = _extract_while_body_lua(walk_body)
    if not while_body:
        while_body = walk_body
        pre_body = ''

    # Parse pre-loop frames
    pre_blocks = _parse_lua_frame_blocks(pre_body, convention, piece_map)
    pre_cmds: Dict[tuple, float] = {}
    for _, cmds in pre_blocks:
        pre_cmds.update(cmds)

    # Parse loop frames
    loop_blocks = _parse_lua_frame_blocks(while_body, convention, piece_map)
    if not loop_blocks:
        return None

    # Determine step_size from frame number diffs
    frame_nums = [fn for fn, _ in loop_blocks]
    diffs = [(frame_nums[i + 1] - frame_nums[i]) % 300
             for i in range(len(frame_nums) - 1)]
    if diffs:
        from collections import Counter
        step_size = Counter(diffs).most_common(1)[0][0]
    else:
        step_size = 4  # Default for LUS (33ms frames)

    # Build tracks with pre-loop as t=0 and closing keyframe
    if pre_cmds:
        # Detect duplicate closing frame
        last_cmds = loop_blocks[-1][1]
        shared_keys = set(pre_cmds) & set(last_cmds)
        n_matching = sum(1 for k in shared_keys if abs(pre_cmds[k] - last_cmds[k]) < 0.5)
        if shared_keys and n_matching >= len(shared_keys) * 0.85 \
                and len(shared_keys) >= len(pre_cmds) * 0.6:
            active_loop = loop_blocks[:-1]
            print(f"  Dropping duplicate closing frame (Frame {loop_blocks[-1][0]} == pre-loop)")
        else:
            active_loop = loop_blocks

        n_loop = len(active_loop)
        duration = (n_loop + 1) * step_size / FPS

        track_dict: Dict[tuple, List[BosKeyframe]] = {}

        # t=0 → pre-loop
        for key, value in pre_cmds.items():
            track_dict.setdefault(key, []).append(BosKeyframe(time=0.0, value=value))

        # t=step, 2*step, ... → loop frames
        for loop_idx, (frame_num, commands) in enumerate(active_loop):
            t = (loop_idx + 1) * step_size / FPS
            for key, value in commands.items():
                track_dict.setdefault(key, []).append(BosKeyframe(time=t, value=value))

        # Tracks in loop but not pre-loop: add t=0 with first value
        first_loop_cmds = active_loop[0][1] if active_loop else {}
        for key, value in first_loop_cmds.items():
            if key not in pre_cmds and key in track_dict:
                track_dict[key].insert(0, BosKeyframe(time=0.0, value=value))

    else:
        # No pre-loop: use loop frames directly
        n_loop = len(loop_blocks)
        duration = n_loop * step_size / FPS

        track_dict = {}
        for loop_idx, (frame_num, commands) in enumerate(loop_blocks):
            t = loop_idx * step_size / FPS
            for key, value in commands.items():
                track_dict.setdefault(key, []).append(BosKeyframe(time=t, value=value))

    if not track_dict:
        return None

    # Build BosTrack objects, sort, deduplicate, add closing keyframe
    tracks = []
    for (piece, axis, is_rot), keyframes in track_dict.items():
        keyframes.sort(key=lambda k: k.time)

        # Deduplicate same-time entries (keep last)
        deduped: List[BosKeyframe] = []
        for kf in keyframes:
            if deduped and abs(deduped[-1].time - kf.time) < 1e-5:
                deduped[-1] = kf
            else:
                deduped.append(kf)

        # Add closing keyframe (loop back to first value)
        if deduped and abs(deduped[-1].time - duration) > 1e-5:
            deduped.append(BosKeyframe(time=duration, value=deduped[0].value))

        if len(deduped) >= 2:
            tracks.append(BosTrack(
                piece=piece, axis=axis, is_rotation=is_rot,
                keyframes=deduped
            ))

    if not tracks:
        return None

    # Parse rest-pose rotations from StopWalking
    now_rots = _parse_lua_stopwalking_pose(lua_content, convention, piece_map)

    n_blocks = len(loop_blocks) + (1 if pre_cmds else 0)
    print(f"  LUA Animation 'walk': {len(tracks)} tracks, "
          f"{n_blocks} keyframes, duration {duration:.2f}s (convention={convention})")

    return 'walk', tracks, now_rots


def _parse_lua_stopwalking_pose(lua_content: str, convention: str,
                                 piece_map: Dict[str, str]) -> Dict[tuple, float]:
    """Extract rest-pose rotations from StopWalking()."""
    result: Dict[tuple, float] = {}
    body = _extract_lua_function_body(lua_content, 'StopWalking')
    if not body:
        return result

    if convention == 'lower':
        for m in _TURN_LOWER_RE.finditer(body):
            var_name = m.group(1)
            piece_name = piece_map.get(var_name, var_name).lower()
            axis = _resolve_axis(m.group(2))
            value_deg = float(m.group(3))
            if abs(value_deg) > 20.0:
                result[(piece_name, axis, True)] = value_deg
    else:
        for m in _TURN_UPPER_RE.finditer(body):
            var_name = m.group(1)
            piece_name = piece_map.get(var_name, var_name).lower()
            axis = _resolve_axis(m.group(2))
            value_deg = math.degrees(float(m.group(3)))
            if axis == 2:
                value_deg = -value_deg
            if abs(value_deg) > 20.0:
                result[(piece_name, axis, True)] = value_deg

    if result:
        print(f"  LUA StopWalking: {len(result)} rest-pose rotations")
    return result


def extract_lua_stopwalking_tracks(lua_content: str) -> Optional[List[BosTrack]]:
    """
    Extract StopWalking pose as BosTrack objects (single keyframe at t=0).
    Compatible with bos_animator.extract_stopwalking_pose().
    """
    piece_map = _parse_piece_names(lua_content)
    convention = _detect_turn_convention(lua_content)

    body = _extract_lua_function_body(lua_content, 'StopWalking')
    if not body:
        return None

    commands: Dict[tuple, float] = {}

    if convention == 'lower':
        for m in _TURN_LOWER_RE.finditer(body):
            var_name = m.group(1)
            piece_name = piece_map.get(var_name, var_name).lower()
            axis = _resolve_axis(m.group(2))
            commands[(piece_name, axis, True)] = float(m.group(3))
        for m in _MOVE_LOWER_RE.finditer(body):
            var_name = m.group(1)
            piece_name = piece_map.get(var_name, var_name).lower()
            axis = _resolve_axis(m.group(2))
            commands[(piece_name, axis, False)] = float(m.group(3))
    else:
        for m in _TURN_UPPER_RE.finditer(body):
            var_name = m.group(1)
            piece_name = piece_map.get(var_name, var_name).lower()
            axis = _resolve_axis(m.group(2))
            value_deg = math.degrees(float(m.group(3)))
            if axis == 2:
                value_deg = -value_deg
            commands[(piece_name, axis, True)] = value_deg
        for m in _MOVE_UPPER_RE.finditer(body):
            var_name = m.group(1)
            piece_name = piece_map.get(var_name, var_name).lower()
            axis = _resolve_axis(m.group(2))
            commands[(piece_name, axis, False)] = float(m.group(3))

    if not commands:
        return None

    tracks = []
    for (piece, axis, is_rot), value in commands.items():
        tracks.append(BosTrack(
            piece=piece, axis=axis, is_rotation=is_rot,
            keyframes=[BosKeyframe(time=0.0, value=value),
                       BosKeyframe(time=0.033, value=value)]
        ))
    print(f"  LUA StopWalking pose: {len(tracks)} tracks")
    return tracks


def extract_lua_spin_animations(lua_content: str) -> List[Tuple[str, List[BosTrack]]]:
    """
    Extract Spin() commands from script.Create().
    Returns list of (clip_name, [BosTrack]) — one clip per spinning piece.

    Spin(piece, axis, speed) where speed is in radians/frame at ~30 sim fps.
    Generates 8+ keyframes at ≤45° steps for smooth quaternion slerp,
    matching the format expected by GLBBuilder.add_spin_animation().
    """
    piece_map = _parse_piece_names(lua_content)

    # Look for Spin() in Create() body
    create_body = _extract_lua_function_body(lua_content, 'Create')
    if not create_body:
        return []

    # Piece name fragments to exclude (non-visual spinners)
    _EXCLUDE_FRAGMENTS = (
        'cagelight', 'light', 'emit', 'blink', 'pad',
        'screw', 'belt', 'nano', 'flare', 'fire', 'glow', 'spark',
    )

    # Collect all spins, grouped by piece
    from collections import defaultdict
    piece_axes: Dict[str, Dict[int, float]] = defaultdict(dict)

    for m in _SPIN_RE.finditer(create_body):
        var_name = m.group(1)
        piece_name = piece_map.get(var_name, var_name).lower()
        axis = _resolve_axis(m.group(2))
        speed_rad = float(m.group(3))  # radians per second (Spring Lua Spin API)

        if abs(speed_rad) < 0.01:
            continue
        if any(frag in piece_name for frag in _EXCLUDE_FRAGMENTS):
            continue

        # Convert to degrees per second
        deg_per_sec = math.degrees(speed_rad)
        piece_axes[piece_name][axis] = deg_per_sec

    clips = []
    STEP_DEG = 45.0

    for piece, axes in piece_axes.items():
        # Compute duration for one full revolution (or LCM for multi-axis)
        periods = [abs(360.0 / spd) for spd in axes.values()]
        duration = periods[0]
        for p in periods[1:]:
            # LCM of periods
            a_ms = round(duration * 1000)
            b_ms = round(p * 1000)
            if a_ms > 0 and b_ms > 0:
                from math import gcd
                duration = (a_ms * b_ms // gcd(a_ms, b_ms)) / 1000.0
        duration = min(duration, 24.0)

        piece_tracks: List[BosTrack] = []
        for axis, speed in axes.items():
            sign = 1.0 if speed > 0 else -1.0
            total_deg = speed * duration
            n_steps = max(8, int(math.ceil(abs(total_deg) / STEP_DEG)))
            keyframes = []
            for i in range(n_steps):
                t = duration * i / n_steps
                deg = total_deg * i / n_steps
                keyframes.append(BosKeyframe(time=t, value=deg))
            piece_tracks.append(BosTrack(
                piece=piece, axis=axis, is_rotation=True,
                keyframes=keyframes
            ))

        if piece_tracks:
            clip_name = f"spin_{piece}"
            clips.append((clip_name, piece_tracks))
            print(f"  LUA Spin: {piece} axes={list(axes.keys())} duration={duration:.2f}s")

    return clips


def extract_lua_hide_pieces(lua_content: str) -> Set[str]:
    """
    Return the set of piece names hidden via Hide(piece) in script.Create().
    """
    piece_map = _parse_piece_names(lua_content)

    create_body = _extract_lua_function_body(lua_content, 'Create')
    if not create_body:
        return set()

    hidden = set()
    for m in _HIDE_RE.finditer(create_body):
        var_name = m.group(1)
        piece_name = piece_map.get(var_name, var_name).lower()
        hidden.add(piece_name)

    return hidden


def _inline_lua_calls(body: str, lua_content: str, depth: int = 3) -> str:
    """
    Inline simple function calls in a Lua body.
    Replaces `funcname()` with the body of `function funcname()...end`.
    Only handles no-argument calls. Recurses up to `depth` levels.
    """
    if depth <= 0:
        return body

    # Find bare function calls: word() that are not Turn/Move/Sleep/etc
    _SKIP = {'turn', 'move', 'Turn', 'Move', 'Spin', 'Sleep', 'Show', 'Hide',
             'Signal', 'SetSignalMask', 'StartThread', 'WaitForTurn', 'WaitForMove',
             'piece', 'print', 'return', 'Spring', 'math', 'pairs', 'ipairs',
             'require', 'include', 'dofile'}

    call_re = re.compile(r'\b(\w+)\s*\(\s*\)')
    changed = True
    iterations = 0
    while changed and iterations < depth:
        changed = False
        iterations += 1
        for m in call_re.finditer(body):
            func_name = m.group(1)
            if func_name in _SKIP:
                continue
            # Try to find the function definition
            func_body = _extract_lua_function_body(lua_content, func_name)
            if func_body:
                body = body[:m.start()] + func_body + body[m.end():]
                changed = True
                break  # restart after replacement since positions shifted

    return body


def extract_lua_fire_animations(lua_content: str) -> Optional[List[Tuple[str, List[BosTrack], None]]]:
    """
    Extract fire/recoil animations from script.FireWeapon().

    LUS FireWeapon uses a combined variant dispatching on weapons[weapon]:
        function script.FireWeapon(weapon)
            if weapons[weapon] == "dgun" then
                turn(luparm, 1, 20)          -- instant (no speed)
                move(barrel, 2, -1.5)        -- instant
                turn(luparm, 1, 5, 100)      -- animated (has speed, in degrees/s)
                move(barrel, 2, 0, 5)        -- animated (has speed, in engine units/s)
            end
        end

    Instant commands (no speed) → keyframe at t=0.
    Speed commands → keyframe at t=duration (based on distance/speed).

    Returns list of (clip_name, tracks, None) triples compatible with BOS FireClipInfo.
    The third element is always None (no rotary support for LUS yet).
    """
    piece_map = _parse_piece_names(lua_content)
    convention = _detect_turn_convention(lua_content)

    # Parse weapons table: {number: type_name}
    weapons_table: Dict[int, str] = {}
    wt_match = _WEAPON_TABLE_RE.search(lua_content)
    if wt_match:
        for em in _WEAPON_ENTRY_RE.finditer(wt_match.group(1)):
            weapons_table[int(em.group(1))] = em.group(2)

    # Extract FireWeapon body
    fire_body = _extract_lua_function_body(lua_content, 'script.FireWeapon')
    if not fire_body:
        # Try numbered variants: script.FireWeapon1, etc.
        fire_body = _extract_lua_function_body(lua_content, 'FireWeapon1')

    if not fire_body:
        return None

    # Split into weapon-type branches
    branches: Dict[str, str] = {}

    # Combined variant: weapons[weapon] == "type" branches
    branch_positions = list(re.finditer(
        r'weapons\s*\[\s*weapon\s*\]\s*==\s*["\'](\w+)["\']', fire_body))

    if branch_positions:
        for i, bm in enumerate(branch_positions):
            weapon_type = bm.group(1)
            start = bm.end()
            if i + 1 < len(branch_positions):
                end = branch_positions[i + 1].start()
            else:
                end = len(fire_body)
            # Strip "then" and leading whitespace
            branch_body = re.sub(r'^\s*then\s*', '', fire_body[start:end])
            branches[weapon_type] = _inline_lua_calls(branch_body, lua_content)
    else:
        # Single body (no dispatch) — assign to weapon 1
        branches['weapon_1'] = _inline_lua_calls(fire_body, lua_content)

    # Regex for turn/move WITH optional speed capture
    # lowercase turn(piece, axis, goal) or turn(piece, axis, goal, speed)
    _TURN_FIRE_RE = re.compile(
        r'\bturn\s*\(\s*(\w+)\s*,\s*(\d)\s*,\s*([-\d.]+)(?:\s*,\s*([-\d.]+))?\s*\)',
        re.IGNORECASE
    )
    # lowercase move(piece, axis, goal) or move(piece, axis, goal, speed)
    _MOVE_FIRE_RE = re.compile(
        r'\bmove\s*\(\s*(\w+)\s*,\s*(\d)\s*,\s*([-\d.]+)(?:\s*,\s*([-\d.]+))?\s*\)',
        re.IGNORECASE
    )
    # uppercase Turn(piece, axis_const, goal) or Turn(piece, axis_const, goal, speed)
    _TURN_UPPER_FIRE_RE = re.compile(
        r'\bTurn\s*\(\s*(\w+)\s*,\s*([xyz]_axis|\d)\s*,\s*([-\d.e]+)(?:\s*,\s*([-\d.e]+))?\s*\)'
    )
    # uppercase Move(piece, axis_const, goal) or Move(piece, axis_const, goal, speed)
    _MOVE_UPPER_FIRE_RE = re.compile(
        r'\bMove\s*\(\s*(\w+)\s*,\s*([xyz]_axis|\d)\s*,\s*([-\d.e]+)(?:\s*,\s*([-\d.e]+))?\s*\)'
    )
    # Sleep(N) in ms
    _SLEEP_FIRE_RE = re.compile(r'\bSleep\s*\(\s*(\d+)\s*\)', re.IGNORECASE)

    clips = []

    # Find the lowest weapon number for each weapon type
    type_to_min_num: Dict[str, int] = {}
    for wnum, wtype in weapons_table.items():
        if wtype not in type_to_min_num or wnum < type_to_min_num[wtype]:
            type_to_min_num[wtype] = wnum

    for weapon_type, body in branches.items():
        # Strip Lua comments
        body_clean = re.sub(r'--[^\n]*', '', body)

        # Tokenize: collect commands and sleeps in order
        tokens = []  # ('cmd', key, target, speed_or_None) or ('sleep', ms)

        # Find all commands and sleeps with their positions
        all_matches = []

        if convention == 'lower':
            for m in _TURN_FIRE_RE.finditer(body_clean):
                var_name = m.group(1)
                piece_name = piece_map.get(var_name, var_name).lower()
                axis = _resolve_axis(m.group(2))
                goal_deg = float(m.group(3))
                speed_str = m.group(4)

                # Z-axis handling for the lowercase wrapper:
                # With speed: wrapper negates z → BOS convention → pass raw degrees
                # Without speed: wrapper does NOT negate z → need pre-negate for builder
                if axis == 2 and speed_str is None:
                    goal_deg = -goal_deg

                if speed_str is not None:
                    # Speed in degrees/second (the wrapper converts to rad)
                    speed = float(speed_str)
                else:
                    speed = None  # instant
                all_matches.append((m.start(), 'cmd', (piece_name, axis, True),
                                    goal_deg, speed))

            for m in _MOVE_FIRE_RE.finditer(body_clean):
                var_name = m.group(1)
                piece_name = piece_map.get(var_name, var_name).lower()
                axis = _resolve_axis(m.group(2))
                goal = float(m.group(3))
                speed_str = m.group(4)
                speed = float(speed_str) if speed_str is not None else None
                all_matches.append((m.start(), 'cmd', (piece_name, axis, False),
                                    goal, speed))
        else:
            # uppercase convention
            for m in _TURN_UPPER_FIRE_RE.finditer(body_clean):
                var_name = m.group(1)
                piece_name = piece_map.get(var_name, var_name).lower()
                axis = _resolve_axis(m.group(2))
                goal_rad = float(m.group(3))
                goal_deg = math.degrees(goal_rad)
                speed_str = m.group(4)

                # Uppercase Turn never negates z → pre-negate for builder
                if axis == 2:
                    goal_deg = -goal_deg

                if speed_str is not None:
                    speed = math.degrees(float(speed_str))
                else:
                    speed = None
                all_matches.append((m.start(), 'cmd', (piece_name, axis, True),
                                    goal_deg, speed))

            for m in _MOVE_UPPER_FIRE_RE.finditer(body_clean):
                var_name = m.group(1)
                piece_name = piece_map.get(var_name, var_name).lower()
                axis = _resolve_axis(m.group(2))
                goal = float(m.group(3))
                speed_str = m.group(4)
                speed = float(speed_str) if speed_str is not None else None
                all_matches.append((m.start(), 'cmd', (piece_name, axis, False),
                                    goal, speed))

        for m in _SLEEP_FIRE_RE.finditer(body_clean):
            all_matches.append((m.start(), 'sleep', int(m.group(1))))

        # Sort by position in source
        all_matches.sort(key=lambda x: x[0])

        # Build tokens list
        for item in all_matches:
            if item[1] == 'cmd':
                tokens.append(('cmd', item[2], item[3], item[4]))
            elif item[1] == 'sleep':
                tokens.append(('sleep', item[2]))

        if not tokens:
            continue

        # Walk tokens, build keyframes (same strategy as BOS _parse_fire_body_to_tracks)
        current_pose: Dict[tuple, float] = {}
        track_kfs: Dict[tuple, List[BosKeyframe]] = {}
        t_cursor = 0.0
        pending_cmds: List[tuple] = []  # (key, target, speed_or_None)

        def flush_pending():
            nonlocal t_cursor
            if not pending_cmds:
                return
            max_end_t = t_cursor
            for key, target, speed in pending_cmds:
                start_val = current_pose.get(key, 0.0)
                if key not in track_kfs:
                    track_kfs[key] = [BosKeyframe(time=0.0, value=start_val)]
                else:
                    last = track_kfs[key][-1]
                    if last.time < t_cursor - 0.001:
                        track_kfs[key].append(BosKeyframe(time=t_cursor, value=last.value))

                if speed is None:
                    # Instant (no speed) — jump at current time
                    end_t = t_cursor + 0.001
                elif speed > 0:
                    cmd_dur = abs(target - start_val) / speed
                    end_t = t_cursor + cmd_dur
                else:
                    end_t = t_cursor

                track_kfs[key].append(BosKeyframe(time=end_t, value=target))
                current_pose[key] = target
                if end_t > max_end_t:
                    max_end_t = end_t
            pending_cmds.clear()
            t_cursor = max_end_t

        for tok in tokens:
            if tok[0] == 'cmd':
                _, key, target, speed = tok
                pending_cmds.append((key, target, speed))
            elif tok[0] == 'sleep':
                flush_pending()
                t_cursor += tok[1] / 1000.0

        flush_pending()

        if not track_kfs:
            continue

        # Ensure all tracks return to rest (0) at the end
        for key, kfs in track_kfs.items():
            last_val = kfs[-1].value
            if abs(last_val) > 0.001:
                is_rot = key[2]
                default_speed = 90.0 if is_rot else 5.0  # deg/s or units/s
                return_dur = abs(last_val) / default_speed
                # Look for last cmd targeting near-0 to get its speed
                for tok in reversed(tokens):
                    if tok[0] == 'cmd' and tok[1] == key and abs(tok[2]) < 0.001:
                        if tok[3] is not None and tok[3] > 0:
                            return_dur = abs(last_val) / tok[3]
                        break
                t_end = kfs[-1].time + 0.01
                kfs.append(BosKeyframe(time=t_end, value=last_val))
                kfs.append(BosKeyframe(time=t_end + return_dur, value=0.0))

        # Compute total duration
        total_duration = max(kf.time for kfs in track_kfs.values() for kf in kfs)
        if total_duration < 0.01:
            total_duration = 0.5

        # Build tracks
        tracks: List[BosTrack] = []
        for key, kfs in track_kfs.items():
            if len(kfs) < 2:
                continue
            if all(abs(kf.value) < 0.001 for kf in kfs):
                continue
            piece, axis, is_rot = key
            tracks.append(BosTrack(piece=piece, axis=axis, is_rotation=is_rot,
                                   keyframes=kfs))

        if not tracks:
            continue

        # Determine weapon number for clip name
        weapon_num = type_to_min_num.get(weapon_type, len(clips) + 1)
        clip_name = f'Fire_{weapon_num}'

        pieces = sorted({t.piece for t in tracks})
        print(f"  LUA Fire animation '{clip_name}' ({weapon_type}): {len(tracks)} tracks, "
              f"{total_duration:.2f}s, pieces: {', '.join(pieces)}")

        clips.append((clip_name, tracks, None))

    return clips if clips else None


def extract_lua_create_now_rotations(lua_content: str,
                                     include_translations: bool = False) -> Dict[tuple, float]:
    """
    Extract rest-pose transforms from script.Create() instant Turn/Move commands.

    In LUS, 'instant' means Turn/Move called without a speed argument, or with
    a very high speed (9999+). These set the initial pose of pieces.

    Also falls back to StopWalking() for significant rest-pose rotations (>20°),
    mirroring the BOS parse_create_now_rotations behaviour.

    Returns {(piece, axis, is_rotation): value_in_degrees_or_units}.
    """
    piece_map = _parse_piece_names(lua_content)
    convention = _detect_turn_convention(lua_content)
    result: Dict[tuple, float] = {}

    # Regex for Turn/Move with optional speed (to detect instant vs animated)
    _TURN_CREATE_RE = re.compile(
        r'\bTurn\s*\(\s*(\w+)\s*,\s*([xyz]_axis|\d)\s*,\s*([-\d.e]+)(?:\s*,\s*([-\d.e]+))?\s*\)'
    )
    _MOVE_CREATE_RE = re.compile(
        r'\bMove\s*\(\s*(\w+)\s*,\s*([xyz]_axis|\d)\s*,\s*([-\d.e]+)(?:\s*,\s*([-\d.e]+))?\s*\)'
    )
    # Also match lowercase wrappers (instant = no speed arg)
    _TURN_LOWER_CREATE_RE = re.compile(
        r'\bturn\s*\(\s*(\w+)\s*,\s*(\d)\s*,\s*([-\d.]+)\s*\)',
        re.IGNORECASE
    )
    _MOVE_LOWER_CREATE_RE = re.compile(
        r'\bmove\s*\(\s*(\w+)\s*,\s*(\d)\s*,\s*([-\d.]+)\s*\)',
        re.IGNORECASE
    )

    create_body = _extract_lua_function_body(lua_content, 'Create')
    if create_body:
        # Strip comments
        clean = re.sub(r'--[^\n]*', '', create_body)

        # Uppercase Turn() without speed = instant, or with very high speed (9999+)
        for m in _TURN_CREATE_RE.finditer(clean):
            speed_str = m.group(4)
            if speed_str is not None and float(speed_str) < 9000:
                continue  # animated turn, not a rest-pose
            var_name = m.group(1)
            piece_name = piece_map.get(var_name, var_name).lower()
            axis = _resolve_axis(m.group(2))
            value_deg = math.degrees(float(m.group(3)))
            if axis == 2:
                value_deg = -value_deg
            result[(piece_name, axis, True)] = value_deg

        if include_translations:
            for m in _MOVE_CREATE_RE.finditer(clean):
                speed_str = m.group(4)
                if speed_str is not None and float(speed_str) < 9000:
                    continue
                var_name = m.group(1)
                piece_name = piece_map.get(var_name, var_name).lower()
                axis = _resolve_axis(m.group(2))
                result[(piece_name, axis, False)] = float(m.group(3))

        # lowercase turn() without speed = instant
        for m in _TURN_LOWER_CREATE_RE.finditer(clean):
            var_name = m.group(1)
            piece_name = piece_map.get(var_name, var_name).lower()
            axis = _resolve_axis(m.group(2))
            value_deg = float(m.group(3))
            # lowercase wrapper without speed does NOT negate z (only the speed branch does)
            if axis == 2:
                value_deg = -value_deg
            result[(piece_name, axis, True)] = value_deg

        if include_translations:
            for m in _MOVE_LOWER_CREATE_RE.finditer(clean):
                var_name = m.group(1)
                piece_name = piece_map.get(var_name, var_name).lower()
                axis = _resolve_axis(m.group(2))
                result[(piece_name, axis, False)] = float(m.group(3))

    # Fallback: StopWalking significant rotations (>20°)
    rot_result = {k: v for k, v in result.items() if k[2]}
    if not rot_result:
        body = _extract_lua_function_body(lua_content, 'StopWalking')
        if body:
            clean = re.sub(r'--[^\n]*', '', body)
            candidate: Dict[tuple, float] = {}
            if convention == 'lower':
                for m in _TURN_LOWER_RE.finditer(clean):
                    var_name = m.group(1)
                    piece_name = piece_map.get(var_name, var_name).lower()
                    axis = _resolve_axis(m.group(2))
                    val = float(m.group(3))
                    if abs(val) > 20.0:
                        candidate[(piece_name, axis, True)] = val
            else:
                for m in _TURN_UPPER_RE.finditer(clean):
                    var_name = m.group(1)
                    piece_name = piece_map.get(var_name, var_name).lower()
                    axis = _resolve_axis(m.group(2))
                    val = math.degrees(float(m.group(3)))
                    if axis == 2:
                        val = -val
                    if abs(val) > 20.0:
                        candidate[(piece_name, axis, True)] = val
            if candidate:
                result.update(candidate)
                print(f"  LUA StopWalking rest pose: {len(candidate)} rotations")

    if result:
        print(f"  LUA Create now-rotations: {len(result)} transforms")
    return result


def extract_lua_weapon_queries(lua_content: str) -> Dict[str, Dict]:
    """
    Extract weapon → piece mappings from script.QueryWeapon and script.AimFromWeapon.

    Returns dict mapping weapon_type → {query_pieces: [...], aim_from_pieces: [...]}.
    """
    piece_map = _parse_piece_names(lua_content)
    result = {}

    # Parse QueryWeapon
    qm = _QUERY_WEAPON_RE.search(lua_content)
    if qm:
        body = qm.group(1)
        # Look for patterns like: if weapons[weapon] == "laser" then return laserflare
        for m in re.finditer(r'weapons\[weapon\]\s*==\s*["\'](\w+)["\']\s*then\s*\n\s*return\s+(\w+)',
                             body):
            wtype = m.group(1)
            var_name = m.group(2)
            piece_name = piece_map.get(var_name, var_name).lower()
            if wtype not in result:
                result[wtype] = {'query_pieces': [], 'aim_from_pieces': []}
            result[wtype]['query_pieces'].append(piece_name)

    # Parse AimFromWeapon
    am = _AIM_FROM_RE.search(lua_content)
    if am:
        body = am.group(1)
        for m in re.finditer(r'weapons\[weapon\]\s*==\s*["\'](\w+)["\']\s*then\s*\n\s*return\s+(\w+)',
                             body):
            wtype = m.group(1)
            var_name = m.group(2)
            piece_name = piece_map.get(var_name, var_name).lower()
            if wtype not in result:
                result[wtype] = {'query_pieces': [], 'aim_from_pieces': []}
            result[wtype]['aim_from_pieces'].append(piece_name)

    return result


def is_lua_script(content: str) -> bool:
    """Check if the content is a LUA/LUS script (vs BOS)."""
    # LUS scripts use piece() function and Lua-style function declarations
    return bool(re.search(r'\bpiece\s*\(', content)) and \
           bool(re.search(r'\bfunction\s+', content))
