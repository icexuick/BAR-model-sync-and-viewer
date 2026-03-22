"""
BOS → glTF Animation Extractor

Extracts keyframe animation data from BAR/Spring BOS scripts.
Handles animations exported by Skeletor_S3O (the tool used for most BAR units).

BOS Walk() structure (Skeletor_S3O output):
  if (isMoving) { //Frame:0
      turn lleg to x-axis <-71.52> speed <3111> / animSpeed;
      move pelvis to y-axis [-1.52] speed [73.29] / animSpeed;
      sleep ((15*animSpeed) -1);
  }
  while(isMoving) {
      if (isMoving) { //Frame:5
          ...
          sleep ((33*animSpeed) -1);
      }
  }

Strategy:
- //Frame:N comments give keyframe indices (from original Blender export at 30fps)
- Timing = frame_number / 30.0 seconds
- <angle> values are degrees (rotation), [value] values are engine units (translation)
- All turn/move in one frame block fire simultaneously; sleep marks end of that frame
- Frame:0 and Frame:30 have identical values → seamless loop
"""

import re
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}
FPS = 30.0

# Regex patterns for BOS commands
# Turn:  turn piece to x-axis <value> speed ...
#   also: turn piece to x-axis ((<value> *animAmplitude)/100) speed ...  (Skeletor animAmplitude format)
# Move:  move piece to y-axis [value] speed ...
#   also: move piece to y-axis ((([value] *MOVESCALE)/100) ...) speed ...
_TURN_RE = re.compile(
    r'\bturn\s+(\w+)\s+to\s+([xyz])-axis\s+(?:\(\s*\(\s*)?<([-\d.]+)>',
    re.IGNORECASE
)
_MOVE_RE = re.compile(
    r'\bmove\s+(\w+)\s+to\s+([xyz])-axis\s+(?:\(\s*\(\s*\(\s*)?\[([-\d.]+)\]',
    re.IGNORECASE
)
_FRAME_RE = re.compile(r'//\s*Frame\s*:?\s*(\d+)', re.IGNORECASE)
# Matches:  turn piece to y-axis <value> now;
_TURN_NOW_RE = re.compile(
    r'\bturn\s+(\w+)\s+to\s+([xyz])-axis\s+<([-\d.]+)>\s+now',
    re.IGNORECASE
)
# Matches:  move piece to z-axis [value] now;
_MOVE_NOW_RE = re.compile(
    r'\bmove\s+(\w+)\s+to\s+([xyz])-axis\s+\[([-\d.]+)\]\s+now',
    re.IGNORECASE
)


@dataclass
class BosKeyframe:
    time: float   # seconds (frame_number / FPS)
    value: float  # degrees (rotation) or engine units (translation)


@dataclass
class BosTrack:
    """Keyframe animation for one piece on one axis."""
    piece: str
    axis: int          # 0=x, 1=y, 2=z
    is_rotation: bool  # True=degrees, False=engine units
    keyframes: List[BosKeyframe] = field(default_factory=list)


def _expand_macros(text: str) -> str:
    """
    Expand simple #define macros (no-arg and arg-less) in BOS scripts.
    Also resolves 'sleep VARNAME' by substituting the last numeric assignment
    to that variable (e.g. WALK_PERIOD=98 → sleep 98).
    Only handles simple cases; recursive/complex macros are left as-is.
    """
    # 1. Collect #define NAME value  (single-line and multi-line continuation macros)
    # Uses line-by-line scanning to handle NAME\ style multi-line defines.
    bs = chr(92)  # backslash — avoids raw-string confusion
    defines: Dict[str, str] = {}
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        m = re.match(r'#define\s+(\w+)(.*)', lines[i])
        if m:
            name = m.group(1)
            rest = m.group(2)
            val_lines: list = []
            if rest.endswith(bs):
                val_lines.append(rest[:-1])
                i += 1
                while i < len(lines) and lines[i].endswith(bs):
                    val_lines.append(lines[i][:-1])
                    i += 1
                if i < len(lines):
                    val_lines.append(lines[i])
            else:
                val_lines.append(rest)
            defines[name] = '\n'.join(val_lines).strip()
        i += 1

    # 2. Remove #define lines themselves
    text = re.sub(r'#define[^\n]*(\\\n[^\n]*)*\n?', '', text)

    # 3. Expand macros — iterate until stable (handles macros that reference other macros)
    for _ in range(8):
        expanded = False
        for name, val in defines.items():
            # Only replace when the macro name appears as a standalone identifier
            new = re.sub(r'\b' + re.escape(name) + r'\b', val, text)
            if new != text:
                text = new
                expanded = True
        if not expanded:
            break

    # 4. Resolve 'sleep VARNAME' — find the minimum numeric assignment to that variable.
    # The walk-speed value is always the smallest (e.g. WALK_PERIOD=98 during walking,
    # vs WALK_PERIOD=400 at rest), so min gives the most accurate animation timing.
    def _resolve_sleep_var(m):
        var = m.group(1)
        assignments = re.findall(r'\b' + re.escape(var) + r'\s*=\s*(\d+)', text)
        if assignments:
            return f'sleep {min(int(v) for v in assignments)}'
        return m.group(0)  # leave unchanged

    text = re.sub(r'\bsleep\s+([A-Za-z_]\w*)', _resolve_sleep_var, text)

    return text


def _strip_comments(text: str) -> str:
    """Remove BOS // and /* */ comments."""
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r'//[^\n]*', '', text)
    return text


def _inline_call_scripts(body: str, bos_content: str, depth: int = 0) -> str:
    """Replace 'call-script FuncName()' with the body of that function (max 3 levels deep)."""
    if depth > 3:
        return body
    _CALL_RE = re.compile(r'\bcall-script\s+(\w+)\s*\([^)]*\)\s*;', re.IGNORECASE)
    def replacer(m):
        sub_body = _extract_function_body(bos_content, m.group(1))
        if sub_body:
            return _inline_call_scripts(sub_body, bos_content, depth + 1)
        return ''
    return _CALL_RE.sub(replacer, body)


def _extract_function_body(content: str, func_name: str) -> Optional[str]:
    """
    Extract the full body (between outer braces) of a named BOS function.
    Returns body text with comments preserved (needed for Frame:N markers).
    """
    # Search directly in original content for the function opening brace.
    # We must search in original (not comment-stripped) to get correct positions.
    # Use comment-stripped only to verify the match is not inside a comment.
    # Allow // line comments between ) and { (e.g. "Activate() // comment\n{")
    pattern = re.compile(
        rf'\b{re.escape(func_name)}\s*\([^)]*\)\s*(?://[^\n]*)?\s*\{{',
        re.IGNORECASE
    )
    match = pattern.search(content)
    if not match:
        return None

    # Walk the ORIGINAL content (with comments) to find matching close brace
    start = match.end() - 1  # position of opening {
    depth = 0
    pos = start
    while pos < len(content):
        if content[pos] == '{':
            depth += 1
        elif content[pos] == '}':
            depth -= 1
            if depth == 0:
                return content[start + 1:pos]
        pos += 1
    return None


def _extract_while_body(body: str) -> Tuple[str, str]:
    """
    Split a Walk() body into (pre_while_text, while_loop_body).
    If no while loop found, returns (body, '').
    """
    clean = _strip_comments(body)
    pattern = re.compile(r'\bwhile\s*\([^)]+\)\s*\{', re.IGNORECASE)
    match = pattern.search(clean)
    if not match:
        return body, ''

    while_start_in_clean = match.start()
    brace_start = match.end() - 1

    depth = 0
    pos = brace_start
    while pos < len(clean):
        if clean[pos] == '{':
            depth += 1
        elif clean[pos] == '}':
            depth -= 1
            if depth == 0:
                # Return pre-while part (use original for comment preservation) and loop body
                pre = body[:while_start_in_clean]
                loop_body = body[brace_start + 1:pos]
                return pre, loop_body
        pos += 1

    return body, ''


def _parse_frame_blocks(text: str) -> List[Tuple[int, Dict[Tuple, float]]]:
    """
    Split text on //Frame:N markers and parse turn/move commands per block.
    Returns list of (frame_number, {(piece, axis, is_rotation): value}).
    """
    frame_positions = [(m.start(), int(m.group(1))) for m in _FRAME_RE.finditer(text)]
    if not frame_positions:
        return []

    blocks = []
    for i, (pos, frame_num) in enumerate(frame_positions):
        end_pos = frame_positions[i + 1][0] if i + 1 < len(frame_positions) else len(text)
        block_text = text[pos:end_pos]

        commands: Dict[Tuple, float] = {}
        for m in _TURN_RE.finditer(block_text):
            key = (m.group(1).lower(), AXIS_INDEX[m.group(2).lower()], True)
            commands[key] = float(m.group(3))
        for m in _MOVE_RE.finditer(block_text):
            key = (m.group(1).lower(), AXIS_INDEX[m.group(2).lower()], False)
            commands[key] = float(m.group(3))

        if commands:
            blocks.append((frame_num, commands))

    return blocks


def parse_create_hide_pieces(bos_content: str) -> set:
    """
    Return the set of piece names that are hidden via 'hide <piece>' in Create().
    These pieces are invisible at game start and should not be rendered in the viewer.
    """
    body = _extract_function_body(bos_content, 'Create')
    if not body:
        return set()
    clean = _strip_comments(body)
    return {m.group(1).lower() for m in re.finditer(r'\bhide\s+(\w+)', clean, re.IGNORECASE)}


def parse_create_now_rotations(bos_content: str) -> Dict[Tuple, float]:
    """
    Parse rest-pose transforms for pieces. Tries two sources in order:
    1. 'turn/move piece to axis <value> now' in Create() — explicit immediate pose
    2. Turn commands in StopWalking() — the idle/rest pose the unit returns to
    Returns {(piece, axis, is_rotation): value}
      is_rotation=True  → degrees (turn)
      is_rotation=False → engine units (move)
    """
    result: Dict[Tuple, float] = {}

    # Source 1: Create() 'now' commands — rotations only.
    # We intentionally skip 'move ... now' translations: those often stow/retract
    # parts to a "hidden" start state (e.g. barrels pushed in), while the S3O
    # geometry already represents the correct displayed rest pose.
    body = _extract_function_body(bos_content, 'Create')
    if body:
        for m in _TURN_NOW_RE.finditer(body):
            key = (m.group(1).lower(), AXIS_INDEX[m.group(2).lower()], True)
            result[key] = float(m.group(3))

    # Source 2: activatescr() — fly pose for aircraft.
    # Aircraft use activatescr() (state=0 = flying) to fold wings/tilt engines to
    # the in-flight position. The S3O rest pose is the landed/stored state.
    # We detect this by: activatescr exists AND has significant turn commands (>15°).
    # Last value per (piece, axis) wins (the final target angle).
    # Applied unconditionally (not blocked by Create() now_rots) so that e.g. armthund
    # can have both thrust hide-turns from Create() AND wing fly-pose from activatescr.
    for _fly_func in ('activatescr', 'Activate'):
        body = _extract_function_body(bos_content, _fly_func)
        if not body:
            continue
        stripped = _strip_comments(body)
        candidate = {}
        for m in _TURN_RE.finditer(stripped):
            val = float(m.group(3))
            if abs(val) > 15.0:
                key = (m.group(1).lower(), AXIS_INDEX[m.group(2).lower()], True)
                candidate[key] = val  # last value wins = final target angle
        for m in _MOVE_RE.finditer(stripped):
            val = float(m.group(3))
            if abs(val) > 0.0:
                key = (m.group(1).lower(), AXIS_INDEX[m.group(2).lower()], False)
                candidate[key] = val
        if candidate:
            result.update(candidate)
            print(f"  Using {_fly_func}() as fly pose ({len(candidate)} transforms)")
            break

    # Source 3: StopWalking() — fallback if Create() has no 'now' rotations.
    # Only use if StopWalking contains significant non-zero rotations (>5°),
    # indicating a non-trivial rest pose that differs from the S3O zero pose.
    rot_result = {k: v for k, v in result.items() if k[2]}  # rotations only
    if not rot_result:
        for func_name in ['StopWalking', 'StopWalk']:
            body = _extract_function_body(bos_content, func_name)
            if not body:
                continue
            stripped = _strip_comments(body)
            candidate = {}
            for m in _TURN_RE.finditer(stripped):
                val = float(m.group(3))
                if abs(val) > 20.0:  # only significant rest-pose rotations
                    key = (m.group(1).lower(), AXIS_INDEX[m.group(2).lower()], True)
                    candidate[key] = val
            if candidate:
                result.update(candidate)
                print(f"  Using {func_name}() as rest pose ({len(candidate)} rotations)")
                break

    return result


def extract_stopwalking_pose(bos_content: str) -> Optional[List[BosTrack]]:
    """
    Extract the rest/idle pose from StopWalking() (or StopWalk()) as a list of
    single-keyframe BosTrack objects at t=0.

    Returns None if no StopWalking function is found or it has no turn/move commands.
    """
    for func_name in ['StopWalking', 'StopWalk', 'StopMoving']:
        body = _extract_function_body(bos_content, func_name)
        if not body:
            continue
        # If StopMoving just calls StopWalking, skip it
        stripped = _strip_comments(body)
        if re.search(r'\bcall-script\s+StopWalk', stripped, re.IGNORECASE) and \
                not re.search(r'\bturn\b', stripped, re.IGNORECASE):
            continue
        commands: Dict[Tuple, float] = {}
        for m in _TURN_RE.finditer(stripped):
            key = (m.group(1).lower(), AXIS_INDEX[m.group(2).lower()], True)
            commands[key] = float(m.group(3))
        for m in _MOVE_RE.finditer(stripped):
            key = (m.group(1).lower(), AXIS_INDEX[m.group(2).lower()], False)
            commands[key] = float(m.group(3))
        if not commands:
            continue
        tracks: List[BosTrack] = []
        # Group by (piece, axis, is_rotation) — last value wins (matches BOS execution order)
        for (piece, axis, is_rot), value in commands.items():
            tracks.append(BosTrack(
                piece=piece, axis=axis, is_rotation=is_rot,
                keyframes=[BosKeyframe(time=0.0, value=value),
                           BosKeyframe(time=0.033, value=value)]  # two frames for valid clip
            ))
        print(f"  StopWalking pose: {len(tracks)} tracks from {func_name}()")
        return tracks
    return None


def extract_walk_animation(bos_content: str) -> Optional[Tuple[str, List[BosTrack]]]:
    """
    Extract walk animation tracks from a BOS script.

    Only uses while-loop frames (not the pre-loop initialisation frame).
    The pre-loop frame (e.g. Frame 0) is played once when the unit starts
    moving; the actual repeating cycle is the while-loop body only.

    Times are remapped so the first while-loop frame = t=0.
    A closing keyframe equal to t=0 is added for every track at t=duration,
    guaranteeing a seamless Three.js LoopRepeat.

    Returns (animation_name, List[BosTrack], now_rots) or None if nothing found.
    now_rots: {(piece, axis, True): degrees} from Create() 'turn ... now' commands.
    These must be applied as initial node rotations in the GLB so the rest pose
    matches what the Spring engine sets up via Create().
    """
    bos_content = _expand_macros(bos_content)
    now_rots = parse_create_now_rotations(bos_content)

    for func_name in ['Walk', 'StartMoving', 'Move', 'DoTheWalking', 'movelegs', 'walkscr', 'Movement']:
        body = _extract_function_body(bos_content, func_name)
        if not body:
            continue

        pre_body, while_body = _extract_while_body(body)
        if not while_body:
            # No while loop — fall back to treating the whole body as the cycle
            while_body = body
            pre_body = ''

        # Parse pre-loop frames (e.g. Frame 0) — these are the "start of cycle"
        # poses played once before the while-loop. We include them as the first
        # keyframe (t=0) so the full cycle is: Frame0 → loop frames → Frame0 again.
        pre_blocks = _parse_frame_blocks(pre_body)
        pre_cmds: Dict[Tuple, float] = {}
        for _, cmds in pre_blocks:
            pre_cmds.update(cmds)

        loop_blocks = _parse_frame_blocks(while_body)
        if not loop_blocks:
            # No //Frame: markers — try sleep-based parsing (cornecro-style)
            # Inline call-script sub-functions one level deep, but only when each
            # sub-function is called exactly once (avoid exploding repeated calls
            # like call-script movelegs() × 13 in walkscr).
            def _inline_call_scripts(text: str) -> str:
                calls = re.findall(
                    r'\b(?:call-script|start-script)\s+(\w+)\s*\([^)]*\)',
                    text, flags=re.IGNORECASE)
                # Only inline sub-functions that appear exactly once
                once = {name for name in calls if calls.count(name) == 1}
                def _replacer(m):
                    name = m.group(1)
                    if name not in once:
                        return m.group(0)
                    sub = _extract_function_body(bos_content, name)
                    return sub if sub else m.group(0)
                return re.sub(
                    r'\b(?:call-script|start-script)\s+(\w+)\s*\([^)]*\)\s*;?',
                    _replacer, text, flags=re.IGNORECASE)
            while_body = _inline_call_scripts(while_body)
            clean_while = _strip_comments(while_body)
            # Detect animSpeed-based sleeps: ((N*animSpeed) -K)
            # Use the pre-body fixed sleep (e.g. 'sleep 131') as the per-keyframe
            # duration since the animSpeed formula scales with unit speed at runtime.
            _ANIMSPEED_SLEEP_RE = re.compile(
                r'\bsleep\s+\(\s*\(\s*\d+\s*\*\s*animSpeed', re.IGNORECASE)
            uses_animspeed = bool(_ANIMSPEED_SLEEP_RE.search(clean_while))
            pre_sleep_ms: Optional[float] = None
            if uses_animspeed:
                # Use the fixed-value sleep from anywhere in the Walk() body as the
                # per-keyframe timing (e.g. 'sleep 131' = the pre-loop frame duration).
                clean_body = _strip_comments(body)
                simple_sleep = re.search(r'\bsleep\s+(\d+)', clean_body, re.IGNORECASE)
                if simple_sleep:
                    pre_sleep_ms = float(simple_sleep.group(1))

            segments = re.split(_SLEEP_RE, clean_while)
            if len(segments) >= 3:
                time_ms = 0.0
                sleep_blocks: List[Tuple[int, Dict[Tuple, float]]] = []
                i = 0
                kf_index = 0
                while i < len(segments):
                    block_text = segments[i]
                    cmds: Dict[Tuple, float] = {}
                    for m in _TURN_RE.finditer(block_text):
                        key = (m.group(1).lower(), AXIS_INDEX[m.group(2).lower()], True)
                        cmds[key] = float(m.group(3))
                    for m in _TURN_NOW_RE.finditer(block_text):
                        key = (m.group(1).lower(), AXIS_INDEX[m.group(2).lower()], True)
                        cmds[key] = float(m.group(3))
                    for m in _MOVE_RE.finditer(block_text):
                        key = (m.group(1).lower(), AXIS_INDEX[m.group(2).lower()], False)
                        cmds[key] = float(m.group(3))
                    for m in _MOVE_NOW_RE.finditer(block_text):
                        key = (m.group(1).lower(), AXIS_INDEX[m.group(2).lower()], False)
                        cmds[key] = float(m.group(3))
                    if cmds:
                        if pre_sleep_ms is not None:
                            sleep_blocks.append((int(kf_index * pre_sleep_ms), cmds))
                        else:
                            sleep_blocks.append((int(time_ms), cmds))
                        kf_index += 1
                    if i + 1 < len(segments):
                        try:
                            time_ms += float(segments[i + 1])
                        except (ValueError, IndexError):
                            pass
                    i += 2
                if len(sleep_blocks) >= 2:
                    # sleep_blocks already has correct ms timestamps — build tracks
                    # directly in seconds without going through the frame-number path,
                    # which loses precision for short sleeps (e.g. 30ms at 30fps = 0.9 frame).
                    total_ms = sleep_blocks[-1][0] + (
                        float(segments[len(sleep_blocks) * 2 - 1])
                        if len(segments) > len(sleep_blocks) * 2 - 1 else 0)
                    duration = total_ms / 1000.0
                    n_blocks = len(sleep_blocks)

                    track_dict_ms: Dict[Tuple, List[BosKeyframe]] = {}
                    for t_ms, cmds in sleep_blocks:
                        t = t_ms / 1000.0
                        for key, value in cmds.items():
                            track_dict_ms.setdefault(key, []).append(BosKeyframe(time=t, value=value))

                    # Closing keyframe — loop back to first block's values
                    first_cmds = sleep_blocks[0][1]
                    for key in track_dict_ms:
                        closing_value = first_cmds.get(key, track_dict_ms[key][0].value)
                        track_dict_ms[key].append(BosKeyframe(time=duration, value=closing_value))

                    tracks_ms = [
                        BosTrack(piece=piece, axis=axis, is_rotation=is_rot, keyframes=kfs)
                        for (piece, axis, is_rot), kfs in track_dict_ms.items()
                        if len(kfs) >= 2
                    ]
                    if tracks_ms:
                        print(f"  Animation '{func_name}': {len(tracks_ms)} tracks, "
                              f"{n_blocks} keyframes (sleep-based), duration {duration:.2f}s")
                        return func_name, tracks_ms, now_rots

            if not loop_blocks:
                continue

        # Determine step_size from most common diff between consecutive frame numbers.
        # Frame numbers may wrap (e.g. 10,15,20,25,30,5), so use modulo.
        frame_nums = [fn for fn, _ in loop_blocks]
        diffs = [(frame_nums[i + 1] - frame_nums[i]) % 300
                 for i in range(len(frame_nums) - 1)]
        if diffs:
            from collections import Counter
            step_size = Counter(diffs).most_common(1)[0][0]
        else:
            step_size = 5  # fallback

        # If we have a pre-loop frame (e.g. Frame 0), use it as t=0 and as the
        # closing keyframe at t=duration. Drop any loop frame that is a duplicate
        # of the pre-loop frame (e.g. Frame 30 == Frame 0) to avoid a stilstand.
        if pre_cmds:
            # Detect if the last loop frame duplicates the pre-loop frame
            last_cmds = loop_blocks[-1][1]
            shared_keys = set(pre_cmds) & set(last_cmds)
            n_matching = sum(1 for k in shared_keys if abs(pre_cmds[k] - last_cmds[k]) < 0.5)
            if shared_keys and n_matching >= len(shared_keys) * 0.85 \
                    and len(shared_keys) >= len(pre_cmds) * 0.6:
                # Last loop frame is a duplicate of Frame 0 — drop it
                active_loop = loop_blocks[:-1]
                print(f"  Dropping duplicate closing frame (Frame {loop_blocks[-1][0]} == Frame 0)")
            else:
                active_loop = loop_blocks

            n_loop   = len(active_loop)
            duration = (n_loop + 1) * step_size / FPS

            track_dict: Dict[Tuple, List[BosKeyframe]] = {}

            # t=0 → pre-loop (Frame 0)
            for key, value in pre_cmds.items():
                track_dict.setdefault(key, []).append(BosKeyframe(time=0.0, value=value))

            # t=step, 2*step, ... → loop frames
            for loop_idx, (frame_num, commands) in enumerate(active_loop):
                t = (loop_idx + 1) * step_size / FPS
                for key, value in commands.items():
                    track_dict.setdefault(key, []).append(BosKeyframe(time=t, value=value))

            # For tracks that appear in the loop but NOT in the pre-loop frame,
            # add t=0 with the first loop-frame value so there is no jump.
            # (e.g. corcat's rthigh is absent from Frame:8 but present in Frame:12)
            first_loop_cmds = active_loop[0][1] if active_loop else {}
            for key, value in first_loop_cmds.items():
                if key not in pre_cmds and key in track_dict:
                    track_dict[key].insert(0, BosKeyframe(time=0.0, value=value))

            n_blocks = n_loop + 1  # for the print
        else:
            # No pre-loop frame — use loop frames directly, t=0 is first loop frame
            n_blocks = len(loop_blocks)
            duration = n_blocks * step_size / FPS

            track_dict = {}
            for loop_idx, (frame_num, commands) in enumerate(loop_blocks):
                t = loop_idx * step_size / FPS
                for key, value in commands.items():
                    track_dict.setdefault(key, []).append(BosKeyframe(time=t, value=value))

        if not track_dict:
            continue

        # Build BosTrack objects, sort by time, deduplicate, add closing kf
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

            if len(deduped) < 1:
                continue

            # Closing keyframe at t=duration = pre-loop value (if available)
            # or first loop frame value. This makes the loop seamless.
            key = (piece, axis, is_rot)
            if abs(deduped[-1].time - duration) < 1e-5:
                pass  # already at t=duration
            else:
                closing_value = pre_cmds.get(key, deduped[0].value)
                deduped.append(BosKeyframe(time=duration, value=closing_value))

            if len(deduped) >= 2:
                tracks.append(BosTrack(piece=piece, axis=axis,
                                       is_rotation=is_rot, keyframes=deduped))

        if tracks:
            print(f"  Animation '{func_name}': {len(tracks)} tracks, "
                  f"{n_blocks} keyframes (step={step_size}), duration {duration:.2f}s")
            return func_name, tracks, now_rots

    return None


# Matches: spin <piece> around <axis>-axis speed <value>
# speed value is either angle-bracketed <number> or a bare variable expression
# (e.g. WindSpeed, WindSpeed / -5.0).  Both forms are captured.
_SPIN_RE = re.compile(
    r'\bspin\s+(\w+)\s+around\s+([xyz])-axis\s+speed\s+'
    r'(?:<([^>]+)>|(\(?-?\d*\.?\d*\)?\s*\*?\s*[A-Za-z_]\w*(?:\s*[*/]\s*-?\d+\.?\d*)?\)?))',
    re.IGNORECASE
)
# Default wind speed (deg/s) used when the BOS speed is a variable like 'WindSpeed'.
# BAR wind turbines have WindSpeed in [0..2000] Spring units at medium wind; 400 is a
# reasonable average that gives a visually believable rotation in the viewer.
_DEFAULT_WIND_SPEED = 264.0
# Default metal extractor spin speed (deg/s) used when speed is a variable like
# 'Static_Var_1' (set at runtime by SetSpeed with the metal extraction rate).
_DEFAULT_MEX_SPEED = 150.0


def _collect_spin_commands(bos_content: str, func_name: str,
                           visited: Optional[set] = None) -> Dict[Tuple[str, int], float]:
    """
    Collect all non-zero spin commands reachable from func_name, following
    start-script / call-script into sub-functions (one level deep).
    Returns {(piece_lower, axis_int): speed_deg_per_s}.
    """
    if visited is None:
        visited = set()
    if func_name.lower() in visited:
        return {}
    visited.add(func_name.lower())

    body = _extract_function_body(bos_content, func_name)
    if not body:
        return {}

    clean = _strip_comments(body)
    clean = re.sub(r'\bstop-spin\b[^\n]*', '', clean, flags=re.IGNORECASE)

    spins: Dict[Tuple[str, int], float] = {}
    for m in _SPIN_RE.finditer(clean):
        piece = m.group(1).lower()
        axis = AXIS_INDEX[m.group(2).lower()]
        # group(3) = bracketed value <...>, group(4) = bare/parenthesized variable expression
        raw = (m.group(3) or m.group(4) or '').strip()
        try:
            speed = float(raw)
        except ValueError:
            # Variable expression — determine sign from any numeric multiplier present
            # (e.g. '-1*var', '(-0.5)*var', 'var*2', 'var / -5').
            # Extract leading numeric coefficient if present.
            coeff_m = re.search(r'\(?\s*(-?\d+\.?\d*)\s*\)?\s*\*', raw)
            if coeff_m:
                try:
                    sign = -1.0 if float(coeff_m.group(1)) < 0 else 1.0
                except ValueError:
                    sign = 1.0
            elif re.search(r'/\s*-', raw):
                sign = -1.0
            else:
                sign = 1.0
            # Wind-speed variables keep the wind default; all others get the mex default.
            if re.search(r'wind', raw, re.IGNORECASE):
                speed = sign * _DEFAULT_WIND_SPEED
            else:
                speed = sign * _DEFAULT_MEX_SPEED
        if speed != 0.0:
            spins[(piece, axis)] = speed

    # Follow start-script / call-script into sub-functions
    for m in re.finditer(r'\b(?:start-script|call-script)\s+(\w+)\s*\(', clean, re.IGNORECASE):
        sub = m.group(1)
        sub_spins = _collect_spin_commands(bos_content, sub, visited)
        # Only add if not already overridden by a direct spin in this function
        for k, v in sub_spins.items():
            if k not in spins:
                spins[k] = v

    return spins


def extract_spin_animation(bos_content: str) -> Optional[List[Tuple[str, List[BosTrack]]]]:
    """
    Extract continuous spin animations from Activate()/Go() BOS functions.

    Follows start-script/call-script into sub-functions (e.g. DishSpin())
    so factory units that delegate dish spinning to a helper are handled.

    Returns a list of (clip_name, [track]) — one clip per spinning piece —
    so each piece loops independently at its own speed. A single shared clip
    would loop at the slowest piece's period, causing faster pieces to pause.

    Each clip has 8 keyframes at 0°..315° (45° steps), no closing 360° frame.
    Three.js LoopRepeat jumps from last keyframe back to t=0 seamlessly.
    """
    # Piece name fragments that indicate non-visual / non-interesting spinners.
    # These are excluded so factory cagelights, pads, wheels etc. don't pollute
    # the spin_pieces list or trigger misleading radar/role tooltips.
    _EXCLUDE_FRAGMENTS = (
        'cagelight', 'light', 'emit', 'blink', 'pad', 'prop',
        'screw', 'belt', 'nano', 'flare', 'fire', 'glow', 'spark',
    )

    def _is_interesting(piece: str) -> bool:
        p = piece.lower()
        return not any(frag in p for frag in _EXCLUDE_FRAGMENTS)

    for func_name in ['Activate', 'Go', 'StartActivate', 'Create', 'StartBuilding', 'MoveRate3', 'MMStatus']:
        spins = _collect_spin_commands(bos_content, func_name)

        # Filter to interesting (visual) spinners only
        spins = {k: v for k, v in spins.items() if _is_interesting(k[0])}

        if not spins:
            continue

        clips: List[Tuple[str, List[BosTrack]]] = []
        for (piece, axis), speed in spins.items():
            duration = abs(360.0 / speed)
            sign = 1.0 if speed > 0 else -1.0
            n = 8
            kfs = [
                BosKeyframe(time=duration * i / n, value=sign * 360.0 * i / n)
                for i in range(n)
            ]
            track = BosTrack(piece=piece, axis=axis, is_rotation=True, keyframes=kfs)
            clip_name = f"{func_name}_{piece}"
            clips.append((clip_name, [track]))

        if clips:
            pieces = [c[0].split('_', 1)[1] for c in clips]
            print(f"  Spin animation '{func_name}': {len(clips)} spinning pieces: {', '.join(pieces)}")
            return clips

    return None


_SLEEP_RE = re.compile(
    r'\bsleep\s+(?:\(\s*\(\s*)?(\d+)(?:\s*\*\s*animSpeed\s*\)\s*[+-]\s*\d+\s*\))?',
    re.IGNORECASE
)


def extract_activate_loop_animation(bos_content: str) -> Optional[List[Tuple[str, List[BosTrack]]]]:
    """
    Extract keyframe animations from functions that run a while(TRUE) loop
    with 'turn piece to axis <deg> speed <spd>' + 'sleep N' steps.

    This handles units like armaser whose jammer spindle/arms animate via a
    dedicated spinarms() helper started from Create() — not a spin command.

    Returns list of (clip_name, [BosTrack]) or None.
    """
    # Find candidate function names: started from Create() or Activate()
    candidates = []
    for src_func in ('Create', 'Activate', 'Go', 'StartActivate'):
        body = _extract_function_body(bos_content, src_func)
        if not body:
            continue
        for m in re.finditer(r'\bstart-script\s+(\w+)\s*\(', body, re.IGNORECASE):
            name = m.group(1)
            if name.lower() not in ('walk', 'stopwalking', 'unitspeed', 'damagedsmoke',
                                    'smokeunit', 'randomsmoke', 'offhit', 'offonhit'):
                candidates.append(name)

    for func_name in candidates:
        body = _extract_function_body(bos_content, func_name)
        if not body:
            continue
        clean = _strip_comments(body)

        # Must contain a while(TRUE) loop
        if not re.search(r'\bwhile\s*\(\s*TRUE\s*\)', clean, re.IGNORECASE):
            continue

        # Extract the while body
        _, while_body = _extract_while_body(clean)
        if not while_body:
            while_body = clean

        # Parse turn-to + sleep segments
        segments = re.split(_SLEEP_RE, while_body)
        if len(segments) < 3:
            continue

        # Build keyframe blocks: for each sleep boundary, collect turn commands
        time_ms = 0.0
        kf_blocks: List[Tuple[float, Dict[Tuple, float]]] = []

        i = 0
        while i < len(segments):
            block_text = segments[i]
            cmds: Dict[Tuple, float] = {}
            for m in _TURN_RE.finditer(block_text):
                key = (m.group(1).lower(), AXIS_INDEX[m.group(2).lower()], True)
                cmds[key] = float(m.group(3))
            for m in _MOVE_RE.finditer(block_text):
                key = (m.group(1).lower(), AXIS_INDEX[m.group(2).lower()], False)
                cmds[key] = float(m.group(3))
            if cmds:
                kf_blocks.append((time_ms / 1000.0, cmds))
            if i + 1 < len(segments):
                try:
                    time_ms += float(segments[i + 1])
                except (ValueError, IndexError):
                    pass
            i += 2

        if len(kf_blocks) < 2:
            continue

        duration = time_ms / 1000.0

        # Build track dict
        track_dict: Dict[Tuple, List[BosKeyframe]] = {}
        for t, cmds in kf_blocks:
            for key, val in cmds.items():
                track_dict.setdefault(key, []).append(BosKeyframe(time=t, value=val))

        # Add closing keyframe = first keyframe value for seamless loop
        for key, kfs in track_dict.items():
            kfs.sort(key=lambda k: k.time)
            if abs(kfs[-1].time - duration) > 1e-3:
                kfs.append(BosKeyframe(time=duration, value=kfs[0].value))

        tracks = [
            BosTrack(piece=key[0], axis=key[1], is_rotation=key[2], keyframes=kfs)
            for key, kfs in track_dict.items()
            if len(kfs) >= 2
        ]

        if tracks:
            pieces = sorted({t.piece for t in tracks})
            print(f"  Activate-loop animation '{func_name}': {len(tracks)} tracks, "
                  f"duration {duration:.2f}s, pieces: {', '.join(pieces)}")
            return [(func_name, tracks)]

    return None


def _parse_turn_move_to_tracks(body: str, start_pose: Optional[Dict[Tuple, float]] = None
                                ) -> Tuple[List[BosTrack], float]:
    """
    Parse 'turn piece to axis <deg> speed <spd>' and 'move piece to axis [val] speed [spd]'
    commands from a BOS function body.  Duration = max(|target - start| / speed) over all pieces.

    Returns (tracks, duration_seconds).  Each track has two keyframes: t=0 (start pose or 0)
    and t=duration (target value).
    """
    clean = _strip_comments(body)
    targets: Dict[Tuple, float] = {}
    speeds: Dict[Tuple, float] = {}

    for m in _TURN_RE.finditer(clean):
        key = (m.group(1).lower(), AXIS_INDEX[m.group(2).lower()], True)
        targets[key] = float(m.group(3))
        # speed follows the angle value: 'speed <N>'
        spd_m = re.search(r'speed\s*<([\d.]+)>', clean[m.start():m.start()+200], re.IGNORECASE)
        if spd_m:
            speeds[key] = float(spd_m.group(1))

    for m in _MOVE_RE.finditer(clean):
        key = (m.group(1).lower(), AXIS_INDEX[m.group(2).lower()], False)
        targets[key] = float(m.group(3))
        spd_m = re.search(r'speed\s*\[([\d.]+)\]', clean[m.start():m.start()+200], re.IGNORECASE)
        if spd_m:
            speeds[key] = float(spd_m.group(1))

    if not targets:
        return [], 0.0

    # Duration = slowest piece (degrees/speed for rotations, units/speed for moves)
    duration = 0.0
    for key, target in targets.items():
        speed = speeds.get(key, 60.0)
        start_val = (start_pose or {}).get(key, 0.0)
        delta = abs(target - start_val)
        if speed > 0:
            duration = max(duration, delta / speed)

    if duration < 0.01:
        duration = 1.0  # fallback

    tracks: List[BosTrack] = []
    for key, target in targets.items():
        piece, axis, is_rot = key
        start_val = (start_pose or {}).get(key, 0.0)
        kfs = [BosKeyframe(time=0.0, value=start_val),
               BosKeyframe(time=duration, value=target)]
        tracks.append(BosTrack(piece=piece, axis=axis, is_rotation=is_rot, keyframes=kfs))

    return tracks, duration


def extract_toggle_animations(bos_content: str) -> Optional[List[Tuple[str, List[BosTrack]]]]:
    """
    Extract open/close (toggle) animations for units with an Activate/Deactivate pattern.

    Looks for:
    1. Open() + Close() functions  → clips 'ActivateOpen' and 'ActivateClose'
    2. MMStatus() with if(State)/else branches → clips 'ActivateOpen' and 'ActivateClose'

    Returns list of (clip_name, tracks) pairs, or None.
    """
    bos_content = _expand_macros(bos_content)

    clips: List[Tuple[str, List[BosTrack]]] = []

    # --- Pattern 0: Go() / Stop() functions (e.g. legsolar) ---
    # Go() = open/active state, Stop() = closed/inactive state
    go_body   = _extract_function_body(bos_content, 'Go')
    stop_body = _extract_function_body(bos_content, 'Stop')
    if go_body and stop_body:
        go_body   = _inline_call_scripts(go_body,   bos_content)
        stop_body = _inline_call_scripts(stop_body, bos_content)
        # Use Create() 'now' poses as the closed start pose (initial model state)
        create_body = _extract_function_body(bos_content, 'Create')
        closed_pose: Dict[Tuple, float] = {}
        if create_body:
            clean_create = _strip_comments(create_body)
            for m in _TURN_NOW_RE.finditer(clean_create):
                key = (m.group(1).lower(), AXIS_INDEX[m.group(2).lower()], True)
                closed_pose[key] = float(m.group(3))
            for m in _MOVE_NOW_RE.finditer(clean_create):
                key = (m.group(1).lower(), AXIS_INDEX[m.group(2).lower()], False)
                closed_pose[key] = float(m.group(3))
        # Fall back to Stop() targets if Create() has no now-poses for a piece
        stop_tracks_raw, _ = _parse_turn_move_to_tracks(stop_body)
        for t in stop_tracks_raw:
            key = (t.piece, t.axis, t.is_rotation)
            if key not in closed_pose:
                closed_pose[key] = t.keyframes[-1].value
        open_tracks, open_dur = _parse_turn_move_to_tracks(go_body, start_pose=closed_pose)
        open_pose = {(t.piece, t.axis, t.is_rotation): t.keyframes[-1].value
                     for t in open_tracks}
        close_tracks, close_dur = _parse_turn_move_to_tracks(stop_body, start_pose=open_pose)
        if open_tracks and close_tracks:
            print(f"  Toggle animation 'ActivateOpen': {len(open_tracks)} tracks, {open_dur:.2f}s")
            print(f"  Toggle animation 'ActivateClose': {len(close_tracks)} tracks, {close_dur:.2f}s")
            clips.append(('ActivateOpen', open_tracks))
            clips.append(('ActivateClose', close_tracks))
            return clips

    # --- Pattern 1: Open() / Close() functions ---
    open_body = _extract_function_body(bos_content, 'Open')
    close_body = _extract_function_body(bos_content, 'Close')
    if open_body and close_body:
        open_body  = _inline_call_scripts(open_body,  bos_content)
        close_body = _inline_call_scripts(close_body, bos_content)
        open_tracks, open_dur = _parse_turn_move_to_tracks(open_body)
        # Closed pose = all targets set to 0 (the Close() targets)
        close_tracks_raw, close_dur = _parse_turn_move_to_tracks(close_body)
        closed_pose = {(t.piece, t.axis, t.is_rotation): t.keyframes[-1].value
                       for t in close_tracks_raw}

        # Re-parse Open with closed pose as the start
        open_tracks, open_dur = _parse_turn_move_to_tracks(open_body, start_pose=closed_pose)
        # Re-parse Close with open pose as the start
        open_pose = {(t.piece, t.axis, t.is_rotation): t.keyframes[-1].value
                     for t in open_tracks}
        close_tracks, close_dur = _parse_turn_move_to_tracks(close_body, start_pose=open_pose)

        if open_tracks and close_tracks:
            print(f"  Toggle animation 'ActivateOpen': {len(open_tracks)} tracks, {open_dur:.2f}s")
            print(f"  Toggle animation 'ActivateClose': {len(close_tracks)} tracks, {close_dur:.2f}s")
            clips.append(('ActivateOpen', open_tracks))
            clips.append(('ActivateClose', close_tracks))
            return clips

    # --- Pattern 2: OpenSilo() / CloseSiloDoors() functions ---
    open_silo_body = _extract_function_body(bos_content, 'OpenSilo')
    close_silo_body = _extract_function_body(bos_content, 'CloseSiloDoors')
    if open_silo_body and close_silo_body:
        open_tracks_raw, _ = _parse_turn_move_to_tracks(open_silo_body)
        close_tracks_raw, _ = _parse_turn_move_to_tracks(close_silo_body)
        closed_pose = {(t.piece, t.axis, t.is_rotation): t.keyframes[-1].value
                       for t in close_tracks_raw}
        open_tracks, open_dur = _parse_turn_move_to_tracks(open_silo_body, start_pose=closed_pose)
        open_pose = {(t.piece, t.axis, t.is_rotation): t.keyframes[-1].value
                     for t in open_tracks}
        close_tracks, close_dur = _parse_turn_move_to_tracks(close_silo_body, start_pose=open_pose)
        if open_tracks and close_tracks:
            print(f"  Toggle animation 'ActivateOpen': {len(open_tracks)} tracks, {open_dur:.2f}s")
            print(f"  Toggle animation 'ActivateClose': {len(close_tracks)} tracks, {close_dur:.2f}s")
            clips.append(('ActivateOpen', open_tracks))
            clips.append(('ActivateClose', close_tracks))
            return clips

    # --- Pattern 3: MMStatus(State) with if(State) / else branches ---
    mm_body = _extract_function_body(bos_content, 'MMStatus')
    if not mm_body:
        # Also try Activate/Deactivate as simple (non-looping) one-shots
        act_body = _extract_function_body(bos_content, 'Activate')
        deact_body = _extract_function_body(bos_content, 'Deactivate')
        if act_body and deact_body:
            open_tracks, open_dur = _parse_turn_move_to_tracks(act_body)
            close_tracks, close_dur = _parse_turn_move_to_tracks(deact_body)
            if open_tracks and close_tracks:
                print(f"  Toggle animation 'ActivateOpen': {len(open_tracks)} tracks, {open_dur:.2f}s")
                print(f"  Toggle animation 'ActivateClose': {len(close_tracks)} tracks, {close_dur:.2f}s")
                clips.append(('ActivateOpen', open_tracks))
                clips.append(('ActivateClose', close_tracks))
                return clips
        return None

    clean_mm = _strip_comments(mm_body)

    # Split on if(State) or if(Active) { ... } else { ... }
    # Find the if-block and else-block via brace counting
    if_m = re.search(r'\bif\s*\(\s*(?:State|Active)\s*\)\s*\{', clean_mm, re.IGNORECASE)
    if not if_m:
        return None

    # Extract if-block body
    start = if_m.end() - 1
    depth = 0
    if_end = start
    for i, c in enumerate(clean_mm[start:]):
        if c == '{': depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                if_end = start + i + 1
                break
    if_block = clean_mm[start:if_end]

    # Find else block
    else_m = re.search(r'\belse\s*\{', clean_mm[if_end:], re.IGNORECASE)
    if not else_m:
        return None
    else_start = if_end + else_m.end() - 1
    depth = 0
    else_end = else_start
    for i, c in enumerate(clean_mm[else_start:]):
        if c == '{': depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                else_end = else_start + i + 1
                break
    else_block = clean_mm[else_start:else_end]

    open_tracks, open_dur = _parse_turn_move_to_tracks(if_block)
    # Parse else (closed) to get start pose for open
    close_tracks_raw, _ = _parse_turn_move_to_tracks(else_block)
    closed_pose = {(t.piece, t.axis, t.is_rotation): t.keyframes[-1].value
                   for t in close_tracks_raw}
    open_tracks, open_dur = _parse_turn_move_to_tracks(if_block, start_pose=closed_pose)
    open_pose = {(t.piece, t.axis, t.is_rotation): t.keyframes[-1].value
                 for t in open_tracks}
    close_tracks, close_dur = _parse_turn_move_to_tracks(else_block, start_pose=open_pose)

    if open_tracks and close_tracks:
        print(f"  Toggle animation 'ActivateOpen': {len(open_tracks)} tracks, {open_dur:.2f}s")
        print(f"  Toggle animation 'ActivateClose': {len(close_tracks)} tracks, {close_dur:.2f}s")
        clips.append(('ActivateOpen', open_tracks))
        clips.append(('ActivateClose', close_tracks))
        return clips

    return None
