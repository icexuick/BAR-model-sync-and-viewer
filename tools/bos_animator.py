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
    r'\bturn\s+(\w+)\s+to\s+([xyz])-axis\s+(?:\(\s*\(\s*)?(?:<([-\d.]+)>|\[([-\d.]+)\])',
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
    """Replace 'call-script FuncName()' with the body of that function (max 3 levels deep).

    Only inlines call-script (synchronous), NOT start-script (asynchronous/background).
    """
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


def parse_create_now_rotations(bos_content: str, skip_activate_flypose: bool = False,
                               include_translations: bool = False) -> Dict[Tuple, float]:
    """
    Parse rest-pose transforms for pieces. Tries two sources in order:
    1. 'turn/move piece to axis <value> now' in Create() — explicit immediate pose
    2. Turn commands in StopWalking() — the idle/rest pose the unit returns to
    Returns {(piece, axis, is_rotation): value}
      is_rotation=True  → degrees (turn)
      is_rotation=False → engine units (move)
    skip_activate_flypose: if True, skip the Activate() fly-pose scan (use for factories).
    include_translations: if True, also include 'move ... now' translations from Create().
    """
    result: Dict[Tuple, float] = {}

    # Source 1: Create() 'now' commands — rotations always, translations only when
    # explicitly requested (some units have pieces with large S3O offsets that are
    # corrected by move...now in Create, e.g. legeconv covers at y=-91 moved +100).
    body = _extract_function_body(bos_content, 'Create')
    if body:
        for m in _TURN_NOW_RE.finditer(body):
            key = (m.group(1).lower(), AXIS_INDEX[m.group(2).lower()], True)
            result[key] = float(m.group(3))
        if include_translations:
            for m in _MOVE_NOW_RE.finditer(body):
                key = (m.group(1).lower(), AXIS_INDEX[m.group(2).lower()], False)
                result[key] = float(m.group(3))

        # If Create() has a build-wait loop (while get BUILD_PERCENT_LEFT),
        # the animated turn/move commands AFTER the loop represent the
        # deployed/unfolded state (e.g. solar panels opening).  Their target
        # values override the folded 'now' values so the GLB shows the unit
        # in its operational state rather than its construction state.
        build_wait = re.search(
            r'while\s*\(\s*get\s+BUILD_PERCENT_LEFT\s*\).*?sleep\s+\d+\s*;.*?\}',
            body, re.DOTALL | re.IGNORECASE)
        if build_wait:
            post_build = body[build_wait.end():]
            overrides = 0
            for m in _TURN_RE.finditer(post_build):
                val = float(m.group(3) or m.group(4))
                key = (m.group(1).lower(), AXIS_INDEX[m.group(2).lower()], True)
                if key in result and abs(result[key] - val) > 0.1:
                    result[key] = val
                    overrides += 1
            for m in _MOVE_RE.finditer(post_build):
                val = float(m.group(3))
                key = (m.group(1).lower(), AXIS_INDEX[m.group(2).lower()], False)
                if key in result and abs(result[key] - val) > 0.1:
                    result[key] = val
                    overrides += 1
            if overrides:
                print(f"  Create() post-build override: {overrides} pieces to deployed pose")

    # Source 2: activatescr() — fly pose for aircraft.
    # Aircraft use activatescr() (state=0 = flying) to fold wings/tilt engines to
    # the in-flight position. The S3O rest pose is the landed/stored state.
    # We detect this by: activatescr exists AND has significant turn commands (>15°).
    # Last value per (piece, axis) wins (the final target angle).
    # Applied unconditionally (not blocked by Create() now_rots) so that e.g. armthund
    # can have both thrust hide-turns from Create() AND wing fly-pose from activatescr.
    for _fly_func in (() if skip_activate_flypose else ('activatescr', 'Activate')):
        body = _extract_function_body(bos_content, _fly_func)
        if not body:
            continue
        # Follow call-script / start-script references recursively (max 3 deep)
        # so that Activate() → Open() or RequestState() → Go() picks up the turns.
        expanded = _strip_comments(body)
        _seen_funcs = {_fly_func}
        _to_expand = [expanded]
        for _depth in range(3):
            new_calls = []
            for chunk in _to_expand:
                for cm in re.finditer(r'(?:call-script|start-script)\s+(\w+)\s*\(', chunk):
                    fn = cm.group(1)
                    if fn not in _seen_funcs:
                        _seen_funcs.add(fn)
                        callee = _extract_function_body(bos_content, fn)
                        if callee:
                            callee_stripped = _strip_comments(callee)
                            expanded += '\n' + callee_stripped
                            new_calls.append(callee_stripped)
            _to_expand = new_calls
            if not _to_expand:
                break
        candidate = {}
        for m in _TURN_RE.finditer(expanded):
            val = float(m.group(3))
            if abs(val) > 15.0:
                key = (m.group(1).lower(), AXIS_INDEX[m.group(2).lower()], True)
                candidate[key] = val  # last value wins = final target angle
        if candidate:
            # Also include move commands when the same pieces have turn
            # commands — these are deploy translations (e.g. legabm pivot
            # slides), not factory-door movements.
            turned_pieces = {k[0] for k in candidate}
            for m in _MOVE_RE.finditer(expanded):
                piece = m.group(1).lower()
                if piece in turned_pieces:
                    val = float(m.group(3))
                    key = (piece, AXIS_INDEX[m.group(2).lower()], False)
                    candidate[key] = val
            result.update(candidate)
            print(f"  Using {_fly_func}() as deploy pose ({len(candidate)} transforms)")
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


def extract_walk_animation(bos_content: str, skip_activate_flypose: bool = False) -> Optional[Tuple[str, List[BosTrack]]]:
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
    _is_factory = bool(re.search(r'\bOpenYard\s*\(|FACTORY_OPEN_BUILD', bos_content, re.IGNORECASE))
    now_rots = parse_create_now_rotations(bos_content, skip_activate_flypose=_is_factory or skip_activate_flypose)

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
            # add t=0 with their earliest loop-frame value so there is no jump.
            # Some pieces only appear in later loop frames (e.g. corack's lbeamhinge
            # is absent from Frame 5 but present from Frame 10+).
            for key in list(track_dict.keys()):
                if key not in pre_cmds:
                    first_kf = track_dict[key][0]
                    if first_kf.time > 1e-5:
                        track_dict[key].insert(0, BosKeyframe(time=0.0, value=first_kf.value))

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
    r'(?:'
    r'\(\s*<([^>]+)>\s*\*\s*\w+\s*\)'           # group(3): (<value> * var)
    r'|<([^>]+)>\s*\*\s*(\w+)'                   # group(4)+group(5): <value>*var
    r'|<([^>]+)>'                                 # group(6): bare <value>
    r'|(\(?-?\d*\.?\d*\)?\s*\*?\s*[A-Za-z_]\w*(?:\s*[*/]\s*-?\d+\.?\d*)?\)?)'  # group(7): variable expr
    r')',
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
        # group(3) = (<value> * var), group(4)+group(5) = <value>*var,
        # group(6) = bare <value>, group(7) = variable expr
        paren_coeff = m.group(3)    # (<-1.0> * currentspeed) → "-1.0"
        bare_coeff = m.group(4)     # <1.0>*var → "1.0"
        bare_var = m.group(5)       # <1.0>*var → "currentspeed"
        bare_val = m.group(6)       # <1.0> → "1.0"
        var_expr = m.group(7)       # variable expression
        if paren_coeff or (bare_coeff and bare_var):
            # <value> * variable — use coefficient as sign, default speed as magnitude
            coeff_str = paren_coeff or bare_coeff
            try:
                coeff = float(coeff_str)
                sign = -1.0 if coeff < 0 else 1.0
            except ValueError:
                sign = 1.0
            speed = sign * _DEFAULT_MEX_SPEED
        elif bare_val:
            raw = bare_val.strip()
            try:
                speed = float(raw)
            except ValueError:
                speed = _DEFAULT_MEX_SPEED
        else:
            raw = (var_expr or '').strip()
            try:
                speed = float(raw)
            except ValueError:
                # Variable expression — determine sign from any numeric multiplier present
                # (e.g. '-1*var', '(-0.5)*var', 'var*2', 'var / -5').
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

        # Also scan FireWeapon*/FirePrimary/FireSecondary for additional spin pieces
        # (e.g. armcir spindle that spins only during firing but is visually always spinning)
        existing_pieces = {k[0] for k in spins}
        fire_funcs = [m.group(1) for m in re.finditer(
            r'\b(Fire(?:Weapon\d*|Primary|Secondary|Tertiary))\s*\(', bos_content, re.IGNORECASE)]
        for ff in fire_funcs:
            for k, v in _collect_spin_commands(bos_content, ff).items():
                if k[0] not in existing_pieces and _is_interesting(k[0]):
                    spins[k] = v

        # Group spins by piece so multi-axis spins merge into one clip
        from collections import defaultdict as _dd
        import math as _math
        piece_axes: Dict[str, Dict[int, float]] = _dd(dict)
        for (piece, axis), speed in spins.items():
            piece_axes[piece][axis] = speed

        def _lcm_float(a: float, b: float) -> float:
            """Least common multiple of two periods (positive floats)."""
            from math import gcd
            # Work in milliseconds to avoid float precision issues
            a_ms = round(a * 1000)
            b_ms = round(b * 1000)
            if a_ms == 0 or b_ms == 0:
                return max(a, b)
            return (a_ms * b_ms // gcd(a_ms, b_ms)) / 1000.0

        clips: List[Tuple[str, List[BosTrack]]] = []
        for piece, axes in piece_axes.items():
            # Compute individual periods and find LCM duration for seamless loop
            periods = [abs(360.0 / spd) for spd in axes.values()]
            duration = periods[0]
            for p in periods[1:]:
                duration = _lcm_float(duration, p)
            # Cap at reasonable max to avoid huge clips
            duration = min(duration, 24.0)

            # Generate tracks for each axis with ≤120° keyframe steps
            # (prevents quaternion shortest-path issues in Three.js)
            piece_tracks: List[BosTrack] = []
            STEP_DEG = 120.0
            for axis, speed in axes.items():
                sign = 1.0 if speed > 0 else -1.0
                total_deg = speed * duration  # total rotation in degrees
                n_steps = max(8, int(_math.ceil(abs(total_deg) / STEP_DEG)))
                kfs = [
                    BosKeyframe(
                        time=round(duration * i / n_steps, 4),
                        value=round(total_deg * i / n_steps, 2)
                    )
                    for i in range(n_steps)
                ]
                piece_tracks.append(BosTrack(piece=piece, axis=axis, is_rotation=True, keyframes=kfs))

            clip_name = f"{func_name}_{piece}"
            clips.append((clip_name, piece_tracks))

        if clips:
            pieces = [c[0].split('_', 1)[1] for c in clips]
            print(f"  Spin animation '{func_name}': {len(clips)} spinning pieces: {', '.join(pieces)}")
            all_clips = clips
            break
    else:
        all_clips = []

    # Extra scan: propeller/screw spins from StartMoving (ships & subs).
    # These pieces are excluded from the main scan above but are the primary
    # visual element for naval units.
    _PROP_NAMES = ('prop', 'screw', 'fan')
    moving_spins = _collect_spin_commands(bos_content, 'StartMoving')
    prop_spins = {k: v for k, v in moving_spins.items()
                  if any(frag in k[0] for frag in _PROP_NAMES)}
    if prop_spins:
        import math as _math2
        from collections import defaultdict as _dd2
        existing = {c[1][0].piece.lower() for c in all_clips} if all_clips else set()
        piece_axes2: Dict[str, Dict[int, float]] = _dd2(dict)
        for (piece, axis), speed in prop_spins.items():
            if piece not in existing:
                piece_axes2[piece][axis] = speed
        STEP_DEG2 = 120.0
        for piece, axes in piece_axes2.items():
            periods = [abs(360.0 / spd) for spd in axes.values()]
            duration = periods[0]
            for p in periods[1:]:
                a_ms = round(duration * 1000); b_ms = round(p * 1000)
                from math import gcd
                duration = (a_ms * b_ms // gcd(a_ms, b_ms)) / 1000.0
            duration = min(duration, 24.0)
            piece_tracks: List[BosTrack] = []
            for axis, speed in axes.items():
                total_deg = speed * duration
                n_steps = max(8, int(_math2.ceil(abs(total_deg) / STEP_DEG2)))
                kfs = [
                    BosKeyframe(
                        time=round(duration * i / n_steps, 4),
                        value=round(total_deg * i / n_steps, 2)
                    )
                    for i in range(n_steps)
                ]
                piece_tracks.append(BosTrack(piece=piece, axis=axis, is_rotation=True, keyframes=kfs))
            all_clips.append((f"StartMoving_{piece}", piece_tracks))
        if piece_axes2:
            prop_names = ', '.join(sorted(piece_axes2.keys()))
            print(f"  Propeller spin from StartMoving: {prop_names}")

    return all_clips if all_clips else None


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
    commands from a BOS function body, respecting sleep and wait-for-turn/move as barriers.

    BOS semantics:
    - Commands fire instantly and run in the background at the given speed.
    - 'sleep N' pauses the script for N ms (background commands keep running).
    - 'wait-for-turn/move' blocks until that specific command finishes.
    - A new command on the same piece+axis interrupts the previous one.

    Strategy: tokenize into commands / sleep / wait-for-* events. Walk linearly,
    tracking in-flight commands. At each barrier, compute where each in-flight
    command has reached, then advance time.

    Returns (tracks, duration_seconds).
    """
    clean = _strip_comments(body)

    # Track in-flight commands: key → (start_time, start_val, target_val, speed)
    InFlight = Tuple[float, float, float, float]  # t_start, v_start, v_target, speed
    in_flight: Dict[Tuple, InFlight] = {}

    current_pose: Dict[Tuple, float] = dict(start_pose) if start_pose else {}
    track_kfs: Dict[Tuple, List[BosKeyframe]] = {}
    t_cursor = 0.0

    def _get_inflight_value(key: Tuple, at_time: float) -> float:
        """Get the value of a piece at a given time, accounting for in-flight commands."""
        if key in in_flight:
            t_start, v_start, v_target, spd = in_flight[key]
            if spd <= 0:
                return v_target
            elapsed = at_time - t_start
            total_dur = abs(v_target - v_start) / spd
            if elapsed >= total_dur:
                return v_target
            frac = elapsed / total_dur if total_dur > 0 else 1.0
            return v_start + (v_target - v_start) * frac
        return current_pose.get(key, 0.0)

    def _inflight_end_time(key: Tuple) -> float:
        """When will the in-flight command for this key finish?"""
        if key not in in_flight:
            return 0.0
        t_start, v_start, v_target, spd = in_flight[key]
        if spd <= 0:
            return t_start
        return t_start + abs(v_target - v_start) / spd

    def _snapshot_key(key: Tuple, at_time: float):
        """Record the current interpolated value of an in-flight key at the given time."""
        val = _get_inflight_value(key, at_time)
        if key not in track_kfs:
            track_kfs[key] = [BosKeyframe(time=0.0, value=current_pose.get(key, 0.0))]
        last = track_kfs[key][-1]
        if at_time > last.time + 0.001 or abs(val - last.value) > 0.001:
            if last.time < at_time - 0.001:
                track_kfs[key].append(BosKeyframe(time=at_time, value=val))
            elif abs(val - last.value) > 0.001:
                track_kfs[key].append(BosKeyframe(time=at_time, value=val))
        current_pose[key] = val

    def _start_command(key: Tuple, target: float, speed: float):
        """Start a new command, interrupting any in-flight command on same key."""
        # Snapshot current in-flight position before overwriting
        if key in in_flight:
            _snapshot_key(key, t_cursor)
        cur_val = current_pose.get(key, 0.0)
        in_flight[key] = (t_cursor, cur_val, target, speed)

    def _complete_inflight(key: Tuple):
        """Force-complete an in-flight command and record final keyframe."""
        if key not in in_flight:
            return
        t_start, v_start, v_target, spd = in_flight[key]
        end_t = _inflight_end_time(key)
        if key not in track_kfs:
            track_kfs[key] = [BosKeyframe(time=0.0, value=v_start)]
        last = track_kfs[key][-1]
        if end_t > last.time + 0.001 or abs(v_target - last.value) > 0.001:
            if last.time < end_t - 0.001:
                track_kfs[key].append(BosKeyframe(time=end_t, value=v_target))
            elif abs(v_target - last.value) > 0.001:
                track_kfs[key].append(BosKeyframe(time=end_t, value=v_target))
        current_pose[key] = v_target
        del in_flight[key]

    # Tokenize into events
    _BARRIER_RE = re.compile(
        r'(?:'
        r'(?P<turn>\bturn\s+\w+\s+to\s+[xyz]-axis\s+(?:\(\s*\(\s*)?(?:<[-\d.]+>|\[[-\d.]+\])[^;]*;)'
        r'|(?P<move>\bmove\s+\w+\s+to\s+[xyz]-axis\s+(?:\(\s*\(\s*\(\s*)?\[[-\d.]+\][^;]*;)'
        r'|(?P<sleep>\bsleep\s+\d+\s*;)'
        r'|(?P<wait>\bwait-for-(?:turn|move)\s+\w+\s+(?:around|along)\s+[xyz]-axis\s*;)'
        r')',
        re.IGNORECASE
    )

    _WAIT_PARSE_RE = re.compile(
        r'\bwait-for-(turn|move)\s+(\w+)\s+(?:around|along)\s+([xyz])-axis',
        re.IGNORECASE
    )

    for m in _BARRIER_RE.finditer(clean):
        if m.group('turn'):
            tm = _TURN_RE.search(m.group('turn'))
            if tm:
                key = (tm.group(1).lower(), AXIS_INDEX[tm.group(2).lower()], True)
                target = float(tm.group(3) or tm.group(4))
                spd_m = re.search(r'speed\s*<([\d.]+)>', m.group('turn'), re.IGNORECASE)
                spd = float(spd_m.group(1)) if spd_m else 60.0
                _start_command(key, target, spd)
        elif m.group('move'):
            mm = _MOVE_RE.search(m.group('move'))
            if mm:
                key = (mm.group(1).lower(), AXIS_INDEX[mm.group(2).lower()], False)
                target = float(mm.group(3))
                txt = m.group('move')
                is_now = bool(re.search(r'\]\s*now\s*;', txt, re.IGNORECASE))
                if is_now:
                    spd = 1e6
                else:
                    spd_m = re.search(r'speed\s*\[([\d.]+)\]', txt, re.IGNORECASE)
                    spd = float(spd_m.group(1)) if spd_m else 10.0
                _start_command(key, target, spd)
        elif m.group('sleep'):
            sm = _SLEEP_RE.search(m.group('sleep'))
            if sm:
                sleep_ms = int(sm.group(1))
                t_cursor += sleep_ms / 1000.0
        elif m.group('wait'):
            wm = _WAIT_PARSE_RE.search(m.group('wait'))
            if wm:
                kind = wm.group(1).lower()  # 'turn' or 'move'
                piece = wm.group(2).lower()
                axis = AXIS_INDEX[wm.group(3).lower()]
                is_rot = (kind == 'turn')
                key = (piece, axis, is_rot)
                # Block until this command finishes
                end_t = _inflight_end_time(key)
                if end_t > t_cursor:
                    t_cursor = end_t
                _complete_inflight(key)

    # Complete remaining in-flight commands.
    # If the script had meaningful timing (sleeps/waits advanced t_cursor),
    # cap background commands that would far outlast the script. Otherwise
    # (t_cursor near 0), let all commands complete — the body likely has
    # no waits and all commands should finish naturally.
    if t_cursor > 0.1:
        cutoff = t_cursor + 1.0  # 1s after last script action
        for key in list(in_flight.keys()):
            end_t = _inflight_end_time(key)
            if end_t <= cutoff:
                _complete_inflight(key)
            else:
                _snapshot_key(key, cutoff)
                del in_flight[key]
    else:
        for key in list(in_flight.keys()):
            _complete_inflight(key)

    if not track_kfs:
        return [], 0.0

    total_duration = max(kf.time for kfs in track_kfs.values() for kf in kfs)
    if total_duration < 0.01:
        total_duration = 0.5

    tracks: List[BosTrack] = []
    for key, kfs in track_kfs.items():
        piece, axis, is_rot = key
        if len(kfs) < 2:
            continue
        # Skip no-op tracks (start and end at same value)
        if abs(kfs[0].value - kfs[-1].value) < 0.001 and all(abs(kf.value - kfs[0].value) < 0.001 for kf in kfs):
            continue
        # Close off tracks that ended before total_duration
        last = kfs[-1]
        if last.time < total_duration - 0.001:
            kfs.append(BosKeyframe(time=total_duration, value=last.value))
        tracks.append(BosTrack(piece=piece, axis=axis, is_rotation=is_rot, keyframes=kfs))

    return tracks, total_duration


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

    # --- Pattern 2c: StartBuilding() / StopBuilding() (constructor nanolathe deploy) ---
    # Skip if unit has Activate/Deactivate with turn/move commands — those are factories
    # whose real open/close animation lives in Activate/Deactivate (Pattern 3).
    build_open = _extract_function_body(bos_content, 'StartBuilding')
    build_close = _extract_function_body(bos_content, 'StopBuilding')
    _act_body_check = _extract_function_body(bos_content, 'Activate')
    _deact_body_check = _extract_function_body(bos_content, 'Deactivate')
    _has_activate_motion = False
    if _act_body_check and _deact_body_check:
        _act_inlined = _strip_comments(_inline_call_scripts(_act_body_check, bos_content))
        _deact_inlined = _strip_comments(_inline_call_scripts(_deact_body_check, bos_content))
        _has_activate_motion = (
            bool(re.search(r'\b(?:turn|move)\s+\w+\s+to\s+[xyz]-axis', _act_inlined, re.IGNORECASE)) or
            bool(re.search(r'\b(?:turn|move)\s+\w+\s+to\s+[xyz]-axis', _deact_inlined, re.IGNORECASE))
        )
    if build_open and build_close and not _has_activate_motion:
        build_open  = _inline_call_scripts(build_open,  bos_content)
        build_close = _inline_call_scripts(build_close, bos_content)
        # Cap large sleeps in StopBuilding (gameplay delay before folding back,
        # not needed in viewer — e.g. "sleep 6000" → "sleep 200")
        build_close = re.sub(r'\bsleep\s+(\d+)', lambda m: f'sleep {min(int(m.group(1)), 200)}', build_close)
        close_tracks_raw, _ = _parse_turn_move_to_tracks(build_close)
        closed_pose = {(t.piece, t.axis, t.is_rotation): t.keyframes[-1].value
                       for t in close_tracks_raw}
        open_tracks, open_dur = _parse_turn_move_to_tracks(build_open, start_pose=closed_pose)
        open_pose = {(t.piece, t.axis, t.is_rotation): t.keyframes[-1].value
                     for t in open_tracks}
        close_tracks, close_dur = _parse_turn_move_to_tracks(build_close, start_pose=open_pose)
        # Skip if all animated pieces are aim/turret pieces — that's builder
        # nanolathe aiming (e.g. cornecro, corfast), not a real toggle.
        _aim_re = re.compile(r'^(aim|turret)', re.IGNORECASE)
        _all_aim = open_tracks and all(_aim_re.match(t.piece) for t in open_tracks)
        if open_tracks and close_tracks and open_dur >= 0.15 and not _all_aim:
            print(f"  Toggle animation 'ActivateOpen' (from StartBuilding): {len(open_tracks)} tracks, {open_dur:.2f}s")
            print(f"  Toggle animation 'ActivateClose' (from StopBuilding): {len(close_tracks)} tracks, {close_dur:.2f}s")
            clips.append(('ActivateOpen', open_tracks))
            clips.append(('ActivateClose', close_tracks))
            return clips

    # --- Pattern 2b-pre: StopMoving/StartMoving deploy (artillery like cormart) ---
    # Some units deploy stabilizers/tracks when stopped and retract when moving.
    # StopMoving = deploy (open), StartMoving = retract (close).
    # Require 6+ turn/move commands to avoid matching subtle tilt-resets.
    stop_body = _extract_function_body(bos_content, 'StopMoving')
    start_body = _extract_function_body(bos_content, 'StartMoving')
    if stop_body and start_body:
        stop_body = _inline_call_scripts(stop_body, bos_content)
        start_body = _inline_call_scripts(start_body, bos_content)
        _mt_re = re.compile(r'\b(?:turn|move)\s+\w+\s+to\s+[xyz]-axis', re.IGNORECASE)
        _stop_cmd_count = len(_mt_re.findall(stop_body))
        if _stop_cmd_count >= 6 and _mt_re.search(start_body):
            close_tracks_raw, _ = _parse_turn_move_to_tracks(start_body)
            closed_pose = {(t.piece, t.axis, t.is_rotation): t.keyframes[-1].value
                           for t in close_tracks_raw}
            open_tracks, open_dur = _parse_turn_move_to_tracks(stop_body, start_pose=closed_pose)
            open_pose = {(t.piece, t.axis, t.is_rotation): t.keyframes[-1].value
                         for t in open_tracks}
            close_tracks, close_dur = _parse_turn_move_to_tracks(start_body, start_pose=open_pose)
            if open_tracks and close_tracks and open_dur >= 0.15:
                print(f"  Toggle animation 'ActivateOpen' (from StopMoving): {len(open_tracks)} tracks, {open_dur:.2f}s")
                print(f"  Toggle animation 'ActivateClose' (from StartMoving): {len(close_tracks)} tracks, {close_dur:.2f}s")
                clips.append(('ActivateOpen', open_tracks))
                clips.append(('ActivateClose', close_tracks))
                return clips

    # --- Pattern 2b: AimWeapon open/close (missile launchers like armmerl, corvroc, corhrk) ---
    # These units open in AimWeapon (turn pieces to firing position) and close
    # in ExecuteRestoreAfterDelay or RestoreAfterDelay (return to rest).
    # Combine all AimWeaponN bodies so multi-weapon units (e.g. legfort with
    # left+right plasma rails) get the full deploy animation.
    aim_body = _extract_function_body(bos_content, 'AimWeapon1')
    if not aim_body:
        aim_body = _extract_function_body(bos_content, 'AimPrimary')
    for _aw_i in range(2, 10):
        _aw_body = _extract_function_body(bos_content, f'AimWeapon{_aw_i}')
        if _aw_body and aim_body:
            aim_body = aim_body + '\n' + _aw_body
    # Prefer RestoreAfterDelay over ExecuteRestoreAfterDelay when it has more
    # turn/move commands (e.g. legbar: ExecuteRestore only resets aim, while
    # RestoreAfterDelay also moves turret/barrel/cover back).
    exec_restore = _extract_function_body(bos_content, 'ExecuteRestoreAfterDelay')
    plain_restore = _extract_function_body(bos_content, 'RestoreAfterDelay')
    _move_turn_re = re.compile(r'\b(?:turn|move)\s+\w+\s+to\s+[xyz]-axis', re.IGNORECASE)
    exec_count = len(_move_turn_re.findall(exec_restore)) if exec_restore else 0
    plain_count = len(_move_turn_re.findall(plain_restore)) if plain_restore else 0
    if plain_count > exec_count and plain_restore:
        restore_body = plain_restore
    elif exec_restore:
        restore_body = exec_restore
    else:
        restore_body = plain_restore
    if aim_body and restore_body:
        aim_body = _inline_call_scripts(aim_body, bos_content)
        restore_body = _inline_call_scripts(restore_body, bos_content)
        # Also inline start-script calls in both aim and restore bodies (these call
        # functions like openAbm/closeAbm/ExecuteRestoreAfterDelay which have
        # additional turn/move commands)
        _START_SCRIPT_RE = re.compile(r'\bstart-script\s+(\w+)\s*\([^)]*\)\s*;', re.IGNORECASE)
        # For aim_body: skip Restore-related functions (those belong to close)
        _RESTORE_NAMES = {'restoreafterdelay', 'executerestoreafterdelay'}
        def _aim_start_replacer(m):
            if m.group(1).lower() in _RESTORE_NAMES:
                return ''  # strip restore calls from open body
            sub = _extract_function_body(bos_content, m.group(1))
            return _inline_call_scripts(sub, bos_content) if sub else ''
        def _restore_start_replacer(m):
            sub = _extract_function_body(bos_content, m.group(1))
            return _inline_call_scripts(sub, bos_content) if sub else ''
        aim_body = _START_SCRIPT_RE.sub(_aim_start_replacer, aim_body)
        restore_body = _START_SCRIPT_RE.sub(_restore_start_replacer, restore_body)
        # Strip leading sleeps from restore body — these are gameplay delays
        # (wait before restoring), not part of the animation.
        # Match both numeric sleeps (sleep 3000) and variable sleeps (sleep restore_delay).
        restore_body = re.sub(r'^\s*sleep\s+[\w]+\s*;', '', restore_body)
        # Merge Activate/Deactivate move/turn commands into the aim/restore
        # bodies so units like legfort get gear retract + rail deploy in one toggle.
        _act_merge = _extract_function_body(bos_content, 'Activate')
        _deact_merge = _extract_function_body(bos_content, 'Deactivate')
        if _act_merge and _deact_merge:
            _act_merge = _inline_call_scripts(_act_merge, bos_content)
            _deact_merge = _inline_call_scripts(_deact_merge, bos_content)
            _mt_re = re.compile(r'\b(?:turn|move)\s+\w+\s+to\s+[xyz]-axis', re.IGNORECASE)
            if _mt_re.search(_act_merge) and _mt_re.search(_deact_merge):
                aim_body = aim_body + '\n' + _act_merge
                restore_body = restore_body + '\n' + _deact_merge
        # Merge MoveRate1 or MoveRate2 (flight pose — thruster rotations) into
        # the open body, and MoveRate0 into close body.  This makes units like
        # legfort tilt their thrusters as part of the deploy animation.
        _mr_open = _extract_function_body(bos_content, 'MoveRate2') or \
                   _extract_function_body(bos_content, 'MoveRate1')
        _mr_close = _extract_function_body(bos_content, 'MoveRate0')
        if _mr_open and _mr_close:
            _mt_re2 = re.compile(r'\b(?:turn|move)\s+\w+\s+to\s+[xyz]-axis', re.IGNORECASE)
            if _mt_re2.search(_mr_open) and _mt_re2.search(_mr_close):
                aim_body = aim_body + '\n' + _mr_open
                restore_body = restore_body + '\n' + _mr_close
        aim_clean = _strip_comments(aim_body)
        restore_clean = _strip_comments(restore_body)
        # Only use this pattern if AimWeapon has turn/move commands with wait-for
        # (indicating a deliberate open animation, not just aiming).
        # Also accept when 3+ distinct non-aim pieces are animated (e.g. silo
        # panels like anpanelf/anpanell/anpanelr) — clearly a deploy, not aiming.
        has_open_wait = bool(re.search(r'wait-for-(turn|move)', aim_clean, re.IGNORECASE))
        _aim_piece_names = set(re.findall(r'\b(?:turn|move)\s+(\w+)\s+to\s+[xyz]-axis', aim_clean, re.IGNORECASE))
        # If AimWeapon has no heading/pitch variable turns, it's a pure deploy
        # (e.g. cormh tilts turret 90° to fire), not turret aiming. In that case
        # aim-named pieces with large rotations count as deploy pieces.
        _aim1_body = _extract_function_body(bos_content, 'AimWeapon1') or ''
        _aim1_clean = _strip_comments(_aim1_body)
        _has_aim_vars = bool(re.search(r'turn\s+\w+\s+to\s+[xyz]-axis\s+(?:heading|.*pitch)', _aim1_clean, re.IGNORECASE))
        _large_rot_pieces = set()
        if not _has_aim_vars:
            for _lr_m in re.finditer(r'\bturn\s+(\w+)\s+to\s+[xyz]-axis\s+<([-\d.]+)>', aim_clean, re.IGNORECASE):
                if abs(float(_lr_m.group(2))) >= 45:
                    _large_rot_pieces.add(_lr_m.group(1).lower())
        _non_aim_pieces = {p for p in _aim_piece_names if not re.match(r'^(aim|turret|sleeve|gun|barrel|aimx|aimy|flare)', p, re.IGNORECASE) or p.lower() in _large_rot_pieces}
        has_many_deploy_pieces = len(_non_aim_pieces) >= 3
        has_open_moves = bool(re.search(r'\b(?:turn|move)\s+\w+\s+to\s+[xyz]-axis', aim_clean, re.IGNORECASE))
        has_close_moves = bool(re.search(r'\b(?:turn|move)\s+\w+\s+to\s+[xyz]-axis', restore_clean, re.IGNORECASE))
        # Don't conflict with existing Open()/Close() pattern
        has_open_close_fn = bool(_extract_function_body(bos_content, 'Open') and
                                 _extract_function_body(bos_content, 'Close'))
        # Skip if only aim-related pieces are animated — that's just turret
        # aiming, not a deploy (e.g. corwolv, armart artillery units).
        has_non_aim_pieces = len(_non_aim_pieces) >= 1
        if (has_open_wait or has_many_deploy_pieces) and has_open_moves and has_close_moves and not has_open_close_fn and has_non_aim_pieces:
            # Parse close first to get closed pose (= rest position targets)
            close_tracks_raw, _ = _parse_turn_move_to_tracks(restore_body)
            closed_pose = {(t.piece, t.axis, t.is_rotation): t.keyframes[-1].value
                           for t in close_tracks_raw}
            open_tracks, open_dur = _parse_turn_move_to_tracks(aim_body, start_pose=closed_pose)
            open_pose = {(t.piece, t.axis, t.is_rotation): t.keyframes[-1].value
                         for t in open_tracks}
            close_tracks, close_dur = _parse_turn_move_to_tracks(restore_body, start_pose=open_pose)
            if open_tracks and close_tracks and open_dur >= 0.15:
                print(f"  Toggle animation 'ActivateOpen' (from AimWeapon): {len(open_tracks)} tracks, {open_dur:.2f}s")
                print(f"  Toggle animation 'ActivateClose' (from Restore): {len(close_tracks)} tracks, {close_dur:.2f}s")
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
            act_body  = _inline_call_scripts(act_body,  bos_content)
            deact_body = _inline_call_scripts(deact_body, bos_content)
            # Cap large sleeps in Deactivate (gameplay delay, e.g. "sleep 5000")
            deact_body = re.sub(r'\bsleep\s+(\d+)', lambda m: f'sleep {min(int(m.group(1)), 200)}', deact_body)
            # Parse close first to get closed pose (rest position)
            close_tracks_raw, _ = _parse_turn_move_to_tracks(deact_body)
            closed_pose = {(t.piece, t.axis, t.is_rotation): t.keyframes[-1].value
                           for t in close_tracks_raw}
            open_tracks, open_dur = _parse_turn_move_to_tracks(act_body, start_pose=closed_pose)
            # Fill in missing open tracks: if Deactivate moves pieces that
            # Activate doesn't mention, generate open tracks that reverse
            # from the closed pose to 0 (S3O rest).  This handles factories
            # like legsplab where arms are opened via StartBuilding/MoveCranes
            # but closed explicitly in Deactivate.
            open_keys = {(t.piece, t.axis, t.is_rotation) for t in open_tracks}
            for ct in close_tracks_raw:
                key = (ct.piece, ct.axis, ct.is_rotation)
                if key not in open_keys:
                    start_val = closed_pose.get(key, 0.0)
                    end_val = 0.0  # S3O rest pose
                    if abs(start_val - end_val) > 0.01:
                        # Use same duration as the existing open animation
                        open_tracks.append(BosTrack(
                            piece=ct.piece, axis=ct.axis, is_rotation=ct.is_rotation,
                            keyframes=[BosKeyframe(0.0, start_val), BosKeyframe(open_dur, end_val)]
                        ))
            open_pose = {(t.piece, t.axis, t.is_rotation): t.keyframes[-1].value
                         for t in open_tracks}
            close_tracks, close_dur = _parse_turn_move_to_tracks(deact_body, start_pose=open_pose)
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


# ── Fire / recoil animation extraction ──────────────────────────────

# Maps legacy names → weapon number
_FIRE_LEGACY = {'FirePrimary': 1, 'FireSecondary': 2, 'FireTertiary': 3}

_FIRE_SLEEP_RE = re.compile(r'\bsleep\s+(\d+)\s*;', re.IGNORECASE)


def _extract_branch_block(body: str, start_pos: int) -> Tuple[str, int]:
    """Extract a { ... } block starting from start_pos, return (content, end_pos)."""
    brace_start = body.index('{', start_pos)
    depth, i = 0, brace_start
    while i < len(body):
        if body[i] == '{': depth += 1
        elif body[i] == '}':
            depth -= 1
            if depth == 0: break
        i += 1
    return body[brace_start + 1:i], i + 1


def _sequence_if_branches(body: str) -> Tuple[str, int, Optional[Tuple[str, int, float, float]]]:
    """
    For fire functions with alternating barrel branches, detect them and merge
    all branches into a single body so all barrels animate simultaneously.

    Handles patterns:
      - if(gun == 0) { ... } if(gun == 1) { ... }  (N branches)
      - if(!gun_1) { ... } else { ... }             (2 branches)
      - if(gun_1) { ... } else { ... }              (2 branches)

    Returns (merged_body, num_branches, rotary_info).
    merged_body is pre_branch + all branch bodies concatenated + post_branch.
    rotary_info is (piece, axis, step_degrees, speed) for gatling-style advance, or None.
    """
    move_turn_re = re.compile(r'\b(?:move|turn)\s+\w+\s+to\s+[xyz]-axis', re.IGNORECASE)

    # --- Pattern A: if(!var) { ... } else { ... }  or  if(var) { ... } else { ... } ---
    bool_pattern = re.compile(
        r'\bif\s*\(\s*!?\s*(\w+)\s*\)\s*\{', re.IGNORECASE
    )
    bool_m = bool_pattern.search(body)
    if bool_m:
        counter_var = bool_m.group(1)
        pre_branch = body[:bool_m.start()]
        try:
            branch1_body, branch1_end = _extract_branch_block(body, bool_m.start())
        except (ValueError, IndexError):
            branch1_body = None
            branch1_end = 0
        if branch1_body:
            rest = body[branch1_end:]
            else_m = re.match(r'\s*else\s*\{', rest, re.IGNORECASE)
            if else_m:
                try:
                    branch2_body, branch2_end = _extract_branch_block(rest, else_m.start())
                except (ValueError, IndexError):
                    branch2_body = None
                    branch2_end = 0
                if branch2_body:
                    post_branch = rest[branch2_end:]
                    branch_bodies = []
                    if move_turn_re.search(branch1_body):
                        branch_bodies.append(branch1_body)
                    if move_turn_re.search(branch2_body):
                        branch_bodies.append(branch2_body)
                    if len(branch_bodies) >= 2:
                        merged = pre_branch + '\n'.join(branch_bodies) + post_branch
                        return merged, len(branch_bodies), None

    # --- Pattern B: if(gun == 0) { ... } if(gun == 1) { ... } ... ---
    gun_pattern = re.compile(
        r'\bif\s*\(\s*(\w+)\s*==\s*(\d+)\s*\)', re.IGNORECASE
    )
    matches = list(gun_pattern.finditer(body))
    if len(matches) < 2:
        return body, 0, None

    counter_var = matches[0].group(1)
    values = [int(m.group(2)) for m in matches]
    start_val = values[0]
    if start_val not in (0, 1) or values != list(range(start_val, start_val + len(values))):
        return body, 0, None

    pre_branch = body[:matches[0].start()]

    # Extract ALL branch bodies, skip branches without move/turn (reset-only branches)
    branch_bodies = []
    for m in matches:
        branch_body, _ = _extract_branch_block(body, m.end())
        if move_turn_re.search(branch_body):
            branch_bodies.append(branch_body)
    if not branch_bodies:
        return body, 0, None
    num_branches = len(branch_bodies)

    # Find post-branch code (after the last branch block)
    _, last_end = _extract_branch_block(body, matches[-1].end())
    post_branch = body[last_end:]

    # Detect rotary advance in post-branch or first branch
    rotary_info = None
    search_text = post_branch + '\n' + branch_bodies[0]

    # Pattern 1: angle multiplied by counter (with optional parens)
    rotary_re1 = re.compile(
        r'\bturn\s+(\w+)\s+to\s+([xyz])-axis\s+<([\d.]+)>\s*\*\s*\(?\s*' + re.escape(counter_var)
        + r'\s*\)?[^;]*speed\s+<([\d.]+)>', re.IGNORECASE
    )
    rm = rotary_re1.search(search_text)
    if rm:
        step_deg = float(rm.group(3))
        raw_speed = float(rm.group(4))
        speed_pos = rm.end(4)
        after_speed = search_text[speed_pos:speed_pos + 30]
        if re.match(r'>\s*\*\s*\w+', after_speed):
            raw_speed = 360.0
        rotary_info = (rm.group(1).lower(), AXIS_INDEX[rm.group(2).lower()],
                        step_deg, raw_speed)

    # Pattern 2: fixed angle with variable speed
    if not rotary_info:
        rotary_re2 = re.compile(
            r'\bturn\s+(\w+)\s+to\s+([xyz])-axis\s+<([\d.]+)>[^;]*speed\s+<[\d.]+>\s*\*\s*\w+',
            re.IGNORECASE
        )
        rm = rotary_re2.search(search_text)
        if rm:
            rotary_info = (rm.group(1).lower(), AXIS_INDEX[rm.group(2).lower()],
                            float(rm.group(3)), 360.0)

    # For rotary weapons (gatling spindle), use only the first branch so
    # the clip fires one barrel per shot.  The JS viewer accumulates the
    # spindle rotation, cycling through barrels on each click.
    # For non-rotary multi-barrel weapons, merge all branches so all
    # barrels animate simultaneously (staggered by the JS player).
    if rotary_info:
        merged = pre_branch + branch_bodies[0] + post_branch
    else:
        merged = pre_branch + '\n'.join(branch_bodies) + post_branch
    return merged, num_branches, rotary_info


_BARREL_SPIN_DUR = 4.0  # total barrel spin animation duration (seconds)
_BARREL_SPIN_REVS = 5   # number of full revolutions during constant-speed phase


def _make_barrel_spin_track(piece: str, axis: int, speed_deg: float) -> BosTrack:
    """
    Generate a barrel spin track: ramp-up, N revolutions at speed, ramp-down.

    Keyframes are placed every 120° (max) so Three.js quaternion slerp
    interpolates the correct direction (slerp takes shortest path, so
    >180° jumps would reverse).
    """
    RAMP = 0.4  # ramp-up / ramp-down time
    REVS = _BARREL_SPIN_REVS
    DUR = _BARREL_SPIN_DUR
    # Use a constant visual speed based on revolutions, ignoring BOS speed
    # (BOS speeds vary wildly; we want a consistent nice-looking spin)
    sign = 1.0 if speed_deg >= 0 else -1.0
    total_spin_deg = sign * REVS * 360.0
    constant_dur = DUR - 2 * RAMP
    deg_per_sec = total_spin_deg / constant_dur if constant_dur > 0 else total_spin_deg

    # Ramp-up phase: 0 → RAMP (accelerating, covers half the per-second distance)
    ramp_deg = deg_per_sec * RAMP * 0.5
    # Constant phase: RAMP → DUR-RAMP
    constant_deg = deg_per_sec * constant_dur
    # Ramp-down phase: last RAMP seconds (decelerating)
    end_deg = ramp_deg + constant_deg + deg_per_sec * RAMP * 0.5

    # Build keyframes with max 120° steps to avoid quaternion shortest-path issues
    STEP_DEG = 120.0
    kfs = [BosKeyframe(time=0.0, value=0.0)]

    def _add_segment(t_start: float, t_end: float, v_start: float, v_end: float):
        """Add keyframes for a segment, subdividing if the angular change > STEP_DEG."""
        delta = abs(v_end - v_start)
        if delta < 0.01:
            return
        n_steps = max(1, int(delta / STEP_DEG))
        for i in range(1, n_steps + 1):
            frac = i / n_steps
            t = t_start + (t_end - t_start) * frac
            v = v_start + (v_end - v_start) * frac
            kfs.append(BosKeyframe(time=round(t, 4), value=round(v, 2)))

    _add_segment(0.0, RAMP, 0.0, ramp_deg)
    _add_segment(RAMP, DUR - RAMP, ramp_deg, ramp_deg + constant_deg)
    _add_segment(DUR - RAMP, DUR, ramp_deg + constant_deg, end_deg)

    return BosTrack(piece=piece, axis=axis, is_rotation=True, keyframes=kfs)


def _parse_fire_body_to_tracks(body: str) -> Tuple[List[BosTrack], float]:
    """
    Parse a fire function body into animation tracks.

    Fire functions use a mix of:
      - move/turn commands (recoil + return)
      - sleep N  (delay in ms)
      - wait-for-move / wait-for-turn  (phase barriers)

    Strategy: walk through the body linearly, collecting move/turn commands.
    When a sleep or wait-for is encountered, advance the time cursor and
    commit pending commands as keyframes.  After all commands, ensure pieces
    return to 0 (rest) if not already there.
    """
    # Handle if(gun==N) branching — merges all branches into one body
    branch_result, num_branches, rotary_info = _sequence_if_branches(body)
    body = branch_result

    # Also detect rotary patterns directly (for scripts without if(gun==N) branches)
    # Pattern: turn PIECE to AXIS <ANGLE> * VAR speed <SPD>
    if not rotary_info:
        clean_pre = _strip_comments(body)
        rm_direct = re.search(
            r'\bturn\s+(\w+)\s+to\s+([xyz])-axis\s+<([\d.]+)>\s*\*\s*\w+[^;]*speed\s+<([\d.]+)>',
            clean_pre, re.IGNORECASE
        )
        if rm_direct:
            rotary_info = (
                rm_direct.group(1).lower(),
                AXIS_INDEX[rm_direct.group(2).lower()],
                float(rm_direct.group(3)),
                float(rm_direct.group(4))
            )

    clean = _strip_comments(body)

    # Tokenize into: (type, data) where type is 'move', 'turn', 'sleep', 'wait'
    tokens = []
    for m in re.finditer(
        r'(?:'
        r'(?P<turn>\bturn\s+\w+\s+to\s+[xyz]-axis\s+(?:\(\s*\(\s*)?(?:<[-\d.]+>|\[[-\d.]+\])[^;]*;)'
        r'|(?P<move>\bmove\s+\w+\s+to\s+[xyz]-axis\s+(?:\(\s*\(\s*\(\s*)?\[[-\d.]+\][^;]*;)'
        r'|(?P<sleep>\bsleep\s+\d+\s*;)'
        r'|(?P<wait>\bwait-for-(?:turn|move)\s+\w+\s+(?:around|along)\s+[xyz]-axis\s*;)'
        r')',
        clean, re.IGNORECASE
    ):
        if m.group('turn'):
            tm = _TURN_RE.search(m.group('turn'))
            if tm:
                # Skip turn commands with variable expressions (e.g. <60> * gun_1)
                turn_txt = m.group('turn')
                angle_end = turn_txt.find('>') if '>' in turn_txt else turn_txt.find(']')
                if angle_end >= 0:
                    after_angle = turn_txt[angle_end + 1:]
                    if re.search(r'\*\s*\w+', after_angle):
                        continue
                spd_m = re.search(r'speed\s*<([\d.]+)>', turn_txt, re.IGNORECASE)
                spd = float(spd_m.group(1)) if spd_m else 60.0
                tokens.append(('cmd', (tm.group(1).lower(), AXIS_INDEX[tm.group(2).lower()], True),
                               float(tm.group(3) or tm.group(4)), spd))
        elif m.group('move'):
            mm = _MOVE_RE.search(m.group('move'))
            if mm:
                txt = m.group('move')
                # 'now' keyword = instant move (infinite speed)
                is_now = bool(re.search(r'\]\s*now\s*;', txt, re.IGNORECASE))
                if is_now:
                    spd = 1e6  # effectively instant
                else:
                    spd_m = re.search(r'speed\s*\[([\d.]+)\]', txt, re.IGNORECASE)
                    spd = float(spd_m.group(1)) if spd_m else 10.0
                tokens.append(('cmd', (mm.group(1).lower(), AXIS_INDEX[mm.group(2).lower()], False),
                               float(mm.group(3)), spd))
        elif m.group('sleep'):
            sm = _SLEEP_RE.search(m.group('sleep'))
            if sm:
                tokens.append(('sleep', int(sm.group(1))))
        elif m.group('wait'):
            tokens.append(('wait',))

    # Detect spin commands (minigun/gatling barrels) — if no move/turn tokens
    # were found, generate a multi-revolution spin animation.
    _SPIN_FIRE_RE = re.compile(
        r'\bspin\s+(\w+)\s+around\s+([xyz])-axis\s+speed\s+<([-\d.]+)>',
        re.IGNORECASE
    )
    spin_cmds = list(_SPIN_FIRE_RE.finditer(clean))
    if not tokens and spin_cmds:
        spin_tracks: List[BosTrack] = []
        for sm in spin_cmds:
            piece = sm.group(1).lower()
            axis = AXIS_INDEX[sm.group(2).lower()]
            speed_deg = float(sm.group(3))
            spin_tracks.append(_make_barrel_spin_track(piece, axis, speed_deg))
        if spin_tracks:
            return spin_tracks, _BARREL_SPIN_DUR, rotary_info

    if not tokens and not rotary_info:
        return [], 0.0, None

    # Walk tokens, build keyframes
    current_pose: Dict[Tuple, float] = {}
    track_kfs: Dict[Tuple, List[BosKeyframe]] = {}
    t_cursor = 0.0
    pending_cmds: List[Tuple] = []  # (key, target, speed)

    def flush_pending():
        """Commit pending commands as keyframes at t_cursor."""
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
            # Each command uses its own duration based on speed
            if speed >= 1e5:
                # 'now' = instant — use tiny epsilon so the step is visible
                # (two keyframes at identical times get collapsed by interpolation)
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
        # Advance cursor to the latest completion time so subsequent
        # sleeps/commands don't overlap with unfinished motions
        t_cursor = max_end_t

    for tok in tokens:
        if tok[0] == 'cmd':
            _, key, target, speed = tok
            pending_cmds.append((key, target, speed))
        elif tok[0] == 'sleep':
            flush_pending()
            t_cursor += tok[1] / 1000.0
        elif tok[0] == 'wait':
            flush_pending()

    flush_pending()

    if not track_kfs and not rotary_info:
        return [], 0.0, None

    # Ensure all tracks return to 0 at the end (rest pose)
    for key, kfs in track_kfs.items():
        last_val = kfs[-1].value
        if abs(last_val) > 0.001:
            # Compute return time based on the last speed used
            return_dur = abs(last_val) / 10.0  # default slow return
            # Look for the last cmd that targeted 0 to get its speed
            for tok in reversed(tokens):
                if tok[0] == 'cmd' and tok[1] == key and abs(tok[2]) < 0.001:
                    return_dur = abs(last_val) / max(tok[3], 0.1)
                    break
            t_cursor_end = kfs[-1].time + 0.01  # small gap
            kfs.append(BosKeyframe(time=t_cursor_end, value=last_val))
            kfs.append(BosKeyframe(time=t_cursor_end + return_dur, value=0.0))

    # Add rotary advance track if detected (e.g. gatling spindle rotation)
    # Always a single step per clip — the JS viewer accumulates rotation on each fire.
    if rotary_info:
        r_piece, r_axis, step_deg, r_speed = rotary_info
        r_key = (r_piece, r_axis, True)
        r_dur_per_step = step_deg / max(r_speed, 1.0)
        r_kfs = [
            BosKeyframe(time=0.0, value=0.0),
            BosKeyframe(time=0.0, value=0.0),
            BosKeyframe(time=r_dur_per_step, value=step_deg),
        ]
        track_kfs[r_key] = r_kfs

    # Compute total duration
    if track_kfs:
        total_duration = max(kf.time for kfs in track_kfs.values() for kf in kfs)
    else:
        total_duration = 0.5
    if total_duration < 0.01:
        total_duration = 0.5

    tracks: List[BosTrack] = []
    for key, kfs in track_kfs.items():
        if len(kfs) < 2:
            continue
        # Skip tracks where all keyframes have value 0 (no-op)
        if all(abs(kf.value) < 0.001 for kf in kfs):
            continue
        piece, axis, is_rot = key
        tracks.append(BosTrack(piece=piece, axis=axis, is_rotation=is_rot, keyframes=kfs))

    # rotary_info stays as single-step (step_deg per shot) —
    # the JS viewer accumulates rotation across fires.

    return tracks, total_duration, rotary_info


# Return type for fire animations: list of (clip_name, tracks, rotary_info) triples
FireClipInfo = Tuple[str, List[BosTrack], Optional[Tuple[str, int, float, float]]]


def extract_fire_animations(bos_content: str) -> Optional[List[FireClipInfo]]:
    """
    Extract weapon fire/recoil animations from BOS scripts.

    Looks for FireWeapon1(), FireWeapon2(), ..., FirePrimary(), FireSecondary(), etc.
    Inlines call-script references (e.g. call-script fireCommon()).

    Returns list of (clip_name, tracks, rotary_info) triples.
    rotary_info is (piece, axis, step_degrees, speed) for gatling-style advance, or None.
    Returns None if no fire animations with actual movement found.
    """
    bos_content = _expand_macros(bos_content)
    clips: List[FireClipInfo] = []
    seen_weapons: set = set()

    # Try FireWeapon1..FireWeapon16, falling back to Shot1..Shot16 for recoil
    for n in range(1, 17):
        func_name = f'FireWeapon{n}'
        body = _extract_function_body(bos_content, func_name)
        if body:
            body = _inline_call_scripts(body, bos_content)
            tracks, dur, rotary = _parse_fire_body_to_tracks(body)
        else:
            tracks, dur, rotary = [], 0.0, None
        # If FireWeaponN has no recoil tracks, try ShotN (per-shot callback
        # used by multi-barrel weapons like corblackhy, armepoch)
        if not tracks:
            shot_body = _extract_function_body(bos_content, f'Shot{n}')
            if shot_body:
                shot_body = _inline_call_scripts(shot_body, bos_content)
                tracks, dur, rotary = _parse_fire_body_to_tracks(shot_body)
                if tracks:
                    func_name = f'Shot{n}'
        if tracks:
            clip_name = f'Fire_{n}'
            pieces = sorted({t.piece for t in tracks})
            rotary_str = f", rotary: {rotary[0]} +{rotary[2]}°" if rotary else ""
            print(f"  Fire animation '{clip_name}' (from {func_name}): {len(tracks)} tracks, "
                  f"{dur:.2f}s, pieces: {', '.join(pieces)}{rotary_str}")
            clips.append((clip_name, tracks, rotary))
            seen_weapons.add(n)

    # Try legacy names
    for legacy_name, wnum in _FIRE_LEGACY.items():
        if wnum in seen_weapons:
            continue
        body = _extract_function_body(bos_content, legacy_name)
        if not body:
            continue
        body = _inline_call_scripts(body, bos_content)
        tracks, dur, rotary = _parse_fire_body_to_tracks(body)
        if tracks:
            clip_name = f'Fire_{wnum}'
            pieces = sorted({t.piece for t in tracks})
            rotary_str = f", rotary: {rotary[0]} +{rotary[2]}°" if rotary else ""
            print(f"  Fire animation '{clip_name}' (from {legacy_name}): {len(tracks)} tracks, "
                  f"{dur:.2f}s, pieces: {', '.join(pieces)}{rotary_str}")
            clips.append((clip_name, tracks, rotary))

    # Detect barrel spins in AimWeapon/FireWeapon functions.
    # For units with existing fire clips: add spin tracks to them.
    # For units without fire clips (minigun-only): create a 3s spin animation.
    _SPIN_CMD_RE = re.compile(
        r'\bspin\s+(\w+)\s+around\s+([xyz])-axis\s+speed\s+<([-\d.]+)>',
        re.IGNORECASE
    )
    for n in range(1, 17):
        # Collect spins from AimWeapon, FireWeapon, and Shot (per-shot callback)
        all_spins = []
        for func in [f'AimWeapon{n}', f'FireWeapon{n}', f'Shot{n}']:
            fb = _extract_function_body(bos_content, func)
            if not fb:
                if func == 'AimWeapon1':
                    fb = _extract_function_body(bos_content, 'AimPrimary')
                elif func == 'FireWeapon1':
                    fb = _extract_function_body(bos_content, 'FirePrimary')
            if fb:
                fb = _strip_comments(_inline_call_scripts(fb, bos_content))
                all_spins.extend(_SPIN_CMD_RE.finditer(fb))
        if not all_spins:
            continue

        # Deduplicate by piece+axis (same barrel may appear in both Aim and Fire)
        seen_spin_keys = set()
        unique_spins = []
        for sm in all_spins:
            key = (sm.group(1).lower(), sm.group(2).lower())
            if key not in seen_spin_keys:
                seen_spin_keys.add(key)
                unique_spins.append(sm)

        # Find existing Fire_N clip for this weapon
        clip_idx = None
        for ci, (cname, ctracks, crotary) in enumerate(clips):
            if cname == f'Fire_{n}':
                clip_idx = ci
                break

        if clip_idx is not None:
            # Add spin tracks to existing fire clip (skip pieces already animated)
            cname, ctracks, crotary = clips[clip_idx]
            existing_keys = {(t.piece, t.axis) for t in ctracks}
            clip_dur = max((kf.time for t in ctracks for kf in t.keyframes), default=0.5)
            added = []
            for sm in unique_spins:
                piece = sm.group(1).lower()
                axis = AXIS_INDEX[sm.group(2).lower()]
                if (piece, axis) in existing_keys:
                    continue  # already has animation on this piece+axis
                speed_deg = float(sm.group(3))
                total_deg = speed_deg * clip_dur
                spin_kfs = [
                    BosKeyframe(time=0.0, value=0.0),
                    BosKeyframe(time=clip_dur, value=total_deg),
                ]
                ctracks.append(BosTrack(piece=piece, axis=axis, is_rotation=True, keyframes=spin_kfs))
                added.append(piece)
            if added:
                print(f"  Fire '{cname}': added barrel spin for {', '.join(added)}")
                clips[clip_idx] = (cname, ctracks, crotary)
        else:
            # No existing fire clip — create a barrel spin animation (minigun/gatling)
            spin_tracks = []
            added = []
            for sm in unique_spins:
                piece = sm.group(1).lower()
                axis = AXIS_INDEX[sm.group(2).lower()]
                speed_deg = float(sm.group(3))
                spin_tracks.append(_make_barrel_spin_track(piece, axis, speed_deg))
                added.append(piece)
            if spin_tracks:
                clip_name = f'Fire_{n}'
                print(f"  Fire animation '{clip_name}' (barrel spin): {len(spin_tracks)} tracks, "
                      f"{_BARREL_SPIN_DUR:.1f}s, pieces: {', '.join(added)}")
                clips.append((clip_name, spin_tracks, None))
                seen_weapons.add(n)

    return clips if clips else None
