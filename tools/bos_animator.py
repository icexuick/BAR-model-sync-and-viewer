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
# Move:  move piece to y-axis [value] speed ...
_TURN_RE = re.compile(
    r'\bturn\s+(\w+)\s+to\s+([xyz])-axis\s+<([-\d.]+)>',
    re.IGNORECASE
)
_MOVE_RE = re.compile(
    r'\bmove\s+(\w+)\s+to\s+([xyz])-axis\s+\[([-\d.]+)\]',
    re.IGNORECASE
)
_FRAME_RE = re.compile(r'//\s*Frame\s*:?\s*(\d+)', re.IGNORECASE)
# Matches:  turn piece to y-axis <value> now;
_TURN_NOW_RE = re.compile(
    r'\bturn\s+(\w+)\s+to\s+([xyz])-axis\s+<([-\d.]+)>\s+now',
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


def _strip_comments(text: str) -> str:
    """Remove BOS // and /* */ comments."""
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r'//[^\n]*', '', text)
    return text


def _extract_function_body(content: str, func_name: str) -> Optional[str]:
    """
    Extract the full body (between outer braces) of a named BOS function.
    Returns body text with comments preserved (needed for Frame:N markers).
    """
    # Search directly in original content for the function opening brace.
    # We must search in original (not comment-stripped) to get correct positions.
    # Use comment-stripped only to verify the match is not inside a comment.
    pattern = re.compile(rf'\b{re.escape(func_name)}\s*\([^)]*\)\s*\{{', re.IGNORECASE)
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


def parse_create_now_rotations(bos_content: str) -> Dict[Tuple, float]:
    """
    Parse rest-pose rotations for pieces. Tries two sources in order:
    1. 'turn piece to axis <value> now' in Create() — explicit immediate pose
    2. Turn commands in StopWalking() — the idle/rest pose the unit returns to
    Returns {(piece, axis, True): degrees}.
    """
    result: Dict[Tuple, float] = {}

    # Source 1: Create() 'now' commands
    body = _extract_function_body(bos_content, 'Create')
    if body:
        for m in _TURN_NOW_RE.finditer(body):
            key = (m.group(1).lower(), AXIS_INDEX[m.group(2).lower()], True)
            result[key] = float(m.group(3))

    # Source 2: StopWalking() — fallback if Create() has no 'now' rotations.
    # Only use if StopWalking contains significant non-zero rotations (>5°),
    # indicating a non-trivial rest pose that differs from the S3O zero pose.
    if not result:
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
                result = candidate
                print(f"  Using {func_name}() as rest pose ({len(result)} rotations)")
                break

    return result


def extract_walk_animation(bos_content: str, reverse: bool = False) -> Optional[Tuple[str, List[BosTrack]]]:
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
    now_rots = parse_create_now_rotations(bos_content)

    for func_name in ['Walk', 'StartMoving', 'Move', 'DoTheWalking']:
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
            if shared_keys and all(
                abs(pre_cmds[k] - last_cmds[k]) < 0.01 for k in shared_keys
            ) and len(shared_keys) >= len(pre_cmds) * 0.6:
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
            if reverse:
                # Mirror all keyframe times: t_new = duration - t_old
                # This reverses the playback direction for units where the
                # BOS cycle is exported backwards relative to movement direction.
                for track in tracks:
                    for kf in track.keyframes:
                        kf.time = duration - kf.time
                    track.keyframes.sort(key=lambda k: k.time)
                print(f"  Animation '{func_name}': {len(tracks)} tracks, "
                      f"{n_blocks} keyframes (step={step_size}) → duration {duration:.2f}s (reversed)")
            else:
                print(f"  Animation '{func_name}': {len(tracks)} tracks, "
                      f"{n_blocks} keyframes (step={step_size}) → duration {duration:.2f}s")
            return func_name, tracks, now_rots

    return None
