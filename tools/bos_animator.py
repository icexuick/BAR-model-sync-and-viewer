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
    # Search in comment-stripped content to locate the function opening brace
    clean = _strip_comments(content)
    pattern = re.compile(rf'\b{re.escape(func_name)}\s*\([^)]*\)\s*\{{', re.IGNORECASE)
    match = pattern.search(clean)
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


def extract_walk_animation(bos_content: str) -> Optional[Tuple[str, List[BosTrack]]]:
    """
    Extract walk animation tracks from a BOS script.

    Only uses while-loop frames (not the pre-loop initialisation frame).
    The pre-loop frame (e.g. Frame 0) is played once when the unit starts
    moving; the actual repeating cycle is the while-loop body only.

    Times are remapped so the first while-loop frame = t=0.
    A closing keyframe equal to t=0 is added for every track at t=duration,
    guaranteeing a seamless Three.js LoopRepeat.

    Returns (animation_name, List[BosTrack]) or None if nothing found.
    """
    for func_name in ['Walk', 'StartMoving', 'Move']:
        body = _extract_function_body(bos_content, func_name)
        if not body:
            continue

        _, while_body = _extract_while_body(body)
        if not while_body:
            # No while loop — fall back to treating the whole body as the cycle
            while_body = body

        loop_blocks = _parse_frame_blocks(while_body)
        if not loop_blocks:
            continue

        # Remap: first frame in the loop becomes t=0
        first_frame = loop_blocks[0][0]
        last_frame  = loop_blocks[-1][0]
        duration    = (last_frame - first_frame) / FPS

        # Accumulate keyframes per (piece, axis, is_rotation)
        track_dict: Dict[Tuple, List[BosKeyframe]] = {}
        for frame_num, commands in loop_blocks:
            t = (frame_num - first_frame) / FPS
            for key, value in commands.items():
                if key not in track_dict:
                    track_dict[key] = []
                track_dict[key].append(BosKeyframe(time=t, value=value))

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

            # Add closing keyframe at t=duration matching t=0 → seamless loop.
            # If the last keyframe is already at t=duration, update its value.
            closing_value = deduped[0].value
            if abs(deduped[-1].time - duration) < 1e-5:
                deduped[-1].value = closing_value
            else:
                deduped.append(BosKeyframe(time=duration, value=closing_value))

            if len(deduped) >= 2:
                tracks.append(BosTrack(piece=piece, axis=axis,
                                       is_rotation=is_rot, keyframes=deduped))

        if tracks:
            print(f"  Animation '{func_name}': {len(tracks)} tracks, "
                  f"frames {first_frame}–{last_frame} → duration {duration:.2f}s")
            return func_name, tracks

    return None
