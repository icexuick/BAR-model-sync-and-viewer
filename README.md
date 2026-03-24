# BAR-modelviewer

Online 3D model viewer for [Beyond All Reason](https://www.beyondallreason.info/) (BAR), an open-source RTS game built on the Spring engine. The viewer renders unit models and game maps using Three.js and is embedded on the BAR website via Webflow.

Live: [beyondallreason.info/units](https://www.beyondallreason.info/units/)

## Repository Structure

```
BAR-modelviewer/
├── glb/                  # All GLB models (1600+ units) — the ONLY location for GLBs
├── tex/                  # Shared faction textures (diffuse, PBR, normal, team color)
├── hdr/                  # HDR environment maps for lighting
├── js/                   # Webflow embed scripts and viewer modules
├── tools/                # Python S3O → GLB converter pipeline
├── 3d-maps*.html         # Standalone map viewers (terrain height + diffuse)
└── mapviewer.html        # Alternate map viewer
```

## Unit Viewer (`js/`)

The unit viewer is a Three.js application that loads GLB models from this repo, applies custom BAR textures and shaders, and renders them with PBR lighting, SSAO, and SMAA post-processing. It runs on the Webflow-hosted BAR website as embedded `<script>` blocks.

### Embed Scripts

Each file is a self-contained HTML embed for Webflow's custom code sections:

| File | Purpose |
|------|---------|
| `webflow-js-script.html` | **Main viewer** — loads GLB, sets up scene, camera, lights, post-processing, auto-rotate, ground shadow, PNG export. Entry point for all other scripts. |
| `webflow-js-shader.html` | **Custom shaders** — `onBeforeCompile` hooks on MeshStandardMaterial for team color blending, PBR map integration, emission/pulse effects. |
| `webflow-js-weapons.html` | **Weapon visualization** — reads weapon metadata from GLB node extras, highlights fire points / aim pieces, plays spin animations. |
| `webflow-js-animtoggle.html` | **Animation UI** — toggle buttons for walk/idle animation and text overlay visibility. Persists state in localStorage. |
| `webflow-js-editor.html` | **Editor panel** — lil-gui control panel (activated by `?editor` URL param) for tuning lights, exposure, SSAO, team color, HDR in real time. |

### Module Files

| File | Purpose |
|------|---------|
| `viewer.js` | Standalone ES module version of the unit viewer (same core as `webflow-js-script.html`). |
| `TerrainCutout.js` | Three.js Mesh class for 3D map terrain with height-based displacement mapping. |

### Key Technical Details

- **Three.js v0.160.0** via unpkg importmap
- **Textures**: `{faction}_color.png`, `{faction}_other.png`, `{faction}_normal.png`, `{faction}_team.png` (faction = arm, cor, leg)
- **Model URL**: `https://raw.githubusercontent.com/icexuick/BAR-modelviewer/main/glb/{unitname}.glb`
- **Post-processing**: SSAOPass + SMAAPass + OutputPass via EffectComposer
- **Ground shadow**: ShadowMaterial with `depthWrite: false` to prevent z-buffer occlusion
- **Import maps**: Each Webflow embed that uses ES modules includes its own `<script type="importmap">` before the `<script type="module">` (required for Firefox/Safari compatibility)

## S3O → GLB Converter (`tools/`)

Python pipeline that converts Spring engine S3O models to standard glTF 2.0 Binary (GLB) with weapon metadata, animations, and proper piece hierarchy.

### Files

| File | Purpose |
|------|---------|
| `convert.py` | **CLI entry point** — single unit, batch, or GitHub-fetch conversion modes. Handles unit defs, weapon defs, ship/fly detection, and all animation types. |
| `s3o_parser.py` | Parses S3O binary format (magic: `"Spring unit\0"` or `"Spring3DO\0"`). Extracts piece tree with vertices, normals, UVs, and indices. |
| `s3o_to_glb.py` | Builds glTF 2.0 Binary from parsed S3O data. Creates named nodes preserving the piece hierarchy, embeds PBR material references, and writes animation clips. |
| `bos_parser.py` | Extracts weapon → piece mappings from BOS/Lua animation scripts (`QueryWeaponN`, `AimFromWeaponN`, `AimWeaponN` functions). |
| `bos_animator.py` | Extracts animations from BOS scripts: walk cycles (from `Walk()`), spin animations (radar dishes, propellers), toggle animations (deploy/undeploy), and activate loops. |
| `anim_overrides.json` | Per-unit config for reversed walk animations (populated manually after verification). |

### Usage

```bash
# Single unit (always use -o to write to glb/):
python tools/convert.py --s3o path/to/unit.s3o --script path/to/unit.bos -o glb/unitname.glb

# Batch convert all units from BAR game directory:
python tools/convert.py --bar-dir C:/Games/Beyond-All-Reason/data/games/BAR.sdd --output-dir glb/

# Fetch unit from GitHub (auto-downloads S3O + script, writes to glb/):
python tools/convert.py --unit unitname --local
```

### Requirements

```bash
pip install numpy
```

### What Gets Embedded in GLB

- **Piece hierarchy** as named glTF nodes (matching S3O piece tree)
- **Weapon metadata** as node `extras`: `weapons` (numbers), `weapon_roles` (fire_point, aim_from, aim_piece), root `weapon_summary`
- **Animations**: walk cycles, spin clips (radar/propeller), toggle open/close, activate loops
- **Flags**: `is_ship`, `can_fly`, `unit_role`, `hide_pieces`, `toggleable` on root extras

### S3O Format

- Header: 52 bytes (magic, version, radius, height, midpoint, texture offsets)
- Pieces: tree of named mesh parts with offset (translation), vertices, indices
- Vertices: 32 bytes each (position 3f + normal 3f + UV 2f)

### BOS Script Parsing

BOS (Block of Script) files are human-readable animation scripts compiled to `.cob` for the Spring engine:

- `QueryWeaponN()` → fire point piece (barrel/flare)
- `AimFromWeaponN()` → aim origin piece (turret center)
- `AimWeaponN()` → turn commands reveal which pieces rotate for aiming
- `Walk()` → keyframed walk cycle (Skeletor_S3O export format with `//Frame:N` comments)
- `Activate()`/`Create()`/`StartMoving()` → spin commands for dishes, propellers, fans

## Map Viewers (Root HTML)

Standalone Three.js map viewers for BAR terrain visualization:

| File | Purpose |
|------|---------|
| `3d-maps.html` | Basic terrain viewer with heightmap displacement, water plane, and metal map overlay. |
| `3d-maps-boxgeo.html` | Enhanced terrain viewer using BoxGeometry with selective vertex displacement. |
| `3d-maps-viewer.html` | Variant of the basic terrain viewer. |
| `mapviewer.html` | Alternate map viewer implementation. |

## Assets

### Textures (`tex/`)

Shared faction textures used by all units of the same faction:

| File | Purpose |
|------|---------|
| `{faction}_color.png` | Diffuse / albedo texture |
| `{faction}_other.png` | PBR channels: roughness (R), metalness (G), ambient occlusion (B) |
| `{faction}_normal.png` | Normal map |
| `{faction}_team.png` | Team color mask (white = team color area) |

Factions: `arm` (Armada), `cor` (Cortex), `leg` (Legion)

### HDR Environment Maps (`hdr/`)

IBL (Image-Based Lighting) environment maps for reflections and ambient lighting. The viewer defaults to `clarens_midday_2k5.hdr`.

## GLB Models (`glb/`)

**All 1600+ unit GLB files live here — this is the only valid location.**

The Webflow viewer loads models directly from GitHub raw content:
```
https://raw.githubusercontent.com/icexuick/BAR-modelviewer/main/glb/{unitname}.glb
```

After any GLB conversion, commit and push immediately so the live viewer serves the updated model.

## Development

### Editor Mode

Append `?editor` to any unit viewer URL to open the lil-gui tuning panel for real-time adjustment of lights, exposure, SSAO, team color, and HDR environment.

### Adding/Updating a Unit

1. Convert: `python tools/convert.py --unit unitname --local`
2. Verify: open the unit page with `?editor` to check rendering
3. Push: `git add glb/unitname.glb && git commit -m "Update unitname.glb" && git push`

### Updating the Viewer

The Webflow embeds in `js/` are the source of truth. To deploy changes:
1. Edit the relevant `js/webflow-js-*.html` file
2. Copy the updated content into the corresponding Webflow embed block
3. Publish the Webflow site
