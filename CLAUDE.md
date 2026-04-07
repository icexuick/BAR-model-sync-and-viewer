# BAR-modelviewer — Project Context

## Wat is dit project
Online 3D model viewer voor Beyond All Reason (BAR), een open-source RTS game.
De website draait op Webflow (beyondallreason.info). Modellen worden getoond met Three.js.

## Repo structuur
- **`glb/`** — **DE ENIGE PLEK** voor alle GLB modellen. Nooit ergens anders.
- `tex/` — Shared textures (arm_color.png, cor_color.png, etc.)
- `hdr/` — HDR environment maps
- `tools/` — Python converter: S3O → GLB met weapon metadata
- `js/` — Webflow embed scripts (.html bestanden)

## Git workflow — BELANGRIJK
- **Commit elke wijziging direct.** Maak na elke code- of GLB-wijziging meteen een commit, zodat alle changes traceerbaar zijn in git history en fouten makkelijk te vinden zijn.
- Na elke GLB update: direct committen + pushen (dit mag altijd zonder te vragen).
- Laat geen uncommitted changes slingeren — dat maakt debugging lastiger.

## GLB bestandslocatie — BELANGRIJK
- **Alle GLBs staan in `glb/`** — dit is de enige juiste plek.
- De viewer laadt van: `https://raw.githubusercontent.com/icexuick/BAR-modelviewer/main/glb/{unitname}.glb`
- **Zoek NOOIT naar GLBs in de repo root of ergens anders dan `glb/`.**

## Convert commando's — altijd naar `glb/`
```bash
# Enkele unit (let op: -o met pad naar glb/):
python tools/convert.py --s3o pad/naar/unit.s3o --script pad/naar/unit.bos -o glb/unitnaam.glb

# Batch (--output-dir voor batch mode):
python tools/convert.py --bar-dir C:/Games/Beyond-All-Reason/data/games/BAR.sdd --output-dir glb/

# Via GitHub (automatisch naar glb/):
python tools/convert.py --unit unitnaam --local
```

## S3O → GLB Converter (tools/)
Converteert Spring engine S3O modellen naar standaard glTF 2.0 Binary (GLB).

### Bestanden
- `s3o_parser.py` — Parseert S3O binair formaat (magic: "Spring unit\0" of "Spring3DO\0")
- `s3o_to_glb.py` — Bouwt GLB met piece-hiërarchie als named nodes
- `bos_parser.py` — Extraheert weapon→piece mappings uit BOS/Lua scripts
- `bos_animator.py` — Extraheert animaties (walk, spin, propeller, toggle) uit BOS scripts
- `convert.py` — CLI tool, single of batch conversie

### Weapon metadata in GLB
Weapon info wordt als glTF `extras` op nodes gezet:
- `node.extras.weapons` → [1, 2] (weapon nummers)
- `node.extras.weapon_roles` → ["fire_point", "aim_from", "aim_piece"]
- Root node bevat `weapon_summary` met complete mapping

### S3O Formaat
- Header: 52 bytes (magic, version, radius, height, midpoint, offsets)
- Pieces: tree van named mesh parts met offset (translation), vertices, indices
- Vertices: 32 bytes each (pos 3f + normal 3f + uv 2f)
- Magic: `"Spring unit\0"` (BAR/nieuwer) of `"Spring3DO\0\0\0"` (oud)

### BOS Scripts
- `QueryWeaponN()` → fire point piece
- `AimFromWeaponN()` → aim origin piece  
- `AimWeaponN()` → turn commands tonen aiming pieces

## Webflow Viewer (Three.js script)
Het Three.js script op Webflow laadt GLBs, past custom textures toe (diffuse + PBR + normal + team),
en gebruikt custom shaders via `onBeforeCompile`. Zie document 2 in chat history voor de volledige code.

Belangrijke Three.js details:
- Three.js v0.160.0 via unpkg importmap
- GLTFLoader + OrbitControls + SSAO + SMAA postprocessing
- Textures: `{faction}_color.png`, `{faction}_other.png`, `{faction}_normal.png`, `{faction}_team.png`
- Models laden van: `https://raw.githubusercontent.com/icexuick/BAR-modelviewer/main/glb/{unitname}.glb`
