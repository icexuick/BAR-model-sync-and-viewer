# BAR-modelviewer — Project Context

## Wat is dit project
Online 3D model viewer voor Beyond All Reason (BAR), een open-source RTS game.
De website draait op Webflow (beyondallreason.info). Modellen worden getoond met Three.js.

## Repo structuur
- `glb/` — **ALLE** geconverteerde GLB modellen komen hier. Nooit in de repo root.
- `tex/` — Shared textures (arm_color.png, cor_color.png, etc.)
- `hdr/` — HDR environment maps
- `tools/` — Python converter: S3O → GLB met weapon metadata
- `js/` — Webflow embed scripts (.html bestanden)

## Workflow regels
1. **GLB bestanden altijd in `glb/`** — gebruik `--output-dir glb/` bij convert.py. De viewer laadt van `glb/{unitName}.glb`.
2. **Na conversie altijd committen en pushen** — zodat de live viewer de nieuwe GLB kan laden van GitHub.
3. **Nooit GLBs in de repo root committen.**

## S3O → GLB Converter (tools/s3o_to_glb/)
Converteert Spring engine S3O modellen naar standaard glTF 2.0 Binary (GLB).

### Bestanden
- `s3o_parser.py` — Parseert S3O binair formaat (magic: "Spring unit\0" of "Spring3DO\0")
- `s3o_to_glb.py` — Bouwt GLB met piece-hiërarchie als named nodes
- `bos_parser.py` — Extraheert weapon→piece mappings uit BOS/Lua scripts
- `convert.py` — CLI tool, single of batch conversie

### Gebruik
```bash
python tools/s3o_to_glb/convert.py --s3o objects3d/corjugg.s3o --script scripts/Units/corjugg.bos
python tools/s3o_to_glb/convert.py --bar-dir /pad/naar/Beyond-All-Reason --output-dir ./
```

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
