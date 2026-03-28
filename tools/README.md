# BAR S3O → GLB Converter

Automated conversion of Spring RTS Engine S3O models to glTF 2.0 Binary (GLB) format, with weapon-to-piece metadata extraction from BOS/Lua animation scripts.

## What it does

1. **Parses S3O files** — the binary 3D model format used by the Spring/Recoil engine
2. **Converts to GLB** — standard glTF 2.0 Binary that Three.js can load directly
3. **Extracts weapon info** — reads BOS/Lua scripts to identify which model pieces are weapons
4. **Embeds metadata** — weapon mappings are stored as glTF `extras` on each node

## Quick Start

```bash
# Install dependency
pip install numpy

# Convert a single unit with weapon info
python convert.py --s3o objects3d/corjugg.s3o --script scripts/Units/corjugg.bos

# Batch convert an entire BAR game directory
python convert.py --bar-dir /path/to/Beyond-All-Reason --output-dir ./glb

# Just show model info (no conversion)
python convert.py --s3o objects3d/armflea.s3o --info-only
```

## File Structure

```
s3o_to_glb/
├── s3o_parser.py    — S3O binary format parser
├── s3o_to_glb.py    — GLB builder and converter
├── bos_parser.py    — BOS/Lua script parser for weapon extraction
├── convert.py       — Main CLI tool (ties everything together)
└── README.md        — This file
```

## S3O Format Reference

The S3O format is a tree of **pieces** (named mesh parts):

```
S3O Header (52 bytes)
├── magic: "Spring3DO\0\0\0"
├── version, radius, height, midpoint
├── texture names (null-terminated strings)
└── rootPieceOffset → Piece
    ├── name, offset (x,y,z)
    ├── vertices (pos + normal + UV, 32 bytes each)
    ├── indices (uint32 triangles)
    └── children → [Piece, Piece, ...]
```

Each piece becomes a **named glTF node** in the output GLB, preserving:
- Piece names (critical for script/animation mapping)
- Piece hierarchy (parent-child relationships)
- Piece offsets (local translations)
- Geometry (vertices, normals, UVs, triangle indices)

## Weapon Metadata

The converter reads BOS/Lua animation scripts to find weapon-piece mappings:

| BOS Function | What it tells us |
|---|---|
| `QueryWeaponN()` | Which piece is the **fire point** (projectile spawn) |
| `AimFromWeaponN()` | Which piece is the **aim origin** (targeting reference) |
| `AimWeaponN()` → `turn` commands | Which pieces **rotate for aiming** |

This information is embedded in the GLB as glTF `extras`:

```json
// On weapon-related nodes:
{
  "name": "turret",
  "extras": {
    "weapons": [1],
    "weapon_roles": ["aim_from", "aim_piece"]
  }
}

// On root node:
{
  "name": "base",
  "extras": {
    "s3o_texture1": "armtank_color.png",
    "s3o_texture2": "armtank_other.png",
    "weapon_count": 1,
    "weapon_summary": {
      "1": {
        "fire_point": "flare",
        "aim_from": "turret",
        "aim_pieces": ["barrel", "turret"]
      }
    }
  }
}
```

## Reading Weapon Data in Three.js (Webflow Viewer)

After loading the GLB with `GLTFLoader`, you can access weapon metadata:

```javascript
const loader = new GLTFLoader();
loader.load(modelURL, (gltf) => {
  const model = gltf.scene;
  
  // Build weapon-piece lookup
  const weaponPieces = {};
  
  model.traverse((node) => {
    if (node.userData?.weapons) {
      node.userData.weapons.forEach(weaponNum => {
        if (!weaponPieces[weaponNum]) weaponPieces[weaponNum] = [];
        weaponPieces[weaponNum].push({
          piece: node,
          name: node.name,
          roles: node.userData.weapon_roles || []
        });
      });
    }
  });
  
  // Get weapon summary from root
  const root = model.children[0];
  const weaponSummary = root?.userData?.weapon_summary;
  
  // Highlight weapon pieces on hover
  function highlightWeapon(weaponNum) {
    const pieces = weaponPieces[weaponNum] || [];
    pieces.forEach(({ piece }) => {
      piece.traverse(child => {
        if (child.isMesh) {
          child.material.emissive.setHex(0x4444ff);
          child.material.emissiveIntensity = 0.5;
        }
      });
    });
  }
  
  function unhighlightWeapon(weaponNum) {
    const pieces = weaponPieces[weaponNum] || [];
    pieces.forEach(({ piece }) => {
      piece.traverse(child => {
        if (child.isMesh) {
          child.material.emissiveIntensity = 0;
        }
      });
    });
  }
});
```

## Matching Weapons to Webflow CMS

To connect GLB weapon data to the BAR Webflow "Unit Weapons" collection:

1. **In the GLB**: each weapon has a number (1, 2, 3...) and associated piece names
2. **In the unitdef Lua file**: weapon numbers correspond to `weaponDefs` entries
3. **In Webflow CMS**: add a field like `glb_weapon_piece` to store the piece name

The weapon number in the BOS script matches the weapon index in the unitdef:

```lua
-- unitdef.lua
weaponDefs = {
  WEAPON1 = { ... },  -- This is weapon 1 → QueryWeapon1/AimWeapon1
  WEAPON2 = { ... },  -- This is weapon 2 → QueryWeapon2/AimWeapon2
}
```

## Known Limitations & Next Steps

### Current
- ✅ S3O geometry (vertices, normals, UVs, triangles)
- ✅ Piece hierarchy with named nodes
- ✅ Piece offsets as node translations
- ✅ Weapon metadata from BOS/Lua scripts
- ✅ Batch conversion

### Planned
- ⬜ **Animations** — parse BOS `walk`/`aim` functions into glTF animations
- ⬜ **Textures** — embed or reference texture files in the GLB
- ⬜ **Tri-strip/quad support** — convert non-triangle primitives (rare in BAR)
- ⬜ **Vertex AO** — preserve ambient occlusion data from S3O vertex colors
- ⬜ **Lua script support** — improved parsing for newer BAR Lua unit scripts
- ⬜ **CI/CD integration** — GitHub Actions workflow for auto-conversion

## Integration with BAR Repository

This tool is designed to eventually live inside the Beyond-All-Reason project:

```
Beyond-All-Reason/
├── objects3d/          ← S3O source files
├── scripts/Units/      ← BOS/Lua animation scripts  
├── units/              ← Unitdef Lua files
├── tools/
│   └── s3o_to_glb/     ← This converter
└── glb/                 ← Generated GLB files
```

## License

Same as Beyond All Reason — intended for use within the BAR project.
