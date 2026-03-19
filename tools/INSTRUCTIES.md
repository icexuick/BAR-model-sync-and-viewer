# Toevoegen aan je BAR-modelviewer repo

## Stap-voor-stap in GitHub Desktop

### 1. Bestanden kopiëren

Download de bestanden uit dit pakket en kopieer ze in je lokale repo:

```
icexuick/BAR-modelviewer/           ← je bestaande repo
├── tex/                            ← bestaat al
├── hdr/                            ← bestaat al
├── armflea.glb                     ← bestaat al
├── corjugg.glb                     ← NIEUW (gegenereerd door converter)
├── tools/                          ← NIEUWE folder
│   └── s3o_to_glb/
│       ├── README.md
│       ├── requirements.txt
│       ├── s3o_parser.py           ← S3O binary parser
│       ├── s3o_to_glb.py           ← GLB builder
│       ├── bos_parser.py           ← BOS/Lua weapon extractor
│       └── convert.py              ← Hoofdscript
└── ...
```

Dus:
- Kopieer de hele `tools/` map naar de root van je repo
- Kopieer `corjugg.glb` naar de root (naast bestaande GLBs)

### 2. Commit & Push

In **GitHub Desktop**:
1. Je ziet de nieuwe bestanden verschijnen in "Changes"
2. Typ als commit message: `Add S3O to GLB converter tools + weapon metadata extraction`
3. Klik "Commit to main"
4. Klik "Push origin"

### 3. Testen op je machine

Open een terminal/command prompt in de repo folder:

```bash
# Installeer dependency
pip install numpy

# Test met een BAR S3O bestand
cd tools/s3o_to_glb
python convert.py --s3o /pad/naar/Beyond-All-Reason/objects3d/corjugg.s3o \
                  --script /pad/naar/Beyond-All-Reason/scripts/Units/corjugg.bos \
                  --output ../../corjugg.glb

# Of batch-convert alles
python convert.py --bar-dir /pad/naar/Beyond-All-Reason --output-dir ../../
```

### 4. Verifieer de GLB

Open https://gltf-viewer.donmccurdy.com/ en sleep je `corjugg.glb` erin.
Je zou het model moeten zien (zonder textures, maar met correcte piece-structuur).
