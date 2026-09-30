"""Lay out the actual renderer outputs and verify their export metadata."""
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
CONCEPTS = [
    ("01-grassland", "Dry grassland", "Rounded limestone, olive grasses and low scrub"),
    ("02-woodland", "Open woodland", "Tree trunks, leaf litter and fallen wood"),
    ("03-dune", "Coastal dune", "Pale sand, fine grasses and weathered driftwood"),
    ("04-garden", "Garden edge", "Broad leaves, small flowers and rounded stones"),
]


def font(size, bold=False):
    for path, index in (("/System/Library/Fonts/Avenir Next.ttc", 2 if bold else 7),
                        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0)):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size, index=index)
    return ImageFont.load_default(size=size)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    canvas = Image.new("RGB", (1848, 1480), "#111a1d")
    d = ImageDraw.Draw(canvas)
    d.text((30, 18), "Four terrain directions", font=font(43, True), fill="#edf1ea")
    d.text((32, 77), "Ant-eye views  /  camera height 10 mm  /  sun azimuth 35°, elevation 38°", font=font(23), fill="#bdc9c5")
    exports = []
    for i, (stem, title, description) in enumerate(CONCEPTS):
        path = HERE / f"{stem}.png"
        pano = HERE / f"{stem}-panorama.png"
        scene = HERE / f"{stem}.blend"
        meta = json.loads((HERE / f"{stem}.json").read_text())
        assert not meta["draft"]
        assert meta["sun"]["azimuth_deg"] == 35 and meta["sun"]["elevation_deg"] == 38
        assert meta["camera"]["eye_height_above_ground_m"] == .01
        with Image.open(path) as im:
            assert im.size == (1800, 1125)
            im.load()
            preview = im.convert("RGB").resize((888, 555), Image.Resampling.LANCZOS)
        with Image.open(pano) as im:
            assert im.size == (1776, 450)
            im.load()
        assert scene.exists() and scene.stat().st_size > 0
        x, y = 24+(i % 2)*912, 132+(i // 2)*650
        d.rounded_rectangle((x, y, x+888, y+628), radius=12, fill="#213034")
        canvas.paste(preview, (x, y))
        d.text((x+16, y+566), f"{i+1:02d}   {title}", font=font(27, True), fill="#eef2ec")
        d.text((x+17, y+602), description, font=font(18), fill="#c0cdca")
        exports.append(dict(concept=meta["concept"], title=title, images=meta["images"],
                            camera=meta["camera"], sun=meta["sun"], geometry_seed=meta["geometry_seed"],
                            files={p.name: dict(bytes=p.stat().st_size, sha256=digest(p))
                                   for p in (path, pano, scene, HERE / f"{stem}.json")}))
    d.text((30, 1440), "Visual prototypes for feedback. Rendered geometry and shadows; no navigation experiments changed.",
           font=font(18), fill="#b4c6bf")
    canvas.save(HERE / "comparison.png")
    manifest = dict(renderer="Blender Cycles", source="apiaviz/research/terrain_concepts.py",
                    source_sha256=digest(ROOT / "apiaviz/research/terrain_concepts.py"),
                    comparison_sha256=digest(HERE / "comparison.png"), concepts=exports,
                    validation="All eight PNGs decoded at expected dimensions; shared camera height and sun settings checked; all editable scenes present; file hashes recorded.")
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("Created comparison.png; verified eight rendered PNGs and four editable scenes.")


if __name__ == "__main__":
    main()
