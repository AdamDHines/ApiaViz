"""Decode every video frame and check media metadata and recorded provenance."""
import json
from fractions import Fraction
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from apiaviz.research.study import file_hash, write_json

HERE = Path(__file__).parent


def main():
    report = dict(videos=[], navigation_decisions=0, heading_scores=0, decoded_frames=0, errors=[])
    for ant in (4, 7, 13):
        stem = f"ant-{ant:02d}-linear-colour"
        meta = json.loads((HERE / f"{stem}.json").read_text())
        video = HERE / f"{stem}.mp4"
        assert file_hash(video) == meta["video_sha256"]
        provenance = meta["provenance"]
        run = ROOT / provenance["source_run"]
        manifest = json.loads((run / "manifest.json").read_text())
        assert manifest["status"] == "complete"
        matches = [r for r in map(json.loads, (run / "results.jsonl").read_text().splitlines())
                   if (r["ant"], r["seed"], r["preprocessing"]) == (ant, 19, "linear_colour")]
        assert len(matches) == 1 and matches[0] == meta["result"]
        assert file_hash(run / matches[0]["trace"]) == provenance["trace_sha256"]
        for key, filename in (("video_source_sha256", "navigation_video.py"),
                              ("encoder_source_sha256", "frontend_refinements.py"),
                              ("renderer_source_sha256", "fast_render.py")):
            assert file_hash(ROOT / "apiaviz/research" / filename) == provenance[key]
        source = ROOT / manifest["stimulus_sources"][str(ant)]["path"]
        assert file_hash(source / f"ant-{ant}-images.pt") == provenance["bank_sha256"]
        assert file_hash(ROOT / manifest["encoders"]["19"]["source"]) == provenance["checkpoint_sha256"]
        assert meta["score_max_absolute_error"] == 0
        assert meta["verified_scans"] == matches[0]["steps"]
        assert meta["verified_heading_scores"] == 13*matches[0]["steps"]
        command = ["ffprobe", "-v", "error", "-count_frames", "-show_entries",
                   "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_read_frames,duration", "-of", "json", str(video)]
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        assert not process.stderr.strip(), process.stderr
        streams = json.loads(process.stdout)["streams"]
        assert len(streams) == 1
        s = streams[0]
        assert (s["codec_name"], s["width"], s["height"], s["pix_fmt"]) == ("h264", 1920, 1080, "yuv420p")
        assert Fraction(s["r_frame_rate"]) == meta["fps"] == 30
        assert int(s["nb_read_frames"]) == meta["frames"]
        assert abs(float(s["duration"]) - meta["duration_s"]) < .001
        assert meta["frames"] == 30*(meta["verified_scans"]+15)
        item = dict(ant=ant, file=video.name, bytes=video.stat().st_size,
                    seconds=meta["duration_s"], frames=meta["frames"],
                    exact_heading_scores=meta["verified_heading_scores"], fully_decoded=True)
        report["videos"].append(item)
        report["navigation_decisions"] += meta["verified_scans"]
        report["heading_scores"] += meta["verified_heading_scores"]
        report["decoded_frames"] += meta["frames"]
        print(json.dumps(item), flush=True)
    assert report["navigation_decisions"] == 410 and report["heading_scores"] == 5330
    report["scope"] = "Every encoded frame decoded without ffprobe errors; H.264/fps/dimensions/durations checked; video/source/input/checkpoint hashes verified; saved result rows matched. Exact scan recomputation and trajectory checks were performed by the video builder."
    write_json(HERE / "audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
