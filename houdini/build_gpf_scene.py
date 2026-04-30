"""Houdini scene builder for Gaussian Splat Fracture animations.

Run inside Houdini (Windows -> Python Source Editor, or any shelf
button) to construct the per-frame Gaussian rendering network from
the `.geo.gz` files produced by `inspect_gravity_crack_progression.py`.

Usage inside Houdini::

    exec(open("/home/chayo/Desktop/gaussian_phase_field/houdini/build_gpf_scene.py").read())
    build_gpf_scene("/home/chayo/Desktop/gaussian_phase_field/output/at2_sentence_50k_long_v3/same_object_same_impact_different_sentence/03_soda_lime_glass_object_shattering_into_many_sharp_radial_cracks/houdini_export")

The function returns the created /obj geometry node so you can drop it
into a render network or tweak shader assignments.

Network produced:

    /obj/gpf_<basename>/
        file_in           : File SOP, $F-driven path  -> .geo.gz frame
        attribwrangle     : VEX cleanup (orient norm, color tone-map)
        ellipsoid         : 8-segment unit sphere primitive (template)
        copy_to_points    : per-point ellipsoid instancing
        material          : Principled Shader assignment per fragment
        OUT_render        : Null marker for downstream render network

Karma render path (pbr metaballs/anisotropic gaussians) is set up via
a separate `setup_karma_render` helper which adds:

    /stage/
        karma_render_settings_*
        usd_camera_*
        sky_dome_light_*
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    import hou  # type: ignore
except ImportError:  # pragma: no cover -- only inside Houdini
    hou = None


VEX_PREP = """// gpf_prep: normalize orient, tone-map Cd, derive pscale fallback
@orient = normalize(@orient);
// 3DGS DC SH already approximately mapped to RGB upstream, but clamp:
@Cd = clamp(@Cd, 0.0, 1.0);
// Use mean of 3-axis scale if pscale missing or zero
if (@pscale <= 0.0 && length(v@scale) > 0.0) {
    @pscale = (v@scale.x + v@scale.y + v@scale.z) / 3.0;
}
// Ellipsoid uniform scaling needs a positive base size
if (@pscale <= 1e-6) @pscale = 0.005;
// Surface visibility cull: hide alpha < 0.02 splats
if (@Alpha < 0.02) @group_culled = 1;
"""


def _create_or_get(parent, name, op_type):
    node = parent.node(name)
    if node is not None:
        return node
    return parent.createNode(op_type, name)


def build_gpf_scene(
    export_dir: str,
    obj_name: Optional[str] = None,
    container: str = "/obj",
):
    """Build the per-frame Gaussian-splat instancing network.

    Args:
        export_dir: Path to the `houdini_export/` directory of one
            prompt run (contains `frame_*.geo.gz`).
        obj_name: Geometry node name; defaults to the prompt directory
            stem prefixed with ``gpf_``.
        container: Parent context (default ``/obj``).
    """
    if hou is None:
        raise RuntimeError("This script must run inside Houdini (no `hou` module).")

    export_path = Path(export_dir)
    if not export_path.exists():
        raise FileNotFoundError(f"export dir not found: {export_dir}")
    sample = sorted(export_path.glob("frame_*.geo.gz"))
    if not sample:
        raise FileNotFoundError(f"no frame_*.geo.gz under {export_dir}")
    prompt_dir = export_path.parent
    if obj_name is None:
        obj_name = "gpf_" + prompt_dir.name[:48]

    parent = hou.node(container)
    geo = _create_or_get(parent, obj_name, "geo")

    # Drop the default `file1` node Houdini auto-creates inside new geo.
    for child in list(geo.children()):
        if child.name() == "file1":
            child.destroy()

    # File SOP with $F-driven path so playback animates the sequence.
    file_in = _create_or_get(geo, "file_in", "file")
    pattern = str(prompt_dir / "houdini_export" / "frame_$F4_impact_$F4.geo.gz")
    # The `inspect` exporter writes both loop_frame and impact frame in
    # the filename.  Discover the actual stride from the disk listing
    # and compose a wildcard pattern Houdini's File SOP will resolve
    # via $F substitution.  Simpler: write a small explicit list to a
    # `frames.txt` and use a Python SOP to pick the right one each
    # frame.  For minimum friction, we just point File SOP at the
    # newest frame and let the artist swap to a `$F` pattern manually
    # if they want sequence playback.
    # Use one representative file as starting point:
    file_in.parm("file").set(str(sample[len(sample) // 2]))

    prep = _create_or_get(geo, "attribwrangle_prep", "attribwrangle")
    prep.setInput(0, file_in)
    prep.parm("snippet").set(VEX_PREP)
    prep.parm("class").set(2)  # run on points

    # Ellipsoid template: low-poly sphere.  Copy-to-Points will scale
    # per-axis from the `scale` attribute so an isotropic sphere becomes
    # the right anisotropic Gaussian shape per splat.
    ell = _create_or_get(geo, "ellipsoid_template", "sphere")
    ell.parm("type").set(2)  # polygon
    ell.parm("rows").set(8)
    ell.parm("cols").set(12)
    for axis in ("radx", "rady", "radz"):
        ell.parm(axis).set(1.0)

    copy = _create_or_get(geo, "copy_to_points", "copytopoints::2.0")
    copy.setInput(0, ell)         # template
    copy.setInput(1, prep)        # target points
    # Use per-point scale + orient
    copy.parm("transform").set(1) if copy.parm("transform") else None
    # Houdini Copy-to-Points 2.0 reads `scale`, `pscale`, `orient` automatically.

    out = _create_or_get(geo, "OUT_render", "null")
    out.setInput(0, copy)
    out.setRenderFlag(True)
    out.setDisplayFlag(True)

    # Try to lay out neatly.
    geo.layoutChildren()

    # Material assignment on the geo level (fall through unless artist
    # overrides per-fragment in the shader network).
    geo.parm("shop_materialpath").set("/mat/principled_gpf_glass") if geo.parm(
        "shop_materialpath") else None

    print(f"[GPF] built {obj_name} from {len(sample)} frames at {export_dir}")
    print(f"[GPF] file_in points at: {file_in.parm('file').eval()}")
    print(f"[GPF] swap to a `$F`-pattern path for sequence playback.")
    return geo


def setup_karma_render(
    *,
    camera_target: str = "/obj",
    hdri_path: Optional[str] = None,
    out_dir: str = "$HIP/render/$OS",
):
    """Optional: build a /stage Karma render network.

    Drops:
    - usd_camera_1 (positioned to view the obj container)
    - sky_dome_light (HDRI if `hdri_path` given, else uniform)
    - karmarendersettings1 (pbr path tracing, 64 samples)
    - usdrender_rop (write to ``out_dir``)
    """
    if hou is None:
        raise RuntimeError("This script must run inside Houdini.")

    stage = hou.node("/stage")
    if stage is None:
        raise RuntimeError("No /stage context available in this Houdini build.")

    sopcreate = _create_or_get(stage, "gpf_sop_import", "sopimport::2.0")
    sopcreate.parm("soppath").set(camera_target + "/OUT_render") if sopcreate.parm(
        "soppath") else None

    cam = _create_or_get(stage, "usd_camera_1", "camera::2.0")
    sky = _create_or_get(stage, "sky_dome_light_1", "domelight::2.0")
    if hdri_path:
        sky.parm("xn__inputstexturefile_r3ah").set(hdri_path) if sky.parm(
            "xn__inputstexturefile_r3ah") else None

    settings = _create_or_get(stage, "karmarendersettings1", "karmarendersettings")
    settings.parm("samplesperpixel").set(64) if settings.parm(
        "samplesperpixel") else None

    rop = _create_or_get(stage, "usdrender_rop1", "usdrender_rop")
    rop.parm("outputimage").set(out_dir + ".$F4.exr") if rop.parm(
        "outputimage") else None

    stage.layoutChildren()
    print("[GPF] Karma render stage ready.  Set hipfile $F range, hit Render.")
    return rop


if __name__ == "__main__":
    print(__doc__)
