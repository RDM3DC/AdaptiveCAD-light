"""Render a 200-frame turntable MP4 for the active object."""

import math
import os

import bpy


def run() -> None:
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "adaptivecad_core", "adaptivecad_out"))
    os.makedirs(out_dir, exist_ok=True)

    obj = bpy.context.active_object
    if obj is None:
        raise RuntimeError("Select an object to render before running render_turntable.py")

    cam_data = bpy.data.cameras.new("turn_cam")
    cam = bpy.data.objects.new("turn_cam", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    distance = max(obj.dimensions) * 2.5
    cam.location = (distance, 0.0, distance * 0.6)
    cam.rotation_euler = (math.radians(35.0), 0.0, math.radians(135.0))

    target = bpy.data.objects.new("EmptyTarget", None)
    bpy.context.collection.objects.link(target)
    target.location = obj.location

    constraint = cam.constraints.new(type='TRACK_TO')
    constraint.target = target
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'

    frame_start, frame_end = 1, 200
    scene = bpy.context.scene
    scene.frame_start = frame_start
    scene.frame_end = frame_end

    for frame in (frame_start, frame_end):
        angle = (frame - frame_start) / (frame_end - frame_start) * 2.0 * math.pi
        cam.location.x = math.cos(angle) * distance
        cam.location.y = math.sin(angle) * distance
        cam.keyframe_insert(data_path="location", frame=frame)

    engine_items = {
        item.identifier
        for item in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items
    }
    scene.render.engine = 'BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in engine_items else 'BLENDER_EEVEE'
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
    scene.render.ffmpeg.ffmpeg_preset = 'GOOD'
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.fps = 30

    scene.render.filepath = os.path.join(out_dir, "turntable.mp4")
    bpy.ops.render.render(animation=True)
    print("Turntable saved to", scene.render.filepath)


if __name__ == "__main__":
    run()
