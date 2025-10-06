import bpy

bpy.ops.wm.open_mainfile(filepath="c:/Users/RDM3D/miniDemo/adaptivecad_core/adaptivecad_out/adaptivecad_memory.blend")
obj = bpy.data.objects.get("PiA_Left")
if not obj:
    print("PiA_Left not found")
else:
    z_vals = [v.co.z for v in obj.data.vertices]
    print("min z:", min(z_vals), "max z:", max(z_vals))
    bottoms = [v.co for v in obj.data.vertices if abs(v.co.z - min(z_vals)) < 1e-5]
    print("bottom vertex count:", len(bottoms))
    print("unique radii at bottom sample:", len({(round(co.x,4), round(co.y,4)) for co in bottoms}))
