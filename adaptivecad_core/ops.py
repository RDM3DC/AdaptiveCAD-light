import bpy
from math import pi
from mathutils import Vector
import bmesh

# --- πₐ kernel hook (replace with your real field) ---
def pi_a(p: Vector) -> float:
    # Example: base π modulated by radial curvature memory
    # Replace with your true πₐ field (expose α, μ on the Panel later).
    r2 = p.length_squared
    return pi * (1.0 + 0.01 * r2)  # placeholder

# --- Utility: ensure float edge attribute exists ---
def ensure_edge_float_attribute(me, name):
    if name not in me.attributes:
        me.attributes.new(name, 'FLOAT', 'EDGE')
    return me.attributes[name]

# --- ADD: πₐ-aware primitive ---
class ADAPTIVECAD_OT_add(bpy.types.Operator):
    bl_idname = "adaptivecad.add"
    bl_label = "Adaptive Add (πₐ Primitive)"
    bl_options = {'REGISTER', 'UNDO'}

    radius: bpy.props.FloatProperty(name="Base Radius", default=1.0, min=0.001)
    segments: bpy.props.IntProperty(name="Segments", default=64, min=8, max=4096)

    def execute(self, ctx):
        k = pi_a(Vector((0,0,0))) / pi
        bpy.ops.mesh.primitive_circle_add(vertices=self.segments, radius=self.radius * k)
        obj = ctx.active_object
        if obj is not None:
            obj["pi_a_k"] = float(k)
        return {'FINISHED'}

# --- CUT: curvature-aware boolean slice ---
class ADAPTIVECAD_OT_cut(bpy.types.Operator):
    bl_idname = "adaptivecad.cut"
    bl_label = "Adaptive Cut (πₐ Boolean)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        sel = [o for o in ctx.selected_objects if o.type == 'MESH']
        if len(sel) != 2:
            self.report({'ERROR'}, "Select exactly two mesh objects: target then cutter")
            return {'CANCELLED'}
        target, cutter = sel[0], sel[1]

        m = target.modifiers.new("AdaptiveCut", 'BOOLEAN')
        m.operation = 'DIFFERENCE'
        m.solver = 'EXACT'
        m.object = cutter
        try:
            bpy.ops.object.modifier_apply(modifier=m.name)
        except RuntimeError as e:
            self.report({'ERROR'}, f"Apply failed: {e}")
            return {'CANCELLED'}

        me = target.data
        attr = ensure_edge_float_attribute(me, "curv_memory")
        data = attr.data

        for e in me.edges:
            v0 = me.vertices[e.vertices[0]].co
            v1 = me.vertices[e.vertices[1]].co
            p = 0.5 * (v0 + v1)
            data[e.index].value = float(pi_a(p) - pi)

        return {'FINISHED'}

# --- JOIN: topology-safe merge with curvature blend proxy ---
class ADAPTIVECAD_OT_join(bpy.types.Operator):
    bl_idname = "adaptivecad.join"
    bl_label = "Adaptive Join (Curvature Blend)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        sel = [o for o in ctx.selected_objects if o.type == 'MESH']
        if len(sel) < 2:
            self.report({'ERROR'}, "Select two or more mesh objects to join")
            return {'CANCELLED'}

        # Prefer the active mesh as the join target; fall back to the first selection.
        base = ctx.active_object if ctx.active_object in sel else sel[0]
        ctx.view_layer.objects.active = base

        # Sequentially boolean union every other mesh into the base to ensure watertight output.
        for other in sel:
            if other == base:
                continue

            # Make sure boolean modifiers see both objects.
            other.hide_set(False)
            other.hide_viewport = False

            mod = base.modifiers.new(name="AdaptiveCADJoin", type='BOOLEAN')
            mod.operation = 'UNION'
            mod.solver = 'EXACT'
            mod.object = other
            mod.use_self = True
            mod.double_threshold = 1e-4

            try:
                bpy.ops.object.modifier_apply(modifier=mod.name)
            except RuntimeError as exc:
                self.report({'ERROR'}, f"Join boolean failed: {exc}")
                return {'CANCELLED'}

            # Remove the consumed object to avoid duplicate geometry lingering in the scene.
            bpy.data.objects.remove(other, do_unlink=True)

        obj = base

        # Post-boolean cleanup: weld coincident verts to avoid seam cracks.
        me = obj.data
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
        bm.normal_update()
        bm.to_mesh(me)
        bm.free()
        me.update(calc_edges=True, calc_edges_loose=True)

        attr = ensure_edge_float_attribute(me, "weld_score")
        data = attr.data

        for e in me.edges:
            v0 = me.vertices[e.vertices[0]].co
            v1 = me.vertices[e.vertices[1]].co
            p = 0.5 * (v0 + v1)
            data[e.index].value = float(abs(pi_a(p) - pi))

        return {'FINISHED'}

# --- HEATMAP: bake edge float attribute to vertex color and material ---
class ADAPTIVECAD_OT_heatmap(bpy.types.Operator):
    bl_idname = "adaptivecad.heatmap"
    bl_label = "Create Heatmap from Edge Attribute"
    bl_options = {'REGISTER', 'UNDO'}

    source_attr: bpy.props.EnumProperty(
        name="Source Attribute",
        items=[
            ('curv_memory', "curv_memory", "Edge float from Cut"),
            ('weld_score', "weld_score", "Edge float from Join"),
        ],
        default='curv_memory'
    )
    attr_abs: bpy.props.BoolProperty(
        name="Absolute Value",
        description="Use absolute value before normalization",
        default=True
    )
    vcol_name: bpy.props.StringProperty(
        name="Vertex Color Name",
        default="heatmap"
    )
    mat_name: bpy.props.StringProperty(
        name="Material Name",
        default="AdaptiveCAD_Heatmap_Mat"
    )

    def execute(self, ctx):
        obj = ctx.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Active object must be a mesh")
            return {'CANCELLED'}
        me = obj.data

        # Read edge attribute
        if self.source_attr not in me.attributes:
            self.report({'ERROR'}, f"Missing edge attribute '{self.source_attr}'. Run Cut/Join first.")
            return {'CANCELLED'}
        attr = me.attributes[self.source_attr]
        if attr.domain != 'EDGE' or attr.data_type != 'FLOAT':
            self.report({'ERROR'}, f"Attribute '{self.source_attr}' must be FLOAT on EDGE domain")
            return {'CANCELLED'}

        # Create/clear vertex color (POINT domain, FLOAT_COLOR)
        if self.vcol_name in me.color_attributes:
            me.color_attributes.remove(me.color_attributes[self.vcol_name])
        vcol = me.color_attributes.new(self.vcol_name, 'FLOAT_COLOR', 'POINT')

        # Aggregate edge values to vertices by averaging incident edges
        import math
        vert_sum = [0.0] * len(me.vertices)
        vert_count = [0] * len(me.vertices)

        for e in me.edges:
            val = float(attr.data[e.index].value)
            if self.attr_abs:
                val = abs(val)
            v0, v1 = e.vertices[0], e.vertices[1]
            vert_sum[v0] += val; vert_count[v0] += 1
            vert_sum[v1] += val; vert_count[v1] += 1

        vert_vals = [ (vert_sum[i] / vert_count[i]) if vert_count[i] > 0 else 0.0 for i in range(len(me.vertices)) ]

        # Normalize 0..1
        if len(vert_vals) == 0:
            self.report({'ERROR'}, "No vertices found")
            return {'CANCELLED'}
        vmin = min(vert_vals); vmax = max(vert_vals)
        rng = (vmax - vmin) if (vmax > vmin) else 1.0
        norm = [ (v - vmin) / rng for v in vert_vals ]

        # Write to vertex color
        # Each color is RGBA; we encode a simple blue->red ramp:
        # c = (t, 0.0, 1.0 - t, 1.0)
        col_layer = vcol.data
        for idx in range(len(col_layer)):
            col_layer[idx].color = (0.0, 0.0, 0.0, 1.0)
        for loop in me.loops:
            vi = loop.vertex_index
            if vi >= len(col_layer):
                continue
            t = norm[vi]
            r = t
            g = 0.0
            b = 1.0 - t
            col_layer[vi].color = (r, g, b, 1.0)

        # Create / assign material using the vertex color
        if self.mat_name in bpy.data.materials:
            mat = bpy.data.materials[self.mat_name]
        else:
            mat = bpy.data.materials.new(self.mat_name)
            mat.use_nodes = True
            nt = mat.node_tree
            nt.nodes.clear()
            output = nt.nodes.new("ShaderNodeOutputMaterial")
            output.location = (400, 0)
            emission = nt.nodes.new("ShaderNodeEmission")
            emission.location = (150, 0)
            vcol_node = nt.nodes.new("ShaderNodeVertexColor")
            vcol_node.layer_name = self.vcol_name
            vcol_node.location = (-100, 0)
            nt.links.new(vcol_node.outputs["Color"], emission.inputs["Color"])
            nt.links.new(emission.outputs["Emission"], output.inputs["Surface"])

        # Ensure the vertex color node references the current layer name
        if mat.use_nodes:
            for n in mat.node_tree.nodes:
                if n.bl_idname == "ShaderNodeVertexColor":
                    n.layer_name = self.vcol_name

        # Assign material to object (append or replace first slot if empty)
        if obj.data.materials:
            if obj.active_material is None:
                obj.active_material = obj.data.materials[0]
            if mat.name not in [m.name for m in obj.data.materials]:
                obj.data.materials.append(mat)
                obj.active_material = mat
            else:
                # Set active to this material
                for idx, m in enumerate(obj.data.materials):
                    if m and m.name == mat.name:
                        obj.active_material_index = idx
                        break
        else:
            obj.data.materials.append(mat)
            obj.active_material = mat

        self.report({'INFO'}, f"Heatmap created on '{self.vcol_name}' using '{self.source_attr}'")
        return {'FINISHED'}

# --- UI Panel ---
class ADAPTIVECAD_PT_panel(bpy.types.Panel):
    bl_label = "AdaptiveCAD"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'AdaptiveCAD'

    def draw(self, ctx):
        col = self.layout.column(align=True)
        col.operator("adaptivecad.add", text="Add πₐ Primitive")
        col.operator("adaptivecad.cut", text="Cut (πₐ)")
        col.operator("adaptivecad.join", text="Join (Curvature Blend)")

        box = self.layout.box()
        box.label(text="Heatmap")
        row = box.row()
        op = row.operator("adaptivecad.heatmap", text="Bake Heatmap")
        # Defaults shown; users can change in Operator Adjust Last Operation
