import bpy
import bmesh
from math import pi, sqrt, exp
from mathutils import Vector

# ---------- πₐ controls on the Scene ----------
class ADAPTIVECAD_PiAProps(bpy.types.PropertyGroup):
    alpha: bpy.props.FloatProperty(name="α (reinforce)", default=0.02, min=0.0, soft_max=1.0)
    mu:    bpy.props.FloatProperty(name="μ (decay)",     default=0.005, min=0.0, soft_max=1.0)
    scale: bpy.props.FloatProperty(name="Field Scale",    default=1.0, min=1e-4, soft_max=10.0)
    beta:  bpy.props.FloatProperty(name="Curv Gain β",    default=0.25, min=0.0, soft_max=2.0,
                                   description="Amplitude that maps curvature energy to π deviation")
    anisX: bpy.props.FloatProperty(name="Anisotropy X",   default=1.0, min=0.1, soft_max=5.0)
    anisY: bpy.props.FloatProperty(name="Anisotropy Y",   default=1.0, min=0.1, soft_max=5.0)
    anisZ: bpy.props.FloatProperty(name="Anisotropy Z",   default=1.0, min=0.1, soft_max=5.0)
    center: bpy.props.FloatVectorProperty(name="Center",  default=(0.0,0.0,0.0), subtype='XYZ')
    regime_thresh: bpy.props.FloatProperty(
        name="Near/Far Threshold",
        default=0.15,
        min=0.0, soft_max=2.0,
        description="If |πₐ-π| < thresh → near regime (gradient term); else far regime (relax only)"
    )

# persistent curvature memory storage
def ensure_memory_attribute(me):
    """Ensure a per-vertex float attribute 'pia_memory' exists."""
    if "pia_memory" not in me.attributes:
        me.attributes.new("pia_memory", 'FLOAT', 'POINT')
    return me.attributes["pia_memory"]


# ---------- πₐ kernel ----------
def pi_a(p: Vector) -> float:
    """Anisotropic πₐ field blended with stored curvature memory."""
    scn = bpy.context.scene
    params = getattr(scn, "pia", None)
    if params is None:
        return pi

    obj = bpy.context.active_object
    me = obj.data if obj and obj.type == 'MESH' else None

    # Anisotropic radius relative to the scene-defined center
    q = Vector((p.x - params.center[0], p.y - params.center[1], p.z - params.center[2]))
    ax, ay, az = max(1e-6, params.anisX), max(1e-6, params.anisY), max(1e-6, params.anisZ)
    r_aniso = sqrt((q.x/ax)**2 + (q.y/ay)**2 + (q.z/az)**2)

    # Curvature kernel baseline
    s = max(1e-6, params.scale)
    K = (r_aniso ** 2) * exp(-r_aniso / s)

    # Pull memory contribution from nearest stored vertex value if available
    memory_val = 0.0
    if me and "pia_memory" in me.attributes:
        mem_attr = me.attributes["pia_memory"]
        if mem_attr.domain == 'POINT' and len(mem_attr.data) == len(me.vertices) and len(me.vertices):
            verts = me.vertices
            mem_data = mem_attr.data
            nearest = min(range(len(verts)), key=lambda i: (verts[i].co - p).length_squared)
            memory_val = mem_data[nearest].value

    K_total = K + memory_val

    beta = params.beta
    delta_raw = beta * K_total

    mu = params.mu
    alpha = params.alpha
    thr = params.regime_thresh

    x = delta_raw
    w_near = 1.0 if abs(x) < thr else thr / (abs(x) + 1e-9)
    w_near = max(0.0, min(1.0, w_near))

    x_star = (1.0 - mu) * x - w_near * alpha * (x - delta_raw)
    return pi + x_star

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
        if obj:
            obj["pi_a_k"] = float(k)
            # Make it solid: fill the disc and add a thin extrusion for boolean stability
            prev_mode = obj.mode
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.fill()
            bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value": (0.0, 0.0, 0.1)})
            bpy.ops.object.mode_set(mode=prev_mode if prev_mode in {'EDIT', 'OBJECT'} else 'OBJECT')
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

        bm = bmesh.new()
        bm.from_mesh(me)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        if bm.verts:
            min_z = min(v.co.z for v in bm.verts)
            tol = 1e-5
            bottom_faces = [f for f in bm.faces if all(abs(v.co.z - min_z) <= tol for v in f.verts)]
            if bottom_faces:
                bmesh.ops.delete(bm, geom=bottom_faces, context='FACES')
                bm.faces.ensure_lookup_table()
                bm.edges.ensure_lookup_table()
                boundary_edges = [
                    e for e in bm.edges
                    if e.is_boundary and all(abs(v.co.z - min_z) <= tol for v in e.verts)
                ]
                if boundary_edges:
                    bmesh.ops.edgenet_fill(bm, edges=boundary_edges)
                    bm.faces.ensure_lookup_table()

        bmesh.ops.dissolve_limit(bm, angle_limit=0.01, verts=bm.verts, edges=bm.edges)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
        bm.to_mesh(me)
        bm.free()
        me.update()

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

    @staticmethod
    def _set_active(ctx, obj):
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        ctx.view_layer.objects.active = obj

    def execute(self, ctx):
        sel = [o for o in ctx.selected_objects if o.type == 'MESH']
        if len(sel) < 2:
            self.report({'ERROR'}, "Select two or more mesh objects to join")
            return {'CANCELLED'}

        active = ctx.active_object if ctx.active_object in sel else sel[0]
        self._set_active(ctx, active)
        target = ctx.active_object

        others = [o for o in sel if o != target]
        for other in others:
            mod = target.modifiers.new(name="AdaptiveJoinUnion", type='BOOLEAN')
            mod.operation = 'UNION'
            mod.solver = 'EXACT'
            mod.object = other
            try:
                bpy.ops.object.modifier_apply(modifier=mod.name)
            except RuntimeError as exc:
                self.report({'ERROR'}, f"Boolean union failed on {other.name}: {exc}")
                return {'CANCELLED'}
            bpy.data.objects.remove(other, do_unlink=True)
            self._set_active(ctx, target)

        me = target.data
        me.update()

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
            ('pia_memory', "pia_memory", "Vertex memory accumulated by Evolve"),
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

        # Read chosen attribute (supports EDGE floats and POINT floats)
        if self.source_attr not in me.attributes:
            self.report({'ERROR'}, f"Missing attribute '{self.source_attr}'. Run Cut/Join/Evolve first.")
            return {'CANCELLED'}
        attr = me.attributes[self.source_attr]
        if attr.data_type != 'FLOAT':
            self.report({'ERROR'}, f"Attribute '{self.source_attr}' must be FLOAT")
            return {'CANCELLED'}

        # Create/clear vertex color (POINT domain, FLOAT_COLOR)
        if self.vcol_name in me.color_attributes:
            me.color_attributes.remove(me.color_attributes[self.vcol_name])
        vcol = me.color_attributes.new(self.vcol_name, 'FLOAT_COLOR', 'POINT')

        vert_vals = [0.0] * len(me.vertices)

        if attr.domain == 'EDGE':
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
        elif attr.domain == 'POINT':
            data_len = len(attr.data)
            for idx in range(len(me.vertices)):
                val = float(attr.data[idx].value) if idx < data_len else 0.0
                vert_vals[idx] = abs(val) if self.attr_abs else val
        else:
            self.report({'ERROR'}, f"Attribute '{self.source_attr}' domain '{attr.domain}' not supported")
            return {'CANCELLED'}

        if attr.domain == 'EDGE' and self.attr_abs is False:
            vert_vals = [float(v) for v in vert_vals]

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
        for elem in col_layer:
            elem.color = (0.0, 0.0, 0.0, 1.0)
        for idx, loop in enumerate(me.loops):
            vi = loop.vertex_index
            if idx >= len(col_layer) or vi >= len(norm):
                continue
            t = norm[vi]
            r = t
            g = 0.0
            b = 1.0 - t
            col_layer[idx].color = (r, g, b, 1.0)

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


# --- EVOLVE MEMORY ----------------------------------------------------------
class ADAPTIVECAD_OT_evolve_memory(bpy.types.Operator):
    """Update vertex memory M ← (1-μ)M + α·K(p) and feed it back into πₐ."""
    bl_idname = "adaptivecad.evolve_memory"
    bl_label = "Evolve Memory (πₐ↻M)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        scn = ctx.scene
        params = getattr(scn, "pia", None)
        if params is None:
            self.report({'ERROR'}, "Scene πₐ properties missing")
            return {'CANCELLED'}

        obj = ctx.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Active object must be a mesh")
            return {'CANCELLED'}

        me = obj.data
        mem_attr = ensure_memory_attribute(me)
        mem_data = mem_attr.data
        vert_len = len(me.vertices)
        if vert_len == 0:
            self.report({'ERROR'}, "Mesh has no vertices")
            return {'CANCELLED'}

        alpha = params.alpha
        mu = params.mu
        s = max(1e-6, params.scale)
        cx, cy, cz = params.center
        ax = max(1e-6, params.anisX)
        ay = max(1e-6, params.anisY)
        az = max(1e-6, params.anisZ)

        for i, vert in enumerate(me.vertices):
            dx = (vert.co.x - cx) / ax
            dy = (vert.co.y - cy) / ay
            dz = (vert.co.z - cz) / az
            r = sqrt(dx * dx + dy * dy + dz * dz)
            K = (r ** 2) * exp(-r / s)
            M_old = mem_data[i].value
            M_new = (1.0 - mu) * M_old + alpha * K
            mem_data[i].value = M_new

        self.report({'INFO'}, f"Memory evolved for {vert_len} vertices")
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
        self.layout.operator("adaptivecad.evolve_memory", text="Evolve Memory (πₐ↻M)")

        box = self.layout.box()
        box.label(text="πₐ Kernel")
        c = box.column(align=True)
        p = getattr(ctx.scene, "pia", None)
        if p is not None:
            c.prop(p, "alpha"); c.prop(p, "mu"); c.prop(p, "beta")
            c.prop(p, "scale"); c.prop(p, "regime_thresh")
            row = box.row(align=True); row.prop(p, "anisX"); row.prop(p, "anisY"); row.prop(p, "anisZ")
            c.prop(p, "center")
        else:
            c.label(text="Scene πₐ props unavailable", icon='INFO')

        box = self.layout.box()
        box.label(text="Heatmap")
        row = box.row()
        op = row.operator("adaptivecad.heatmap", text="Bake Heatmap")
        # Defaults shown; users can change in Operator Adjust Last Operation
