
bl_info = {
    "name": "AdaptiveCAD: Revolve (πₐ-aware sampling)",
    "author": "RDM3DC LLC",
    "version": (0, 2, 0),
    "blender": (3, 4, 0),
    "location": "3D Viewport > N-Panel > AdaptiveCAD",
    "description": "Revolve active Curve with adaptive θ based on chord error & curvature proxy",
    "category": "Add Curve",
}

import bpy, math
import numpy as np
from mathutils import Vector
from mathutils.geometry import interpolate_bezier

# ---------------- revolve core ----------------
EPS = 1e-9
def _unit(v): 
    v = np.asarray(v, float)
    n=np.linalg.norm(v); 
    return v/(n+EPS)

def _project_point_to_axis(p,a0,ad):
    ap=p-a0; t=float(np.dot(ap,ad)); foot=a0+t*ad; radial=p-foot; return foot, radial

def _rotate_about_axis(v,a_dir,theta):
    k=_unit(a_dir); c,s=math.cos(theta),math.sin(theta)
    return v*c + np.cross(k,v)*s + k*np.dot(k,v)*(1.0-c)

def _discrete_curve_curvature(P):
    n=len(P)
    if n<3: return np.zeros(n, float)
    kappa=np.zeros(n,float)
    for i in range(1,n-1):
        v1=P[i]-P[i-1]; v2=P[i+1]-P[i]
        l1=max(np.linalg.norm(v1), EPS); l2=max(np.linalg.norm(v2), EPS)
        cosang=np.clip(np.dot(v1,v2)/(l1*l2), -1.0, 1.0)
        ang=math.acos(cosang); ds=0.5*(l1+l2); kappa[i]=ang/max(ds,EPS)
    kappa[0]=kappa[1]; kappa[-1]=kappa[-2]
    return kappa

def _theta_segments(profile, a0, ad, angle, tol, eta_k=0.4):
    P=profile; kappa=_discrete_curve_curvature(P); thetas=[]
    for i,p in enumerate(P):
        _,rad=_project_point_to_axis(p,a0,ad); r=float(np.linalg.norm(rad))
        if r<1e-6: continue
        val=1.0-(tol/max(r,tol)); val=float(np.clip(val,-1.0,1.0))
        dtheta=2.0*math.acos(val); dtheta=dtheta/max(1.0+eta_k*kappa[i],1e-3)
        thetas.append(max(dtheta, math.radians(0.5)))
    if not thetas: return 32
    return int(np.clip(math.ceil(angle/min(thetas)), 16, 720))

def revolve_adaptive(profile_pts, axis_point, axis_dir, angle=math.radians(360.0), theta0=0.0, tol=1e-3):
    P=np.asarray(profile_pts,float); a0=np.asarray(axis_point,float); ad=_unit(axis_dir)
    R=_theta_segments(P,a0,ad,angle,tol)
    verts=[]; rings=[]
    for p in P:
        foot,rad=_project_point_to_axis(p,a0,ad); h=float(np.dot(p-a0,ad)); ring=[]
        for j in range(R+1):
            th=theta0+angle*(j/R); pos=_rotate_about_axis(rad,ad,th)+(a0+h*ad)
            ring.append(len(verts)); verts.append(pos)
        rings.append(ring)
    faces=[]
    for i in range(len(P)-1):
        r0,r1=rings[i],rings[i+1]
        for j in range(R):
            a,b=r0[j],r0[j+1]; c=r1[j+1]; d=r1[j]
            faces.append([a,b,c]); faces.append([a,c,d])
    seam=[rings[i][0] for i in range(len(P))]
    return np.array(verts), np.array(faces,dtype=np.int32), np.array(seam,dtype=np.int32)

# ---------------- curve sampling ----------------
def sample_curve_object(obj, samples_per_segment=32):
    assert obj.type == 'CURVE'
    M = obj.matrix_world
    pts = []
    for sp in obj.data.splines:
        if sp.type == 'BEZIER':
            bz = sp.bezier_points
            segs = len(bz) if sp.use_cyclic_u else len(bz) - 1
            for i in range(segs):
                p0 = M @ bz[i].co
                h1 = M @ bz[i].handle_right
                if i == len(bz) - 1:  # cyclic wrap
                    p3 = M @ bz[0].co
                    h2 = M @ bz[0].handle_left
                else:
                    p3 = M @ bz[i+1].co
                    h2 = M @ bz[i+1].handle_left
                seg = interpolate_bezier(p0, h1, h2, p3, samples_per_segment+1)
                if i < segs - 1 or sp.use_cyclic_u:
                    seg = seg[:-1]  # avoid duplicates
                pts.extend([np.array([v.x, v.y, v.z], float) for v in seg])
        elif sp.type in {'POLY', 'NURBS'}:
            for p in sp.points:
                v = M @ Vector((p.co.x, p.co.y, p.co.z))
                pts.append(np.array([v.x, v.y, v.z], float))
        else:
            continue
    out = [pts[0]] if pts else []
    for q in pts[1:]:
        if np.linalg.norm(q - out[-1]) > 1e-7:
            out.append(q)
    return np.array(out, float)

# ---------------- operator & panel ----------------
class ADAPTIVE_OT_revolve_curve(bpy.types.Operator):
    bl_idname = "adaptivecad.revolve_curve"
    bl_label = "Adaptive Revolve"
    bl_description = "Revolve active Curve with adaptive θ (πₐ-like)"
    bl_options = {'REGISTER', 'UNDO'}

    axis: bpy.props.EnumProperty(
        name="Axis",
        items=[('X','X',''),('Y','Y',''),('Z','Z','')],
        default='Z'
    )
    angle: bpy.props.FloatProperty(name="Angle", default=360.0, min=1.0, max=360.0, subtype='ANGLE')
    theta0: bpy.props.FloatProperty(name="Seam Offset", default=0.0, min=0.0, max=360.0, subtype='ANGLE')
    tol: bpy.props.FloatProperty(name="Chord Tol", default=0.001, min=1e-6, soft_max=0.01, precision=6)
    samples: bpy.props.IntProperty(name="Samples/Seg", default=24, min=4, max=256)

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'CURVE':
            self.report({'ERROR'}, "Select a Curve object first")
            return {'CANCELLED'}

        profile = sample_curve_object(obj, self.samples)
        if len(profile) < 2:
            self.report({'ERROR'}, "Profile too short")
            return {'CANCELLED'}

        a_dir = {'X': np.array([1,0,0],float),
                 'Y': np.array([0,1,0],float),
                 'Z': np.array([0,0,1],float)}[self.axis]
        a_pt = np.array([0.0,0.0,0.0], float)

        V,F,seam = revolve_adaptive(
            profile_pts=profile,
            axis_point=a_pt,
            axis_dir=a_dir,
            angle=math.radians(self.angle),
            theta0=math.radians(self.theta0),
            tol=self.tol
        )

        # Build mesh
        me = bpy.data.meshes.new("AdaptiveRevolve")
        me.from_pydata(V.tolist(), [], F.tolist())
        me.validate(verbose=False)
        me.update()
        if hasattr(me, "use_auto_smooth"):
            me.use_auto_smooth = True

        ob = bpy.data.objects.new("AdaptiveRevolve", me)
        context.collection.objects.link(ob)
        ob.select_set(True); context.view_layer.objects.active = ob

        # Add seam memory as a point attribute (1.0 on seam)
        try:
            attr = me.attributes.new("AdaptiveMemory", 'FLOAT', 'POINT')
            data = attr.data
            for i in range(len(me.vertices)): data[i].value = 0.0
            for vid in seam:
                if 0 <= int(vid) < len(me.vertices):
                    data[int(vid)].value = 1.0
        except Exception:
            # Fallback: vertex group
            vg = ob.vertex_groups.new(name="AdaptiveMemory")
            vg.add(list(range(len(me.vertices))), 0.0, 'REPLACE')
            vg.add([int(i) for i in seam if 0 <= int(i) < len(me.vertices)], 1.0, 'REPLACE')

        self.report({'INFO'}, f"Revolved {len(profile)} pts → {len(V)} verts / {len(F)} tris")
        return {'FINISHED'}

class ADAPTIVECAD_PT_panel(bpy.types.Panel):
    bl_label = "AdaptiveCAD"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'AdaptiveCAD'
    def draw(self, ctx):
        col = self.layout.column(align=True)
        col.label(text="Revolve (πₐ‑aware θ)")
        col.operator(ADAPTIVE_OT_revolve_curve.bl_idname, text="Revolve Active Curve")

def register():
    bpy.utils.register_class(ADAPTIVE_OT_revolve_curve)
    bpy.utils.register_class(ADAPTIVECAD_PT_panel)

def unregister():
    bpy.utils.unregister_class(ADAPTIVECAD_PT_panel)
    bpy.utils.unregister_class(ADAPTIVE_OT_revolve_curve)

if __name__ == "__main__":
    register()
