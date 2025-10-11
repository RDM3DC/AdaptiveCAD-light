# HF Deployment Package for π-adaptive Möbius

This folder contains the production-ready files for deploying the π-adaptive Möbius generator to Hugging Face Spaces.

## Files

- **`main.py`**: Updated FastAPI app with `adaptive_mobius` shape type integration
- **`adaptive_mobius_unity.py`**: Core π-adaptive geometry generator (copied from workspace root)
- **`requirements.txt`**: Dependencies including numpy==1.26.4
- **`app.py`**: Entry point (copy from existing HF Space)
- **`Dockerfile`**: Container config (copy from existing HF Space)

## Deployment Steps

1. **Copy existing HF files** that don't need changes:
   ```powershell
   # Assuming you have HF Space cloned or access to these files
   # Copy app.py and Dockerfile from the current HF deployment
   ```

2. **Push to Hugging Face Space**:
   ```powershell
   git checkout -b deploy-pi-adaptive
   git add hf_deployment/
   git commit -m "Add π-adaptive Möbius generation with kappa parameter"
   
   # Replace the HF Space files (in separate HF repo/Space):
   # - Copy hf_deployment/main.py → adaptivecad_lite_api/main.py
   # - Copy hf_deployment/adaptive_mobius_unity.py → adaptive_mobius_unity.py
   # - Copy hf_deployment/requirements.txt → requirements.txt
   
   # Then push to HF Space repository:
   git push hf deploy-pi-adaptive:main
   ```

3. **Test the Endpoint**:
   ```bash
   curl -X POST "https://rdm3dc-adaptivecad-lite.hf.space/generate_shape" \
     -H "Content-Type: application/json" \
     -d '{
       "type": "adaptive_mobius",
       "params": {
         "radius_mm": 40,
         "kappa": 0.15,
         "tau": 0.75,
         "proj_mode": "hybrid"
       },
       "output": "obj"
     }'
   ```

## Key Changes

### `main.py`
- Added `adaptive_mobius` to `AVAILABLE_SHAPES` catalog with description
- Implemented real generation logic in `/generate_shape` endpoint:
  - Imports `adaptive_mobius_unity`, `save_obj`, `write_binary_stl`
  - Passes all params with sensible defaults
  - Exports to temp file and returns metadata
- Returns vertex/face counts and file path

### `requirements.txt`
- Added `numpy==1.26.4` (required for mesh generation)

### `adaptive_mobius_unity.py`
- No changes needed; copied as-is from workspace root
- Contains full π-adaptive implementation with kappa parameter

## Parameters

When calling `/generate_shape` with `type: "adaptive_mobius"`, you can override:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `radius_mm` | 40.0 | Base radius of Möbius band |
| `half_width_mm` | 8.0 | Half-width of the band |
| `twists` | 1.5 | Number of half-twists |
| `gamma` | 0.25 | 4D projection strength |
| `tau` | 0.5 | Time-like morphing factor |
| `proj_mode` | "hybrid" | Projection: euclidean, lorentz, complex, hybrid |
| `thickness_mm` | 2.0 | Band thickness (manifold) |
| `kappa` | 0.0 | π-adaptive curvature warping |
| `samples_major` | 480 | Tessellation around band |
| `samples_width` | 48 | Tessellation across width |

## Output Formats

- `"obj"`: Wavefront OBJ (text-based, includes normals)
- `"stl"`: Binary STL (compact binary mesh)

## Notes

- The HF Space will need to rebuild to install numpy
- For production, replace temp file handling with blob storage (S3, GCS, etc.)
- Current implementation returns local file path; update with CDN URLs for production
- Ensure `app.py` imports from `adaptivecad_lite_api.main` as currently configured in HF Space
