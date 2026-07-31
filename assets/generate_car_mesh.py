"""
3D Vehicle Model Generator for Pre-Hackathon Setup
Author: Karthi (Person B - UE5 & Rendering & Compositing)

Generates a stylized 3D sports coupe mesh in Wavefront OBJ format
with separate material groups (CarPaint, Glass, Wheels, Chrome, Headlights)
for testing in Unreal Engine 5 and Blender Cycles.
"""

import os

def create_car_obj(output_obj_path: str = "assets/car_model.obj", output_mtl_path: str = "assets/car_model.mtl"):
    os.makedirs(os.path.dirname(output_obj_path), exist_ok=True)
    
    mtl_filename = os.path.basename(output_mtl_path)
    
    # 1. Write MTL Material definitions
    mtl_content = """# Car Materials Library
newmtl CarPaint
Kd 0.05 0.15 0.75
Ks 0.95 0.95 0.95
Ns 250
Ni 1.5

newmtl Glass
Kd 0.1 0.1 0.1
Ks 1.0 1.0 1.0
Ns 500
d 0.3
Ni 1.52

newmtl Wheels
Kd 0.05 0.05 0.05
Ks 0.2 0.2 0.2
Ns 50

newmtl Chrome
Kd 0.8 0.8 0.8
Ks 1.0 1.0 1.0
Ns 400

newmtl Headlights
Kd 1.0 0.95 0.8
Ks 1.0 1.0 1.0
Ns 300
"""
    with open(output_mtl_path, "w") as f:
        f.write(mtl_content)
    print(f"[3D Generator] Saved MTL material file: {output_mtl_path}")
    
    # 2. Geometry definition for sports coupe chassis
    # Coordinates centered at origin (0,0,0) resting on Z=0
    # Length ~ 4.2m (Y: -2.1 to 2.1), Width ~ 1.8m (X: -0.9 to 0.9), Height ~ 1.2m (Z: 0.0 to 1.2)
    vertices = [
        # Base chassis bottom
        (-0.9, -2.1, 0.15), (0.9, -2.1, 0.15), (0.9, 2.1, 0.15), (-0.9, 2.1, 0.15),
        # Belt line
        (-0.95, -2.0, 0.65), (0.95, -2.0, 0.65), (0.95, 2.0, 0.65), (-0.95, 2.0, 0.65),
        # Hood / Trunk top
        (-0.85, -1.2, 0.75), (0.85, -1.2, 0.75), (0.85, 1.1, 0.70), (-0.85, 1.1, 0.70),
        # Roof pillars (Cabin)
        (-0.75, -0.5, 1.25), (0.75, -0.5, 1.25), (0.75, 0.6, 1.20), (-0.75, 0.6, 1.20),
        # Windshield top/bottom & Rear window top/bottom
        (-0.80, -0.9, 0.78), (0.80, -0.9, 0.78), (0.80, 0.8, 0.73), (-0.80, 0.8, 0.73)
    ]
    
    # Simple faces indexing (1-based)
    obj_lines = [
        f"mtllib {mtl_filename}",
        "o Vehicle_Body",
        "g CarPaint"
    ]
    
    for v in vertices:
        obj_lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}")
        
    # Normals
    obj_lines.extend([
        "vn 0.0 0.0 1.0", "vn 0.0 0.0 -1.0",
        "vn 1.0 0.0 0.0", "vn -1.0 0.0 0.0",
        "vn 0.0 1.0 0.0", "vn 0.0 -1.0 0.0"
    ])
    
    # Body faces
    obj_lines.extend([
        # Hood
        "usemtl CarPaint",
        "f 5//5 6//5 10//5 9//5",
        "f 9//5 10//5 18//5 17//5",
        # Trunk
        "f 11//5 12//5 8//5 7//5",
        "f 19//5 20//5 12//5 11//5",
        # Sides
        "f 1//4 5//4 8//4 4//4",
        "f 2//3 3//3 7//3 6//3",
        # Cabin Roof
        "f 13//1 14//1 15//1 16//1",
        # Windows / Glass
        "g Glass",
        "usemtl Glass",
        # Windshield
        "f 17//1 18//1 14//1 13//1",
        # Rear Window
        "f 15//1 16//1 20//1 19//1",
        # Side Windows
        "f 13//4 16//4 11//4 9//4",
        "f 14//3 10//3 12//3 15//3"
    ])
    
    with open(output_obj_path, "w") as f:
        f.write("\n".join(obj_lines))
        
    print(f"[3D Generator] Successfully created 3D vehicle OBJ mesh: {output_obj_path}")
    return output_obj_path

if __name__ == "__main__":
    create_car_obj()
