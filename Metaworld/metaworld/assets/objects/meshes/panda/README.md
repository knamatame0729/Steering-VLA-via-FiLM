# Franka Panda Mesh Files for Metaworld

This directory should contain the Franka Panda robot mesh files.

## Required Files

### Collision meshes (.stl)
- link0.stl
- link1.stl
- link2.stl
- link3.stl
- link4.stl
- link6.stl
- link7.stl
- hand.stl

### Collision meshes (.obj)
- link5_collision_0.obj
- link5_collision_1.obj
- link5_collision_2.obj

### Visual meshes (.obj)
- link0_0.obj to link0_11.obj (excluding link0_6.obj)
- link1.obj
- link2.obj
- link3_0.obj to link3_3.obj
- link4_0.obj to link4_3.obj
- link5_0.obj to link5_2.obj
- link6_0.obj to link6_16.obj
- link7_0.obj to link7_7.obj
- hand_0.obj to hand_4.obj
- finger_0.obj, finger_1.obj

## How to Get These Files

### Option 1: MuJoCo Menagerie (Recommended)
Download from the official MuJoCo Menagerie repository:
```bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git
cp -r mujoco_menagerie/franka_emika_panda/assets/* .
```

### Option 2: Franka Robotics Official
Download from Franka Robotics or use the URDF files from `franka_description` ROS package.

### Option 3: MuJoCo Repository
The mesh files can also be found in the official MuJoCo model library.

## Verification

After copying the mesh files, verify with:
```bash
ls -la *.stl *.obj
```

You should see all the required mesh files listed above.
