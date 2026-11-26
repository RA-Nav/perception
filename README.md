# perception

The perception of the RaNAV quadruped, in one repository with three ROS 2 packages. In the workspace it is the
submodule `src/perception`.

| package | what it does |
| --- | --- |
| [lidar_perception](lidar_perception/README.md) | People detection and tracking on the lidar scans of FAST-LIO: keeps the people out of the SC-PGO map and of the TARE coverage, and hands them to the Opti-Pessi controller as obstacles |
| [camera_perception](camera_perception/README.md) | YOLO26 on the RGB-D camera (YOLOs-CPP, ONNX Runtime): persons and forklifts as obstacles, and the exit sign detector of `exit_inspection` |
| [exit_inspection](exit_inspection/README.md) | Emergency exits: exit sign landmarks, verdicts of the zone in front of every door, the inspection mission |

`camera_perception` needs YOLOs-CPP and ONNX Runtime in `third_party/` of the workspace
(`./setup_ranav.sh sources --camera`). The other two packages build without them. `exit_inspection` runs the detector
of `camera_perception`, so colcon has to be told to leave it out:

```bash
colcon build --packages-select lidar_perception exit_inspection --packages-ignore camera_perception
```

## History

The packages come from three repositories, merged with their history:

| package | was |
| --- | --- |
| `lidar_perception` | package `people_perception`, this repository (renamed from `RA-Nav/people_perception`) |
| `camera_perception` | `RA-Nav/opti_pessi_perception`, a submodule of `legged_control` |
| `exit_inspection` | `RA-Nav/exit_inspection` |
