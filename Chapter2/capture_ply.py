import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

for _ in range(90):
    pipeline.wait_for_frames()
frames = pipeline.wait_for_frames()
depth = frames.get_depth_frame()
color = frames.get_color_frame()

if not depth or not color:
    raise RuntimeError("Depth 또는 Color 프레임을 가져오지 못했습니다.")

pc = rs.pointcloud()
pc.map_to(color)                # 색 정보 매핑
points = pc.calculate(depth)    # 포인트클라우드 계산

points.export_to_ply("output.ply", color)
print("Saved to output.ply")

pipeline.stop()

