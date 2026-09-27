"""Generate the small rosbag2 fixtures. Run inside a Jazzy container with rosbag2_py: python3 gen_fixtures.py <output dir>"""
import os
import shutil
import sys

import rosbag2_py
from geometry_msgs.msg import TransformStamped
from rclpy.serialization import serialize_message
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

OUT = sys.argv[1]
BASE = 1_700_000_000_000_000_000
STEP = 100_000_000


def chatter(i):
    m = String()
    m.data = f"hello {i}"
    return m


def tf(i, static=False):
    t = TransformStamped()
    t.header.stamp.sec = 1_700_000_000 + (i // 10)
    t.header.stamp.nanosec = (i % 10) * STEP
    t.header.frame_id = "map" if not static else "base_link"
    t.child_frame_id = "base_link" if not static else "laser"
    t.transform.translation.x = 0.1 * i
    t.transform.rotation.w = 1.0
    m = TFMessage()
    m.transforms = [t]
    return m


def scan(i):
    m = LaserScan()
    m.header.stamp.sec = 1_700_000_000 + (i // 10)
    m.header.stamp.nanosec = (i % 10) * STEP
    m.header.frame_id = "laser"
    m.angle_min = -0.5
    m.angle_max = 0.5
    m.angle_increment = 0.125
    m.range_min = 0.1
    m.range_max = 10.0
    m.ranges = [1.0 + 0.5 * k + i for k in range(9)]
    return m


def write(name, storage_id, preset="", max_size=0, config="", count=20):
    uri = os.path.join(OUT, name)
    shutil.rmtree(uri, ignore_errors=True)
    w = rosbag2_py.SequentialWriter()
    so = rosbag2_py.StorageOptions(
        uri=uri,
        storage_id=storage_id,
        max_bagfile_size=max_size,
        storage_preset_profile=preset,
        storage_config_uri=config,
    )
    w.open(so, rosbag2_py.ConverterOptions("", ""))
    for topic, ty in [
        ("/chatter", "std_msgs/msg/String"),
        ("/tf", "tf2_msgs/msg/TFMessage"),
        ("/tf_static", "tf2_msgs/msg/TFMessage"),
        ("/scan", "sensor_msgs/msg/LaserScan"),
    ]:
        w.create_topic(rosbag2_py.TopicMetadata(id=0, name=topic, type=ty, serialization_format="cdr"))
    w.write("/tf_static", serialize_message(tf(0, static=True)), BASE)
    for i in range(count):
        t = BASE + i * STEP
        w.write("/chatter", serialize_message(chatter(i)), t)
        w.write("/tf", serialize_message(tf(i)), t + 1)
        if i % 2 == 0:
            w.write("/scan", serialize_message(scan(i)), t + 2)
    w.close()
    del w
    print(name, sorted(os.listdir(uri)))


write("mini_mcap", "mcap", preset="zstd_fast")
config = os.path.join(OUT, "split_config.yaml")
with open(config, "w") as f:
    f.write("chunkSize: 512\ncompression: Zstd\n")
write("mini_mcap_split", "mcap", max_size=20000, config=config, count=60)
os.remove(config)
write("mini_sqlite3", "sqlite3")
