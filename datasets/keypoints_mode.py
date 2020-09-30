'''The stickman configuration for OpenPose's keypoint output with 18
   joints.
   See the corresponding
   [page](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#pose-output-format-coco).

   Numbers shown below correspond to numbers in the image on the page linked
   above.
   '''

OPEN_POSE18 = {
    "order": ['cnose', 'cneck', 'rshoulder', 'relbow', 'rwrist', 'lshoulder', 'lelbow', 'lwrist', 'rhip', 'rknee',
              'rankle', 'lhip', 'lknee', 'lankle', 'reye', 'leye', 'rear', 'lear'],
    "body": ["lhip", "lshoulder", "rshoulder", "rhip"],
    "right_lines": [
        ("rankle", "rknee"),
        ("rknee", "rhip"),
        ("rhip", "rshoulder"),
        ("rshoulder", "relbow"),
        ("relbow", "rwrist")],
    "left_lines": [
        ("lankle", "lknee"),
        ("lknee", "lhip"),
        ("lhip", "lshoulder"),
        ("lshoulder", "lelbow"),
        ("lelbow", "lwrist")],
    "right_shoulder": "rshoulder",
    "left_shoulder": "lshoulder",
    "center_nose": "cnose",
    "left_eye": "leye",
    "right_eye": "reye",
}
