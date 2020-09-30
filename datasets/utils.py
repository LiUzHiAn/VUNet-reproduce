import numpy as np
import PIL
from datasets.keypoints_mode import OPEN_POSE18
from PIL import Image, ImageDraw
import cv2


# COLORS = {
#     "body": (0, 0, 255, 0),  # blue
#     "left": (0, 255, 0, 0),  # green
#     "right": (255, 0, 0, 0)  # red
# }


def valid_joints(*joints):
    j = np.stack(joints)
    return (j >= 0).all()


def preprocess_img(x):
    """From uint8 image to [-1,1]."""
    x = np.cast[np.float32](x / 127.5 - 1.0)
    # x = np.transpose(x, axes=[2, 0, 1])
    return x


def postprocess(x):
    """[-1,1] to uint8."""
    x = (x + 1.0) / 2.0
    x = np.clip(255 * x, 0, 255)
    x = np.cast[np.uint8](x)
    return x


def load_img(path, target_shape):
    """Load image. target_size is specified as (height, width, channels)
    where channels == 1 means grayscale. uint8 image returned."""
    img = PIL.Image.open(path)
    grayscale = target_shape[2] == 1
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    wh_tuple = (target_shape[1], target_shape[0])
    if img.size != wh_tuple:
        img = img.resize(wh_tuple, resample=PIL.Image.BILINEAR)

    x = np.asarray(img, dtype="uint8")
    if len(x.shape) == 2:
        x = np.expand_dims(x, -1)

    return x


def cv2_keypoints2stickman(kps, spatial_size=256, keypoints_mode=OPEN_POSE18):
    # Create canvas

    scale_factor = spatial_size / 128
    thickness = int(3 * scale_factor)

    imgs = list()
    for i in range(3):
        imgs.append(np.zeros((spatial_size, spatial_size), dtype="uint8"))

    joints_order = keypoints_mode["order"]
    # 组成身体的部分,用蓝色
    body_point_indices = [joints_order.index(point) for point in keypoints_mode["body"]]
    body_pts = []
    for idx in body_point_indices:
        point = np.int_(kps[idx]).tolist()
        # if point[0] <= 0 and point[1] <= 0:
        #     continue  # 忽略小于0的部分
        body_pts += [tuple(point)]
    cv2.fillPoly(imgs[2], np.array([body_pts], dtype=np.int32), 255)

    # 人的右边
    for line in keypoints_mode["right_lines"]:
        line_indices = [joints_order.index(line[0]), joints_order.index(line[1])]
        a = tuple(np.int_(kps[line_indices[0]]))
        b = tuple(np.int_(kps[line_indices[1]]))
        if (a[0] <= 0 and a[1] <= 0) \
                or (b[0] <= 0 and b[1] <= 0):
            continue
        cv2.line(imgs[0], a, b, color=255, thickness=thickness)
    # 人的左边
    for line in keypoints_mode["left_lines"]:
        line_indices = [joints_order.index(line[0]), joints_order.index(line[1])]
        a = tuple(np.int_(kps[line_indices[0]]))
        b = tuple(np.int_(kps[line_indices[1]]))
        if (a[0] <= 0 and a[1] <= 0) \
                or (b[0] <= 0 and b[1] <= 0):
            continue
        cv2.line(imgs[1], a, b, color=255, thickness=thickness)

    # 头部的连线
    nose_point = kps[joints_order.index(keypoints_mode["center_nose"])]  # 鼻子
    right_shoulder_ponit = kps[joints_order.index(keypoints_mode["right_shoulder"])]
    right_eye_ponit = kps[joints_order.index(keypoints_mode["right_eye"])]
    left_shoulder_ponit = kps[joints_order.index(keypoints_mode["left_shoulder"])]
    left_eye_ponit = kps[joints_order.index(keypoints_mode["left_eye"])]
    neck = (right_shoulder_ponit + left_shoulder_ponit) / 2
    # 鼻子到脖子的连线
    a = tuple(np.int_(neck))
    b = tuple(np.int_(nose_point))
    if np.min(a) >= 0 and np.min(b) >= 0:
        cv2.line(imgs[0], a, b, color=127, thickness=thickness)
        cv2.line(imgs[1], a, b, color=127, thickness=thickness)

    # 鼻子到眼睛的连线
    cn = tuple(np.int_(nose_point))
    leye = tuple(np.int_(left_eye_ponit))
    reye = tuple(np.int_(right_eye_ponit))
    if np.min(reye) >= 0 and np.min(leye) >= 0 and np.min(cn) >= 0:
        cv2.line(imgs[0], cn, reye, color=255, thickness=thickness)  # 右边眼睛为[255,0,0],蓝色
        cv2.line(imgs[1], cn, leye, color=255, thickness=thickness)  # 右边眼睛为[0,255,0],绿色

    # todo: Draw Points  画所有的关节点
    img = np.stack(imgs, axis=-1)

    return img


def keypoints2stickman(kps, spatial_size=256, keypoints_mode=OPEN_POSE18):
    # Create canvas
    im = Image.fromarray(np.zeros([spatial_size] * 2 + [3], dtype='uint8'))
    draw = ImageDraw.Draw(im)
    scale_factor = spatial_size / 128
    thickness = int(3 * scale_factor)

    joints_order = keypoints_mode["order"]
    # Draw Body Polygon  # 组成身体的部分,用蓝色
    body_point_indices = [joints_order.index(point) for point in keypoints_mode["body"]]
    body = []
    for idx in body_point_indices:
        point = np.int_(kps[idx]).tolist()
        # if point[0] <= 0 and point[1] <= 0:
        #     continue  # 忽略小于0的部分
        body += [tuple(point)]
    draw.polygon(body, outline=(0, 0, 255), fill=(0, 0, 255))  # blue

    # 人的右边
    for line in keypoints_mode["right_lines"]:
        line_indices = [joints_order.index(line[0]), joints_order.index(line[1])]
        a = tuple(np.int_(kps[line_indices[0]]))
        b = tuple(np.int_(kps[line_indices[1]]))
        if (a[0] <= 0 and a[1] <= 0) \
                or (b[0] <= 0 and b[1] <= 0):
            continue
        draw.line([a, b], fill=(255, 0, 0), width=thickness)
    # 人的左边
    for line in keypoints_mode["left_lines"]:
        line_indices = [joints_order.index(line[0]), joints_order.index(line[1])]
        a = tuple(np.int_(kps[line_indices[0]]))
        b = tuple(np.int_(kps[line_indices[1]]))
        if (a[0] <= 0 and a[1] <= 0) \
                or (b[0] <= 0 and b[1] <= 0):
            continue
        draw.line([a, b], fill=(0, 255, 0), width=thickness)

    # 头部的连线
    nose_point = kps[joints_order.index(keypoints_mode["center_nose"])]  # 鼻子
    right_shoulder_ponit = kps[joints_order.index(keypoints_mode["right_shoulder"])]
    right_eye_ponit = kps[joints_order.index(keypoints_mode["right_eye"])]
    left_shoulder_ponit = kps[joints_order.index(keypoints_mode["left_shoulder"])]
    left_eye_ponit = kps[joints_order.index(keypoints_mode["left_eye"])]
    neck = (right_shoulder_ponit + left_shoulder_ponit) / 2
    # 鼻子到脖子的连线
    a = tuple(np.int_(neck))
    b = tuple(np.int_(nose_point))
    if np.min(a) >= 0 and np.min(b) >= 0:
        draw.line([a, b], fill=(127, 127, 0), width=thickness)

    # 鼻子到眼睛的连线
    cn = tuple(np.int_(nose_point))
    leye = tuple(np.int_(left_eye_ponit))
    reye = tuple(np.int_(right_eye_ponit))
    if np.min(reye) >= 0 and np.min(leye) >= 0 and np.min(cn) >= 0:
        draw.line([cn, reye], fill=(255, 0, 0), width=thickness)
        draw.line([cn, leye], fill=(0, 255, 0), width=thickness)

    # todo: Draw Points  画所有的关节点

    return np.array(im)


def get_crop(bpart, joints, jo, wh, o_w, o_h, ar=1.0):
    dst = np.float32([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    part_dst = np.float32(wh * dst)

    bpart_indices = [jo.index(b) for b in bpart]
    part_src = np.float32(joints[bpart_indices])

    # fall backs
    if not valid_joints(part_src):
        if bpart[0] == "lhip" and bpart[1] == "lknee":
            bpart = ["lhip"]
            bpart_indices = [jo.index(b) for b in bpart]
            part_src = np.float32(joints[bpart_indices])
        elif bpart[0] == "rhip" and bpart[1] == "rknee":
            bpart = ["rhip"]
            bpart_indices = [jo.index(b) for b in bpart]
            part_src = np.float32(joints[bpart_indices])
        elif bpart[0] == "lshoulder" and bpart[1] == "rshoulder" and bpart[2] == "cnose":
            bpart = ["lshoulder", "rshoulder", "rshoulder"]
            bpart_indices = [jo.index(b) for b in bpart]
            part_src = np.float32(joints[bpart_indices])

    if not valid_joints(part_src):  # 没有手的跳过
        return None, part_src, part_dst

    if part_src.shape[0] == 1:  # 只有1个点
        # leg fallback
        a = part_src[0]
        b = np.float32([a[0], o_h - 1])
        part_src = np.float32([a, b])

    if part_src.shape[0] == 4:
        pass
    elif part_src.shape[0] == 3:
        # lshoulder, rshoulder, cnose
        if bpart == ["lshoulder", "rshoulder", "rshoulder"]:
            segment = part_src[1] - part_src[0]
            normal = np.array([-segment[1], segment[0]])
            if normal[1] > 0.0:
                normal = -normal

            a = part_src[0] + normal
            b = part_src[0]
            c = part_src[1]
            d = part_src[1] + normal
            part_src = np.float32([a, b, c, d])
        else:
            assert bpart == ["lshoulder", "rshoulder", "cnose"]
            neck = 0.5 * (part_src[0] + part_src[1])
            neck_to_nose = part_src[2] - neck
            part_src = np.float32([neck + 2 * neck_to_nose, neck])

            # segment box
            segment = part_src[1] - part_src[0]
            normal = np.array([-segment[1], segment[0]])
            alpha = 1.0 / 2.0
            a = part_src[0] + alpha * normal
            b = part_src[0] - alpha * normal
            c = part_src[1] - alpha * normal
            d = part_src[1] + alpha * normal
            # part_src = np.float32([a,b,c,d])
            part_src = np.float32([b, c, d, a])
    else:
        assert part_src.shape[0] == 2

        segment = part_src[1] - part_src[0]
        normal = np.array([-segment[1], segment[0]])
        alpha = ar / 2.0
        a = part_src[0] + alpha * normal
        b = part_src[0] - alpha * normal
        c = part_src[1] - alpha * normal
        d = part_src[1] + alpha * normal
        part_src = np.float32([a, b, c, d])

    M = cv2.getPerspectiveTransform(part_src, part_dst)
    return M, part_src, part_dst


def normalize(img, coords, stickman, joints_order, box_factor):
    """in-plane normalization, 输入的img和stickman的范围都是 [-1,1]"""
    verbose = True

    o_h, o_w = img.shape[0], img.shape[1]
    h = o_h // 2 ** box_factor
    w = o_w // 2 ** box_factor
    wh = np.array([w, h])
    wh = np.expand_dims(wh, 0)  # [64,64]

    bparts = [
        ["lshoulder", "lhip", "rhip", "rshoulder"],
        ["lshoulder", "rshoulder", "cnose"],
        ["lshoulder", "lelbow"],
        ["lelbow", "lwrist"],
        ["rshoulder", "relbow"],
        ["relbow", "rwrist"],
        ["lhip", "lknee"],
        ["rhip", "rknee"]]
    ar = 0.5

    part_imgs = list()
    part_stickmen = list()
    for bpart in bparts:
        part_img = np.zeros((h, w, 3))
        part_stickman = np.zeros((h, w, 3))
        M, part_src, part_dst = get_crop(bpart, coords, joints_order, wh, o_w, o_h, ar)

        if M is not None:
            part_img = cv2.warpPerspective(img, M, (h, w),
                                           borderMode=cv2.BORDER_REPLICATE)  # 将源图片的 当前bpart做透视变换 得到（h,w）大小的patch
            part_stickman = cv2.warpPerspective(stickman, M, (h, w), borderMode=cv2.BORDER_REPLICATE)
            if verbose:
                verbose_img = postprocess(img)[:, :, ::-1]
                verbose_img = cv2.circle(verbose_img, tuple(part_src[0]), radius=2, color=(255, 0, 0), thickness=2)
                verbose_img = cv2.circle(verbose_img, tuple(part_src[1]), radius=2, color=(0, 255, 0), thickness=2)
                verbose_img = cv2.circle(verbose_img, tuple(part_src[2]), radius=2, color=(0, 0, 255), thickness=2)
                verbose_img = cv2.circle(verbose_img, tuple(part_src[3]), radius=2, color=(0, 0, 0), thickness=2)

                verbose_stickman = postprocess(stickman)
                verbose_stickman = cv2.circle(verbose_stickman, tuple(part_src[0]), radius=2, color=(255, 0, 0),
                                              thickness=2)  # 蓝
                verbose_stickman = cv2.circle(verbose_stickman, tuple(part_src[1]), radius=2, color=(0, 255, 0),
                                              thickness=2)  # 绿
                verbose_stickman = cv2.circle(verbose_stickman, tuple(part_src[2]), radius=2, color=(0, 0, 255),
                                              thickness=2)  # 红
                verbose_stickman = cv2.circle(verbose_stickman, tuple(part_src[3]), radius=2, color=(0, 0, 0),
                                              thickness=2)  # 黑

                cv2.imshow("part_img", postprocess(part_img)[:, :, ::-1])
                cv2.imshow("img", verbose_img)
                cv2.imshow("part_stickman", postprocess(part_stickman))
                cv2.imshow("stickman", verbose_stickman)
                cv2.waitKey(0)

        part_imgs.append(part_img)
        part_stickmen.append(part_stickman)
    img = np.concatenate(part_imgs, axis=2)
    stickman = np.concatenate(part_stickmen, axis=2)

    return img, stickman
