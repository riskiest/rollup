from cv2 import cv2
from math import cos, sin
import numpy as np
from pytransform3d.rotations import q_id, q_prod_vector, active_matrix_from_extrinsic_euler_zyx, quaternion_from_matrix
from pytransform3d.batch_rotations import quaternion_slerp_batch
import os

os.makedirs('output', exist_ok=True)
os.makedirs('video', exist_ok=True)

# 步数
n_steps = 50

# 画板的大小 1000*1400
canvas = cv2.imread(r'input\bird.png')
canvas_y, canvas_x, _ = canvas.shape
# canvas = np.zeros((canvas_y, canvas_x, 3), dtype=np.uint8)

# 读取纸张
paper = cv2.imread(r'input\paper.png')
jmax, imax, _ = paper.shape

# 纸张放在画布的初始位置，这里定位中心，并建立起纸张坐标->画布坐标的转换关系
dx, dy = (canvas_x - imax)//2, (canvas_y-jmax)//2
def paper2canvas(x, y):
    return x+dx, y+dy

# 通过欧拉角获得旋转的四元数
z_rot, y_rot, x_rot = 30, 30, 15
rot_matrix = active_matrix_from_extrinsic_euler_zyx(((z_rot/180*np.pi), (y_rot/180*np.pi), (x_rot/180*np.pi)))
qs = quaternion_slerp_batch(q_id, quaternion_from_matrix(rot_matrix), np.linspace(0, 1, n_steps))

def do_rot(v, t):
    '''对t时刻时的向量（点）v做旋转变换'''
    return q_prod_vector(qs[t], v)

# 除了旋转之外，还会有平移
trans_start, trans_max = (0, 0, 0), (0, 0, 1600)
trans = np.linspace(trans_start, trans_max, n_steps)
def do_trans(v, t):
    return (v[0]+trans[t][0]), (v[1]+trans[t][1]), (v[2]+trans[t][2])

# 透视的话需要知道眼睛的位置
eye = (imax//2, jmax//2, -1600)
def proj(v):
    '''将空间坐标映射到纸张初始平面'''
    return eye[2]/(v[2]-eye[2])*(eye[0]-v[0])+eye[0], eye[2]/(v[2]-eye[2])*(eye[1]-v[1])+eye[1]

# 非线性的变换
theta0, theta1 = 0, 180
thetas = np.linspace(theta0/180*np.pi, theta1/180*np.pi, n_steps)

b = imax/2

def roll(x, y, t):
    if t==0:
        return x,y,0
    radius = b/thetas[t]
    theta = (x-b)/radius
    return (b+radius*sin(theta), y, radius*(1-cos(theta)))

# 每次循环获取一张图
for t in range(n_steps):
    canvas_t = canvas.copy()
    frontest = {}
    for i in range(imax):
        for j in range(jmax):
            rolled = roll(i, j, t)
            # transed = do_trans(rolled, t)
            # roted = do_rot(transed, t)
            # paper_i, paper_j = proj(roted)
            roted = do_rot(rolled, t)
            transed = do_trans(roted, t)
            paper_i, paper_j = proj(transed)
            canvas_i, canvas_j = paper2canvas(paper_i, paper_j)
            x, y = int(canvas_i), int(canvas_j)
            if (0<=x<canvas_x and 0<=y<canvas_y) and ((x,y) not in frontest or frontest[(x,y)]>transed[2]):
                frontest[(x,y)] = transed[2]
                canvas_t[y][x][:] = paper[j][i][:]
    print(f"t: {t}")
    cv2.imwrite(f"./output/roll_{t}.png", canvas_t)

# 视频制作
image_folder = 'output'
video_name = 'video/video.avi'
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort(key=lambda x:int(x[5:][:-4]))
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
video = cv2.VideoWriter(video_name, 0, 20, (width,height))
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))
cv2.destroyAllWindows()
video.release()            