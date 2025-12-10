"""
mp4.py

根據輸入的 CSV 檔案（space 或 comma 分隔、無表頭）產生 n-body 軌跡的 MP4 視覺化影片。

CSV 格式說明（n = 5, d = 3 的範例）：
frame 0 (5 rows)
x1 y1 z1
x2 y2 z2
x3 y3 z3
x4 y4 z4
x5 y5 z5
frame 1 (5 rows)
x1 y1 z1
... 依此類推，共 d 個 time-steps，每個 time-step 包含 n 行 (x y z)。

用法範例：
    python mp4.py input.csv --n 5 --d 100 --t 10 --output out.mp4

參數：
  input.csv    : 必填，輸入檔案路徑
  --n N        : 每個時間點的質點數量 (int)
  --d D        : 時間間隔數量 / frames (int)
  --t T        : 影片總時長（秒，float）
  --output     : 輸出 mp4 檔名（預設：nbody.mp4）
  --fps        : 影片寫出時的 fps (預設 24)。注意：我們使用 MoviePy 的 durations 參數以確保每張影格顯示 t/d 秒。
  --size       : 影像尺寸（像素，例如 800，代表 800x800，預設 800）
  --trail      : 是否繪製軌跡線（預設關閉），設定為整數表示軌跡長度（多少個先前位置），例如 --trail 10

需要安裝的套件：
    pip install numpy matplotlib moviepy pillow

"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip


def parse_args():
    p = argparse.ArgumentParser(description='Generate MP4 from n-body CSV')
    p.add_argument('csv', help='input CSV path (space/comma separated, no header)')
    p.add_argument('--n', type=int, required=True, help='number of bodies per time-step')
    p.add_argument('--d', type=int, required=True, help='number of time-steps (frames)')
    p.add_argument('--t', type=float, required=True, help='total video duration (seconds)')
    p.add_argument('--output', default='nbody.mp4', help='output mp4 file')
    p.add_argument('--fps', type=int, default=24, help='output FPS for writer (default: 24)')
    p.add_argument('--size', type=int, default=800, help='square image size in pixels (default: 800)')
    p.add_argument('--trail', type=int, default=0, help='trail length in frames (0 = none)')
    return p.parse_args()


def load_and_validate(csv_path, n, d):
    # 支援 space 或 comma 分隔
    try:
        data = np.loadtxt(csv_path, delimiter=',')
    except Exception as e:
        raise RuntimeError(f'讀取 CSV 失敗: {e}')

    if data.ndim == 1:
        # 只有一列或一行
        if data.size % 3 != 0:
            raise ValueError('輸入檔案的資料數量無法被 3 整除，確認每行包含 x y z')
        data = data.reshape((-1, 3))

    if data.shape[1] != 3:
        raise ValueError('輸入檔案每行必須有 3 個欄位 (x y z)')

    total_rows = data.shape[0]
    expected = n * d
    if total_rows != expected:
        raise ValueError(f'輸入檔案列數 ({total_rows}) 與 n*d ({n}*{d}={expected}) 不符')

    frames = data.reshape((d, n, 3))  # shape: (d, n, 3)
    return frames


def make_colors(n):
    # 若 n <= 20 則使用 tab20，否則用 HSV 分佈
    import matplotlib.cm as cm
    if n <= 20:
        cmap = cm.get_cmap('tab20')
        cols = [cmap(i) for i in np.linspace(0, 1, n)]
    else:
        # HSV 分佈
        cols = [matplotlib.colors.hsv_to_rgb((i / n, 0.9, 0.9)) for i in range(n)]
    # 轉成 matplotlib 可接受格式 (RGB tuple)
    return cols


def render_frames(frames, colors, size_px=800, trail=0):
    d, n, _ = frames.shape

    # 全域範圍（x, y）便於固定座標軸
    all_xy = frames[:, :, :2].reshape((-1, 2))
    x_min, y_min = np.min(all_xy, axis=0)
    x_max, y_max = np.max(all_xy, axis=0)

    # 若範圍過小，補一個小 margin
    xr = x_max - x_min
    yr = y_max - y_min
    if xr == 0:
        xr = 1.0
    if yr == 0:
        yr = 1.0
    margin = 0.06 * max(xr, yr)
    x_min -= margin
    x_max += margin
    y_min -= margin
    y_max += margin

    imgs = []

    for fi in range(d):
        fig = plt.figure(figsize=(size_px / 100, size_px / 100), dpi=100, facecolor='black')
        ax = fig.add_subplot(111, projection='3d')
        # 黑色背景
        ax.set_facecolor('black')

        # 畫每一個點
        xs = frames[fi, :, 0]
        ys = frames[fi, :, 1]
        zs = frames[fi, :, 2]

        # marker size 根據 z 做一個相對大小 (可自行調整係數)
        zmin = np.min(frames[:, :, 2])
        zmax = np.max(frames[:, :, 2])
        if zmax - zmin == 0:
            sizes = np.full(n, 60.0)
        else:
            sizes = 20 + 80 * ( (zs - zmin) / (zmax - zmin) )

        # scatter
        ax.scatter(xs, ys, zs, s=sizes, c=colors, marker='o', linewidths=0)

        # 若啟用 trail，畫出先前的軌跡
        if trail > 0:
            start = max(0, fi - trail)
            for i in range(n):
                traj = frames[start:fi+1, i, :2]
                # 使用較淡的顏色來畫線
                ax.plot(traj[:, 0], traj[:, 1], alpha=0.6, linewidth=1)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        # x_min = y_min = -1
        # x_max = y_max = 1
        ax.set_aspect('equal', 'box')
        ax.axis('off')

        # 把 figure 轉成 numpy array (RGB)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = buf.reshape((h, w, 4))[:, :, :3]  # 去掉 alpha
        imgs.append(img)
        plt.close(fig)

    return imgs


def main():
    args = parse_args()
    frames = load_and_validate(args.csv, args.n, args.d)
    

    colors = make_colors(args.n)

    imgs = render_frames(frames, colors, size_px=args.size, trail=args.trail)

    # 每個影格的顯示時間
    seconds_per_frame = float(args.t) / float(args.d)
    durations = [seconds_per_frame] * args.d

    # 使用 ImageSequenceClip，並以 durations 指定每張影像的顯示時間
    clip = ImageSequenceClip(imgs, durations=durations)

    # 寫出影片
    # moviepy 仍需要 fps 參數；即便我們用 durations 控制每張影格時間，寫出時的 fps 也必須給定
    clip.write_videofile(args.output, fps=args.fps, codec='libx264', audio=False)


if __name__ == '__main__':
    main()
