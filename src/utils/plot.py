
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from src.utils.utils import get_box_coord


def plot_boxes(box, H=448, W=448, S=7, fig=None, ax=None, color="red", plot_corners=False):
    if fig is None and ax is None:
        fig, ax = plt.subplots()

    img = mpimg.imread("/home/aldi/workspace/yolo-v1/img/cat.png")[:H, :W, :]
    ax.imshow(img)

    for i in range(box.shape[0]):
        for j in range(box.shape[1]):
            if box[i, j, 4:].any() != 0:
                x_ = H / S * i
                y_ = W / S * j

                # Plot Points:
                points = get_box_coord(box[i, j, :])
                points += [x_, y_]
                if plot_corners:
                    for i in range(points.shape[0]):
                        ax.plot(points[i, 0], points[i, 1], color=color, marker="o")
                        ax.text(points[i, 0], points[i, 1], str(i))

                # Plot Box:
                pt_idx = [[0, 1], [1, 2], [2, 3], [3, 0]]
                for idx in range(len(pt_idx)):
                    i, j = pt_idx[idx][0], pt_idx[idx][1]
                    ax.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], color=color)
                
    return fig, ax
