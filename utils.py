import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Tuple


def plot_joint_distribution(joint_matirx:np.ndarray, 
                                 margin_x: np.ndarray=None, 
                                 margin_y: np.ndarray=None, 
                                 title: str=None,
                                 save_dir: str=None,
                                 *args,
                                 **kwargs,
                                 )->Tuple[plt.Figure, plt.Axes]:
    """
    plot the joint distribution and the marginal distribution
    :param joint_matirx: the joint distribution. shape: (n, n)
    :param margin_x: the marginal distribution of x. shape: (n,)
    :param margin_y: the marginal distribution of y. shape: (n,)
    :param title: the title of the plot
    :param save_dir: the directory to save the plot
    :param args: the arguments of plt.subplots
    :param kwargs: the keyword arguments of plt.subplots
    return: the figure and the axes
    """
    assert joint_matirx.ndim == 2 and joint_matirx.shape[0] == joint_matirx.shape[1]
    n = joint_matirx.shape[0]
    margin_x = joint_matirx.sum(axis=0) if margin_x is None else margin_x
    margin_y = joint_matirx.sum(axis=1) if margin_y is None else margin_y

    margin_x = margin_x.astype(np.float64)
    margin_y = margin_y.astype(np.float64)
    joint_matirx = joint_matirx.astype(np.float64)
    
    # plt.figure(figsize=(10, 10))
    fig, axs = plt.subplots(1, 1, figsize=(10, 10), *args, **kwargs)

    axs.get_xaxis().set_visible(False)
    axs.get_yaxis().set_visible(False)
    divider = make_axes_locatable(axs)

    # 创建额外的轴
    ax_x = divider.append_axes("top", size=1.2, pad=0.1, sharex=axs)
    ax_y = divider.append_axes("left", size=1.2, pad=0.1, sharey=axs)

    # 热力图
    cax = axs.imshow(joint_matirx,cmap='Reds')

    # 绘制边缘分布
    ax_x.plot(range(n), margin_x, color='blue')
    ax_y.plot(-margin_y, range(n), color='red')

    # 设置轴的属性
    ax_x.get_xaxis().set_visible(False)
    ax_x.get_yaxis().set_visible(False)
    ax_y.get_xaxis().set_visible(False)
    ax_y.get_yaxis().set_visible(False)

    if title:
        axs.set_title(title)
    fig.show()
    
    if save_dir:
        fig.savefig(save_dir, bbox_inches='tight')
    
    return fig, axs


def plot_precomputed_pdf(pdf_values: np.ndarray,
                         x_range: Tuple[float, float]=None,
                         time_range: Tuple[float, float]=None,
                         title: str=None,
                         save_dir: str=None,
                         ):
    """
    绘制预先计算好的随时间变化的一维概率密度函数。

    :param pdf_values: 形状为 (n_times, num_points) 的数组，包含概率密度值。
    :param x_range: 元组，表示一维空间变量的范围 (min, max)。
    :param time_range: 元组，表示时间的范围 (start, end)。
    """
    # 验证输入数组的形状
    n_times, num_points = pdf_values.shape

    # 创建空间变量和时间点
    times = np.linspace(0, n_times, n_times) if time_range is None else np.linspace(time_range[0], time_range[1], n_times)
    x =  np.linspace(0, num_points, num_points) if x_range is None else np.linspace(x_range[0], x_range[1], num_points)

    # 创建三维图形
    fig = plt.figure(figsize=(10,8))
    ax = plt.axes(projection='3d')

    # 用不同的颜色表示不同的时间
    colors = plt.cm.viridis(np.linspace(0, 1, n_times))

    # 绘制每个时间点的概率密度函数
    for t, y, c in zip(times[::-1], pdf_values[::-1], colors):
        ax.plot(x, y, zs=t, zdir='y', color=c, alpha=0.8, linewidth=2)

    # 设置坐标轴标签
    ax.set_xlabel('X Axis (Space)')
    ax.set_ylabel('Y Axis (Time)')
    ax.set_zlabel('Z Axis (Probability Density)')

    # 设置刻度字体大小
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='z', labelsize=10)
    
    # make grid alpha=0.5
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0.5)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0.5)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0.5)

    ax.xaxis.set_pane_color((0.9, 0.9, 0.9, 1.0))
    ax.yaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))
    ax.zaxis.set_pane_color((0.85, 0.85, 0.85, 1.0))


    
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0., top=1)

    # 显示图形
    if title:
        ax.set_title(title)

    fig.show()
    
    if save_dir:
        fig.savefig(save_dir, bbox_inches='tight')
    
    return fig, ax

# 为什么训练这么慢