from Function.tangent import AdvancedFunction
from ML.LinearModel.LinearRegression import LinearRegression
import pandas as pd
from manim import *

def tan():
    # 创建并渲染第一个函数
    viz1 = AdvancedFunction(
        func=lambda x: (x + 3) * (x - 2) * (x + 1),
        tangent_at=1,
        show_limit_animation=False,
        x_range=[-14, 14, 1],
        y_range=[-10, 10, 1],
        x_length=14,
        y_length=8,
        output_file="polynomial.mp4",
        quality="high"
    )
    viz1.render()
    # 创建并渲染第二个函数
    viz2 = AdvancedFunction(
        func=lambda x: np.sin(x),
        tangent_at=np.pi / 4,
        show_limit_animation=True,
        x_range=[-np.pi, np.pi, np.pi / 2],
        y_range=[-1.5, 1.5, 0.5],
        output_file="sine.mp4",
        quality="high"
    )
    viz2.render()

def linear():
    # 创建示例数据
    np.random.seed(42)
    x = np.linspace(-4, 4, 20)
    y = -1.3 * x + 0.5 + np.random.randn(20) * 2.0
    example_data = pd.DataFrame({'x': x, 'y': y})
    # 创建并渲染动画
    scene = LinearRegression(
        x_range=[-5, 5],
        y_range=[-8, 8],
        x_length=10,
        y_length=8,
        axis_config={"color": WHITE},
        target_color=BLUE,
        fitting_color=RED,
        example_point=example_data
    )
    scene.UnivariateLinearRegression()
    scene.render()

if __name__ == "__main__":
    tan()
    linear()