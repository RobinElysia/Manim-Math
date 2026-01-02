from manim import *


class AdvancedFunction:
    """
    完全封装的函数可视化工具类。

    使用示例：
        # 创建并渲染第一个函数
        viz1 = AdvancedFunction(
            func=lambda x: (x + 3) * (x - 2) * (x + 1),
            tangent_at=1,
            show_limit_animation=False,
            x_range=[-14, 14, 1],
            y_range=[-10, 10, 1],
            x_length=14,
            y_length=8,
            output_file="polynomial.mp4"
        )
        viz1.render()

        # 创建并渲染第二个函数
        viz2 = AdvancedFunction(
            func=lambda x: np.sin(x),
            tangent_at=np.pi / 4,
            show_limit_animation=True,
            x_range=[-np.pi, np.pi, np.pi / 2],
            y_range=[-1.5, 1.5, 0.5],
            output_file="sine.mp4"
        )
        viz2.render()
    """

    def __init__(
            self,
            func: callable,
            x_range: list = [-3, 3, 1],
            y_range: list = [-4, 4, 1],
            x_length: float = 6,
            y_length: float = 5,
            axis_config: dict = None,
            graph_color=BLUE,
            tangent_at: float = None,
            tangent_color=ORANGE,
            tangent_len: float = 2,
            show_limit_animation: bool = False,
            h_values: list = None,
            secant_color=GREEN,
            limit_point_color=RED,
            run_time: float = 2,
            output_file: str = "function_viz.mp4",
            quality: str = "low"
    ):
        """
        :param func: 函数表达式 (lambda或函数)
        :param x_range: x轴范围 [最小, 最大, 步长]
        :param y_range: y轴范围 [最小, 最大, 步长]
        :param x_length: x轴屏幕长度
        :param y_length: y轴屏幕长度
        :param axis_config: 坐标轴配置
        :param graph_color: 函数图像颜色
        :param tangent_at: 切点x坐标
        :param tangent_color: 切线颜色
        :param tangent_len: 切线延伸长度
        :param show_limit_animation: 是否显示极限动画
        :param h_values: 逼近序列
        :param secant_color: 割线颜色
        :param limit_point_color: 动点颜色
        :param run_time: 动画时长
        :param output_file: 输出视频文件名
        :param quality: 质量 "low", "medium", "high"
        """
        self.config = {
            "func": func,
            "x_range": x_range,
            "y_range": y_range,
            "x_length": x_length,
            "y_length": y_length,
            "axis_config": axis_config or {"color": WHITE},
            "graph_color": graph_color,
            "tangent_at": tangent_at,
            "tangent_color": tangent_color,
            "tangent_len": tangent_len,
            "show_limit_animation": show_limit_animation,
            "h_values": h_values or [1, 0.5, 0.25, 0.1, 0.05, 0.01],
            "secant_color": secant_color,
            "limit_point_color": limit_point_color,
            "run_time": run_time,
        }
        self.output_file = output_file
        self.quality = quality

    def render(self):
        """
        渲染视频文件。
        自动处理Manim配置，用户无需关心内部实现。
        """
        # 保存原始配置
        from manim import config
        original_config = {k: config[k] for k in config.keys()}

        # 设置输出参数
        config.output_file = self.output_file
        config.quality = {
            "low": "low_quality",
            "medium": "medium_quality",
            "high": "high_quality"
        }.get(self.quality, "low_quality")

        # 配置视频格式
        config.media_dir = "./media"
        config.video_dir = "{media_dir}/videos/{module_name}"

        try:
            # 创建临时场景类并渲染
            scene = self._create_scene()
            scene.render()

        finally:
            # 恢复原始配置
            for k, v in original_config.items():
                config[k] = v

    def _create_scene(self) -> Scene:
        """创建内部场景类（完全隐藏Manim细节）"""

        class _FunctionVizScene(Scene):
            def __init__(self, viz_config):
                self.viz_config = viz_config
                super().__init__()

            def construct(self):
                cfg = self.viz_config

                # 创建坐标轴
                axes = Axes(
                    x_range=cfg["x_range"],
                    y_range=cfg["y_range"],
                    x_length=cfg["x_length"],
                    y_length=cfg["y_length"],
                    axis_config=cfg["axis_config"]
                )

                # 创建函数图像
                graph = axes.plot(cfg["func"], color=cfg["graph_color"])

                # 初始动画
                self.play(Create(axes), Create(graph), run_time=cfg["run_time"])

                # 切线相关逻辑
                if cfg["tangent_at"] is not None:
                    if cfg["show_limit_animation"]:
                        self._animate_secant_to_tangent(axes)
                    else:
                        self._draw_static_tangent(axes)

                self.wait(2)

            def _draw_static_tangent(self, axes: Axes):
                """绘制静态切线"""
                cfg = self.viz_config
                x0, y0 = cfg["tangent_at"], cfg["func"](cfg["tangent_at"])

                # 计算导数
                tangent_line, derivative = self.derivative(cfg, axes, x0, y0)

                # 创建切点和方程
                dot = Dot(axes.c2p(x0, y0), color=YELLOW, radius=0.08)
                equation = MathTex(
                    f"y = {derivative:.2f}(x-{x0:.2f})+{y0:.2f}",
                    color=cfg["tangent_color"],
                    font_size=24
                ).next_to(dot, RIGHT, buff=0.1)

                self.play(Create(tangent_line), Create(dot), Write(equation), run_time=cfg["run_time"])

            def _animate_secant_to_tangent(self, axes: Axes):
                """极限动画"""
                cfg = self.viz_config
                x0, y0 = cfg["tangent_at"], cfg["func"](cfg["tangent_at"])

                # 固定切点
                fixed_dot = Dot(axes.c2p(x0, y0), color=YELLOW, radius=0.08)
                fixed_label = MathTex(f"P({x0}, {y0:.2f})", color=YELLOW, font_size=20).next_to(
                    fixed_dot, LEFT, buff=0.1
                )
                self.play(Create(fixed_dot), Write(fixed_label))

                # 割线序列
                elements = []
                for i, h in enumerate(cfg["h_values"]):
                    x1 = x0 + h
                    y1 = cfg["func"](x1)
                    slope = (y1 - y0) / h

                    secant_line = Line(
                        axes.c2p(x0, y0),
                        axes.c2p(x1, y1),
                        color=cfg["secant_color"],
                        stroke_width=max(2.5 - i * 0.2, 1.0)
                    )

                    moving_dot = Dot(axes.c2p(x1, y1), color=cfg["limit_point_color"], radius=0.06)
                    moving_label = MathTex(f"Q({x1:.3f}, {y1:.3f})", color=cfg["limit_point_color"], font_size=18
                                           ).next_to(moving_dot, RIGHT, buff=0.1)

                    precision = 3 if i > 0 else 2
                    slope_text = MathTex(
                        f"m = \\frac{{f({x0}+{h:.3f})-f({x0})}}{{{h:.3f}}}",
                        f"=\\frac{{{y1:.{precision}f}-{y0:.2f}}}{{{h:.3f}}}={slope:.{precision}f}",
                        color=cfg["secant_color"],
                        font_size=20
                    ).to_corner(UR)

                    elements.append({
                        'line': secant_line,
                        'dot': moving_dot,
                        'label': moving_label,
                        'slope_text': slope_text
                    })

                # 动画序列
                first = elements[0]
                self.play(
                    Create(first['line']),
                    Create(first['dot']),
                    Write(first['label']),
                    Write(first['slope_text']),
                    run_time=cfg["run_time"]
                )

                for i in range(1, len(elements)):
                    cur, prev = elements[i], elements[i - 1]
                    self.play(
                        ReplacementTransform(prev['line'], cur['line']),
                        ReplacementTransform(prev['dot'], cur['dot']),
                        ReplacementTransform(prev['label'], cur['label']),
                        ReplacementTransform(prev['slope_text'], cur['slope_text']),
                        run_time=1.5
                    )

                # 最终切线
                tangent_line, derivative = self.derivative(cfg, axes, x0, y0)

                limit_text = MathTex(
                    "\\lim_{h \\to 0} \\frac{f(x+h)-f(x)}{h}=f'(x)",
                    color=YELLOW,
                    font_size=26
                ).to_corner(UL)

                tangent_eq = MathTex(
                    f"y={derivative:.3f}(x-{x0:.2f})+{y0:.2f}",
                    color=cfg["tangent_color"],
                    font_size=18
                ).next_to(axes, DOWN + RIGHT, buff=0.2)

                self.play(Write(limit_text), run_time=1)
                self.wait(0.5)
                self.play(
                    ReplacementTransform(elements[-1]['line'], tangent_line),
                    FadeOut(elements[-1]['dot']),
                    FadeOut(elements[-1]['label']),
                    run_time=2
                )
                self.play(Write(tangent_eq))
                self.play(FadeOut(elements[-1]['slope_text']))
                self.play(tangent_line.animate.set_stroke(width=4), run_time=0.5)

            def derivative(self, cfg, axes: Axes, x0, y0):
                h_final = 1e-4
                derivative = (cfg["func"](x0 + h_final) - cfg["func"](x0 - h_final)) / (2 * h_final)

                # 创建切线
                xs, xe = x0 - cfg["tangent_len"], x0 + cfg["tangent_len"]
                ys = derivative * (xs - x0) + y0
                ye = derivative * (xe - x0) + y0

                tangent_line = Line(
                    axes.c2p(xs, ys),
                    axes.c2p(xe, ye),
                    color=cfg["tangent_color"],
                    stroke_width=3
                )
                return tangent_line, derivative

        return _FunctionVizScene(self.config)