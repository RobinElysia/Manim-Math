import pandas as pd
from manim import *
from scipy.stats import linregress


class LinearRegression(Scene):
    def __init__(
            self,
            x_range=[-5, 5],
            y_range=[-5, 5],
            x_length=8,
            y_length=6,
            axis_config={"color": WHITE},
            target_color=BLUE,
            fitting_color=RED,
            example_point: pd.DataFrame = None,
            **kwargs
    ):
        """
        Initialize the linear regression animation class
        Args:
        x_range: Range of x-axis values
        y_range: Range of y-axis values
        x_length: Length of x-axis
        y_length: Length of y-axis
        axis_config: Axis configuration
        target_color: Target function color
        fitting_color: Fitting function color
        example_point: Sample point DataFrame, containing 'x' and 'y' columns
        """
        super().__init__(**kwargs)
        self.x_range = x_range
        self.y_range = y_range
        self.x_length = x_length
        self.y_length = y_length
        self.axis_config = axis_config
        self.target_color = target_color
        self.fitting_color = fitting_color
        self.example_point = example_point

        # Calculate real parameters
        self.calculate_true_regression()

    def calculate_true_regression(self):
        """Calculate the true least squares regression parameters"""
        x = self.example_point['x'].values
        y = self.example_point['y'].values
        self.true_slope, self.true_intercept, _, _, _ = linregress(x, y)

    def get_line(self, slope, intercept):
        """A linear function that yields a given slope and intercept"""
        return lambda x: slope * x + intercept

    def create_regression_line(self, slope, intercept, color, label):
        """Create a regression line object"""
        line = self.axes.plot(
            self.get_line(slope, intercept),
            color=color,
            stroke_width=3
        )

        # Add tag
        label = MathTex(
            f"y = {slope:.2f}x + {intercept:.2f}",
            color=color,
            font_size=18
        )

        return line, label

    def create_scatter_plot(self):
        """Create a scatter plot"""
        points = []
        for _, row in self.example_point.iterrows():
            point = Dot(
                point=self.axes.coords_to_point(row['x'], row['y']),
                color=YELLOW,
                radius=0.08
            )
            points.append(point)
        return VGroup(*points)

    def calculate_loss(self, slope, intercept):
        """Calculate the loss (mean squared error) under the current parameters."""
        x = self.example_point['x'].values
        y = self.example_point['y'].values
        y_pred = slope * x + intercept
        mse = np.mean((y - y_pred) ** 2)
        return mse

    def create_loss_display(self, slope, intercept):
        """Create a loss display"""
        loss = self.calculate_loss(slope, intercept)
        loss_text = Text(f"mean square error: {loss:.4f}", font_size=18, color=GREEN)
        return loss_text

    def create_parameter_display(self, slope, intercept):
        """Create parameter display"""
        param_text = Text(
            f"slope: {slope:.2f}, intercept: {intercept:.2f}",
            font_size=18,
            color=WHITE
        )
        param_text.to_corner(UL)
        return param_text

    def gradient_descent_step(self, slope, intercept, learning_rate=0.01):
        """Perform one gradient descent step"""
        x = self.example_point['x'].values
        y = self.example_point['y'].values

        # Calculate gradient
        n = len(x)
        y_pred = slope * x + intercept
        d_slope = (-2 / n) * np.sum(x * (y - y_pred))
        d_intercept = (-2 / n) * np.sum(y - y_pred)

        # update data
        new_slope = slope - learning_rate * d_slope
        new_intercept = intercept - learning_rate * d_intercept

        return new_slope, new_intercept

    def UnivariateLinearRegression(self):
        """
        Execute univariate linear regression animation

        Returns:
            Scene: Manim scene containing all animations
        """
        # create axis
        self.axes = Axes(
            x_range=self.x_range,
            y_range=self.y_range,
            x_length=self.x_length,
            y_length=self.y_length,
            axis_config=self.axis_config
        )
        axes_labels = self.axes.get_axis_labels(x_label="x", y_label="y")
        # create Scatter plot
        scatter_plot = self.create_scatter_plot()
        # create target linear
        line, label = self.create_regression_line(
            self.true_slope, self.true_intercept,
            self.target_color, "target linear"
        )
        label.to_corner(DR)
        # init random data
        np.random.seed(42)
        initial_slope = np.random.uniform(-2, 2)
        initial_intercept = np.random.uniform(-3, 3)
        # create init linear
        fitting_line, fitting_label = self.create_regression_line(
            initial_slope, initial_intercept,
            self.fitting_color, "fit linear"
        )
        fitting_label.to_corner(DR+UP*0.8)
        # create loss display
        loss_display = self.create_loss_display(initial_slope, initial_intercept)
        # create data display
        param_display = self.create_parameter_display(initial_slope, initial_intercept)
        loss_display.next_to(param_display, DOWN)
        # add all element to scene
        self.play(Create(self.axes), Write(axes_labels))
        self.play(Create(scatter_plot))
        self.play(Create(line), Write(label))
        self.wait(1)
        # add fit linear
        self.play(Create(fitting_line), Write(fitting_label))
        self.play(Write(loss_display), Write(param_display))
        self.wait(1)
        # exec SGD
        current_slope, current_intercept = initial_slope, initial_intercept
        n_iterations = 20

        for i in range(n_iterations):
            # calculate new data
            new_slope, new_intercept = self.gradient_descent_step(
                current_slope, current_intercept,
                learning_rate=0.1 / (1 + i * 0.1)  # 动态学习率
            )
            # create new liner
            new_fitting_line, new_fitting_label = self.create_regression_line(
                new_slope, new_intercept,
                self.fitting_color, "fit linear"
            )
            new_fitting_label.to_corner(DR+UP*0.8)
            # create new loss
            new_loss_display = self.create_loss_display(new_slope, new_intercept)
            # create new data
            new_param_display = self.create_parameter_display(new_slope, new_intercept)
            new_loss_display.next_to(new_param_display, DOWN)
            # transform
            self.play(
                Transform(fitting_label, new_fitting_label),
                Transform(fitting_line, new_fitting_line),
                Transform(loss_display, new_loss_display),
                Transform(param_display, new_param_display),
                run_time=0.5
            )
            # update current data
            current_slope, current_intercept = new_slope, new_intercept
            # 5s stop
            if (i + 1) % 5 == 0:
                self.wait(0.5)
        self.wait(2)
        return self