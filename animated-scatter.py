import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

## These are some hard coded values for data/demo.avi
# We make a transformation form the camera angle to a
# strategic (birds eye) view.
RECT_COORD = [[683, 519], [1259, 169], [844, 120], [110, 250]]
TARGET_COORD = [[0, 0], [280, 0], [280, 150], [0, 150]]
TRANSFORM_MAT = cv2.getPerspectiveTransform(
    np.float32(np.array(RECT_COORD)), np.float32(np.array(TARGET_COORD))
)


class AnimatedScatter:
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""

    def __init__(self, data):
        self.data = data
        self.stream = self.data_stream()

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        # FuncAnimation calls self.update repeatedly, yielding the animation.
        self.ani = animation.FuncAnimation(
            self.fig, self.update, interval=5, init_func=self.setup_plot, blit=True
        )

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y, s, c = next(self.stream).T
        self.scat = self.ax.scatter(x, y, vmin=0, vmax=80, cmap="jet", edgecolor="k")
        self.ax.axis([-50, 320, -50, 200])
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return (self.scat,)

    def data_stream(self):
        """Create an iterator, that return the next frame in the animaton."""
        for _idx, row in self.data.iterrows():
            xy = cv2.perspectiveTransform(
                # Create a 2D np array with the x-y coordinates of each player in the frame.
                np.nan_to_num(
                    np.array(
                        [
                            list(
                                zip(
                                    # Each row is of format x_p1, y_p1, x_p2, y_p2, ...
                                    row.iloc[range(0, len(row.index), 2)].astype(float),
                                    row.iloc[range(1, len(row.index), 2)].astype(float),
                                )
                            )
                        ],
                        dtype="float32",
                    )
                ),
                TRANSFORM_MAT,
            )[0, :, :]
            yield np.c_[
                xy[:, 0],
                xy[:, 1],
                [0.5] * (len(row) // 2),
                range(len(row) // 2),
            ]

    def update(self, i):
        """Update the scatter plot."""
        data = next(self.stream)

        # Set x and y data...
        self.scat.set_offsets(data[:, :2])
        # # Set sizes...
        # self.scat.set_sizes(300 * abs(data[:, 2])**1.5 + 100)
        # Set colors..
        self.scat.set_array(data[:, 3])

        # We need to return the updated "artist" for FuncAnimation to draw..
        # Note that it expects a sequence of "artists", thus the trailing comma.
        return (self.scat,)


if __name__ == "__main__":
    data = pd.read_csv("outputs/locations.csv", index_col=0)
    # data = pd.read_csv("./outputs/locations_limited.csv", index_col=0)
    a = AnimatedScatter(data)
    plt.show()
