import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from matplotlib.backend_bases import MouseButton
from matplotlib.colors import Normalize
import tifffile as tiff


class IndexTracker(object):
    def __init__(self, ax, X, y, y_pred):
        self.ax = ax
        ax.set_title('Yeast cell image')

        self.X = X
        self.y = y
        self.y_pred = y_pred
        self.frames, self.slices, self.rows, self.cols = y.shape
        self.frame_ind = self.frames // 2
        self.slice_ind = self.slices // 2
        self.mask_visible = True
        self.navigation_mode = 0 # 0: rows and columns, 1: rows and slices, 2: columns and slices

        self.im_X = ax.imshow(self.X[self.frame_ind, self.slice_ind, :, :], cmap="gray", alpha=0.8)
        self.im = ax.imshow(self.y[self.frame_ind, self.slice_ind, :, :], cmap="Purples", alpha=0.5)
        self.mask_im = ax.imshow(self.y_pred[self.frame_ind, self.slice_ind, :, :], cmap="Greens", alpha=0.5)
        self.update()

    def onscroll(self, event):
        if self.navigation_mode == 0:
            if event.button == 'up':
                self.slice_ind = (self.slice_ind + 1) % self.slices
            else:
                self.slice_ind = (self.slice_ind - 1) % self.slices
        elif self.navigation_mode == 1:
            if event.button == 'up':
                self.slice_ind = (self.slice_ind + 1) % self.rows
            else:
                self.slice_ind = (self.slice_ind - 1) % self.rows
        elif self.navigation_mode == 2:
            if event.button == 'up':
                self.slice_ind = (self.slice_ind + 1) % self.cols
            else:
                self.slice_ind = (self.slice_ind - 1) % self.cols
        self.update()

    def on_click(self, event):
        if event.button is MouseButton.LEFT:
            self.frame_ind = (self.frame_ind + 1) % self.frames
            self.update()
        elif event.button is MouseButton.RIGHT:
            self.frame_ind = (self.frame_ind - 1) % self.frames
            self.update()

    def on_key(self, event):
        if event.key == 'm':
            self.mask_visible = not self.mask_visible
            self.mask_im.set_visible(self.mask_visible)
            self.ax.figure.canvas.draw()
        elif event.key == 'd':
            self.navigation_mode = (self.navigation_mode + 1) % 3
            self.update()

    def update(self):
        if self.navigation_mode == 0:
            self.im_X.set_data(self.X[self.frame_ind, self.slice_ind, :, :])
            self.im.set_data(self.y[self.frame_ind, self.slice_ind, :, :])
            self.mask_im.set_data(self.y_pred[self.frame_ind, self.slice_ind, :, :])
            self.ax.set_ylabel(f'Frame {self.frame_ind}, Slice {self.slice_ind}')
        elif self.navigation_mode == 1:
            self.im_X.set_data(self.X[self.frame_ind, :, self.slice_ind, :])
            self.im.set_data(self.y[self.frame_ind, :, self.slice_ind, :])
            self.mask_im.set_data(self.y_pred[self.frame_ind, :, self.slice_ind, :])
            self.ax.set_ylabel(f'Frame {self.frame_ind}, Row {self.slice_ind}')
        elif self.navigation_mode == 2:
            self.im_X.set_data(self.X[self.frame_ind, :, :, self.slice_ind])
            self.im.set_data(self.y[self.frame_ind, :, :, self.slice_ind])
            self.mask_im.set_data(self.y_pred[self.frame_ind, :, :, self.slice_ind])
            self.ax.set_ylabel(f'Frame {self.frame_ind}, Column {self.slice_ind}')
        self.im_X.axes.figure.canvas.draw()
        self.im.axes.figure.canvas.draw()


def plot4d(X,y, y_pred):
    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, X, y, y_pred)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    fig.canvas.mpl_connect('button_press_event', tracker.on_click)
    fig.canvas.mpl_connect('key_press_event', tracker.on_key) # Connect the keypress event
    plt.show()

if __name__ == "__main__":
    # Load y and y_pred
    y_filename = 'CNN_data/y_test.npy' # Replace with your image file name
    y_pred_filename = 'CNN_data/y_pred.npy' # Replace with your mask file name
    X_filename = 'CNN_data/X_test.npy'
    X = np.load(X_filename)
    y = np.load(y_filename)
    y_pred = np.load(y_pred_filename)
    print(y.shape, y_pred.shape)
    print(y.dtype, y_pred.dtype)
    y_pred = np.reshape(y_pred,(y_pred.shape[0],y_pred.shape[1],y_pred.shape[2],y_pred.shape[3]))
    # Call the plot function
    plot4d(X[4:],y[4:], y_pred[4:])