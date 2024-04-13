import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from matplotlib.backend_bases import MouseButton
from matplotlib.colors import Normalize
import tifffile as tiff


class IndexTracker(object):
    def __init__(self, ax, X, mask):
        self.ax = ax
        ax.set_title('Yeast cell image')

        self.X = X
        self.mask = mask
        self.frames, self.slices, self.rows, self.cols = X.shape
        self.frame_ind = self.frames // 2
        self.slice_ind = self.slices // 2
        self.mask_visible = True
        self.navigation_mode = 0 # 0: rows and columns, 1: rows and slices, 2: columns and slices

        self.im = ax.imshow(self.X[self.frame_ind, self.slice_ind, :, :], cmap="gray")
        self.mask_im = ax.imshow(self.mask[self.frame_ind, self.slice_ind, :, :], cmap="Reds", alpha=0.5)
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
            self.im.set_data(self.X[self.frame_ind, self.slice_ind, :, :])
            self.mask_im.set_data(self.mask[self.frame_ind, self.slice_ind, :, :])
            self.ax.set_ylabel(f'Frame {self.frame_ind}, Slice {self.slice_ind}')
        elif self.navigation_mode == 1:
            self.im.set_data(self.X[self.frame_ind, :, self.slice_ind, :])
            self.mask_im.set_data(self.mask[self.frame_ind, :, self.slice_ind, :])
            self.ax.set_ylabel(f'Frame {self.frame_ind}, Row {self.slice_ind}')
        elif self.navigation_mode == 2:
            self.im.set_data(self.X[self.frame_ind, :, :, self.slice_ind])
            self.mask_im.set_data(self.mask[self.frame_ind, :, :, self.slice_ind])
            self.ax.set_ylabel(f'Frame {self.frame_ind}, Column {self.slice_ind}')
        self.im.axes.figure.canvas.draw()

def plot4d(image, mask):
    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, image, mask)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    fig.canvas.mpl_connect('button_press_event', tracker.on_click)
    fig.canvas.mpl_connect('key_press_event', tracker.on_key) # Connect the keypress event
    plt.show()

if __name__ == "__main__":
    # Load the image and the mask
    image_filename = 'data/cell5.tif' # Replace with your image file name
    mask_filename = 'data/cell5_mask.tif' # Replace with your mask file name
    image = tiff.imread(image_filename)
    mask = tiff.imread(mask_filename)
    print(image.shape, mask.shape)
    print(image.dtype, mask.dtype)
    alpha_mask = Normalize(0, 255, clip=True)(mask)

    # Call the plot function
    test = 0
    plot4d(np.array(np.pad(image,[(0,test),(0,test),(0,test),(0,test)])), np.array(np.pad(mask,[(0,test),(0,test),(0,test),(0,test)])))
