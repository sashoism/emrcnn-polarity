# %%
import argparse
import os
import shutil
import re

# %%
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("dest_dir", default="./dataset", type=str, nargs="?", help="output directory")
parser.add_argument("--data-dir", default="./orig", type=str, help="directory containing .tif volumes")
parser.add_argument("--keep", default=1.0, type=float, help="percentage of volumes to keep")
parser.add_argument("--train-test-split", default=0.75, type=float, help="percentage of training samples")
parser.add_argument("--no-pad", dest="pad", action="store_false", default=True, help="disable padding")
parser.add_argument("--pad-shape", default=[64, 64, 64], type=int, nargs=3, metavar=('z', 'y', 'x'), help="target shape when padding")
parser.add_argument("--random-seed", default=None, type=int, help="seed for np.random.seed()")
parser.add_argument("--num-frames", default=1, type=int, help="number of frames to sample from each cell")
parser.add_argument("--no-weighted-sampling", dest="weighted_sampling", action="store_false", default=True, help="disable weighted frame sampling based on mask volume")
parser.add_argument("--no-smoothed-weights", dest="smoothed_weights", action="store_false", default=True, help="disable smoothing for weighted frame sampling")
args = parser.parse_args()

dest_dir = args.dest_dir
data_dir = args.data_dir
keep = args.keep
train_test_split = args.train_test_split
pad = args.pad
pad_shape = tuple(args.pad_shape)
random_seed = args.random_seed
num_frames = args.num_frames
weighted_sampling = args.weighted_sampling
smoothed_weights = args.smoothed_weights

# %%
import numpy as np
import skimage.io as io
from tqdm import tqdm

# %%
if not os.path.exists(data_dir):
  raise FileNotFoundError(f"The directory '{data_dir}' does not exist.")
elif not os.path.isdir(data_dir):
  raise NotADirectoryError(f"'{data_dir}' is not a directory.")

if os.path.exists(dest_dir):
  shutil.rmtree(dest_dir)

for split in ("train", "test"):
  for type in ("real", "gt"):
    os.makedirs(os.path.join(dest_dir, split, type))

# %%
ids = list(set(
  re.search(r".+\d+", os.path.basename(fp)).group()
  for fp in os.listdir(data_dir)
  if fp.endswith('.tif')
))

if random_seed is not None:
  np.random.seed(random_seed)

ids = sorted(
  np.random.choice(ids, size=int(keep * len(ids)), replace=False),
  key=lambda id: int(re.search(r"\d+", id).group())
)

try:
  with tqdm(ids) as bar:
    for id in bar:
      data = io.imread(os.path.join(data_dir, f"{id}.tif"))
      mask = io.imread(os.path.join(data_dir, f"{id}_mask.tif"))

      # gimp the data to 8-bit :(
      data = (data * np.iinfo(np.uint8).max / data.max()).astype(np.uint8)

      if not np.count_nonzero(mask):
        tqdm.write(f"{id}_mask contains no annotations, skipping...")
        continue

      if pad:
        if any(data_dim > target_dim for (data_dim, target_dim) in zip(data.shape[1:], pad_shape)):
          tqdm.write(f"{id}.shape={data.shape[1:]} overflows pad_shape={pad_shape}, skipping...")
          continue

        padding = (
          (0, 0),
          *(
            ((target_dim - data_dim) // 2,
             (target_dim - data_dim) // 2 + (target_dim - data_dim) % 2)
            for (data_dim, target_dim) in zip(data.shape[1:], pad_shape)
          ),
        )
        data, mask = np.pad(data, padding), np.pad(mask, padding)

      # sample <num_frames> frames (with preference for larger masks)
      mask_sizes = np.count_nonzero(mask, axis=(1, 2, 3)) + (1 if smoothed_weights else 0)
      weights = mask_sizes / mask_sizes.sum() if weighted_sampling else None
      selected_frames = np.random.choice(len(data), size=num_frames, p=weights, replace=False)

      # assign split for the (TZYX) volume
      split = "train" if np.random.rand() < train_test_split else "test"

      for frame in selected_frames:
        for z in range(data.shape[1]):
          filename = f"{id:03}_z{z:03}_frame{frame:03}.png"
          bar.set_postfix_str(filename)
          io.imsave(os.path.join(dest_dir, split, "real", filename), data[frame, z, :], check_contrast=False)
          io.imsave(os.path.join(dest_dir, split, "gt", filename), mask[frame, z, :], check_contrast=False)
except KeyboardInterrupt:
  shutil.rmtree(dest_dir)
