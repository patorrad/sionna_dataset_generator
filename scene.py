import os
gpu_num = "" # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sionna

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

tf.random.set_seed(1) # Set global random seed for reproducibility

import matplotlib.pyplot as plt
import numpy as np
import time

TX_POSITION = [20, 1, -2.8]
RX_POSITION = [35, 1.5, -4.5]
CENTER_X, CENTER_Y, CENTER_Z = (27, 1, -3)
SCENE_NAME = "canyon"
IMAGE_FOLDER = f"images/{SCENE_NAME}"


resolution = [480*4,270*4] # increase for higher quality of renderings

# Allows to exit cell execution in Jupyter
class ExitCell(Exception):
    def _render_traceback_(self):
        pass


# Import Sionna RT components
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera

# For link-level simulations
from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies, OFDMChannel, ApplyOFDMChannel, CIRDataset
from sionna.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.utils import compute_ber, ebnodb2no, PlotBER
from sionna.ofdm import KBestDetector, LinearDetector
from sionna.mimo import StreamManagement

start = time.perf_counter()


scene_name = "canyon"
scene = load_scene(f"models/{scene_name}.xml")



# Create new camera with different configuration
my_cam = Camera("my_cam", position=[CENTER_X - 20, CENTER_Y - 30, CENTER_Z + 35], look_at=[CENTER_X, CENTER_Y, CENTER_Z])
scene.add(my_cam)

scene.tx_array = PlanarArray(num_rows=8,
                          num_cols=2,
                          vertical_spacing=0.7,
                          horizontal_spacing=0.5,
                          pattern="tr38901",
                          polarization="VH")

# Configure antenna array for all receivers
scene.rx_array = PlanarArray(num_rows=1,
                          num_cols=1,
                          vertical_spacing=0.5,
                          horizontal_spacing=0.5,
                          pattern="dipole",
                          polarization="cross")

# Create transmitter
tx = Transmitter(name="tx",
              position=TX_POSITION,
              orientation=[0,0,0])
scene.add(tx)

# Create a receiver
rx = Receiver(name="rx",
           position=RX_POSITION,
           orientation=[0,0,0])
scene.add(rx)

# TX points towards RX
tx.look_at(rx)



scene_loaded = time.perf_counter()
out(f"scene load time:                        {scene_loaded - start} seconds", f)

scene.render_to_file("my_cam", resolution=resolution, num_samples=512, filename=f"{IMAGE_FOLDER}/tx_rx.png")

scene_rendered = time.perf_counter()
out(f"scene render time:                      {scene_rendered - scene_loaded} seconds", f)

paths_simple = scene.compute_paths(
                            check_scene=False
                            )

paths_simple_computed = time.perf_counter()
out(f"path compute time:                      {paths_simple_computed - scene_rendered} seconds", f)

paths = scene.compute_paths(
                            check_scene=False,
                            diffraction=True,
                            scattering=True
                            )

paths_computed = time.perf_counter()
out(f"path compute time (diff and scatter):   {paths_computed - paths_simple_computed} seconds", f)




cm = scene.coverage_map(cm_cell_size=[1.,1.], # Configure size of each cell
                        cm_center=[CENTER_X, CENTER_Y, -3.0],
                        cm_orientation=[0,0,0],
                        cm_size=[20,20],
                        diffraction=True,
                        scattering=True,
                        num_samples=1e7)

map_computed = time.perf_counter()
out(f"coverage map compute time:              {map_computed - paths_computed} seconds", f)

# scene.render_to_file(camera="my_cam",
#                      filename=f"{IMAGE_FOLDER}/simple_paths.png",
#                      resolution=resolution,
#                      num_samples=512,
#                      paths=paths_simple) # Render scene with paths to file

# path_simple_rendered = time.perf_counter()
# out(f"path render time:                       {path_simple_rendered - map_computed} seconds", f)

# scene.render_to_file(camera="my_cam",
#                      filename=f"{IMAGE_FOLDER}/paths.png",
#                      resolution=resolution,
#                      num_samples=512,
#                      paths=paths) # Render scene with paths to file

# paths_rendered = time.perf_counter()
# out(f"path render time (diff and scatter):    {paths_rendered - path_simple_rendered} seconds", f)

# scene.render_to_file(camera="my_cam",
#                      filename=f"{IMAGE_FOLDER}/cm.png",
#                      resolution=resolution,
#                      num_samples=512,
#                      coverage_map=cm)

# map_rendered = time.perf_counter()
# out(f"coverage map render time:               {map_rendered - paths_rendered} seconds", f)

# out(f"total time:                             {map_rendered - start} seconds", f)
