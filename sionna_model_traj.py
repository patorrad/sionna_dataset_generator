import os
gpu_num = "" # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sionna
from sionna.channel.utils import subcarrier_frequencies, cir_to_ofdm_channel
from sionna.ofdm import ResourceGrid

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

TX_POSITION = [10, 1, -2.]
CENTER_X, CENTER_Y, CENTER_Z = (27, 1, -3)
SCENE_NAME = "canyon"
IMAGE_FOLDER = f"images/{SCENE_NAME}"


# Import Sionna RT components
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera

import numpy as np

def dbm_to_watts(dbm):
    return 10. ** ((dbm-30)/10)

def watts_to_dbm(watts):
    epsilon = 1e-10  # Small constant to avoid log(0)
    return 10 * np.log10(watts + epsilon) + 30

def generate_positions(tx_position, rx_position, n):
    """
    Generate n positions between tx_position and rx_position.

    :param tx_position: List or array of the transmitter position [x, y, z]
    :param rx_position: List or array of the receiver position [x, y, z]
    :param n: Number of positions to generate
    :return: Array of generated positions
    """
    tx_position = np.array(tx_position)
    rx_position = np.array(rx_position)
    
    # Generate n+2 points including the endpoints
    positions = np.linspace(tx_position, rx_position, n + 2)
    
    return positions



trajectories = np.load("trajectories_2.npy")
positions = trajectories[0, :, :]
print(positions)
# import pdb; pdb.set_trace()

# For link-level simulations
# from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies, OFDMChannel, ApplyOFDMChannel, CIRDataset
# from sionna.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
# from sionna.utils import compute_ber, ebnodb2no, PlotBER
# from sionna.ofdm import KBestDetector, LinearDetector
# from sionna.mimo import StreamManagement

csi_readings = np.array([])

for i in range(len(positions)):
    start = time.perf_counter()
    scene = load_scene(f"models/{SCENE_NAME}.xml")
    scene.frequency = 2.462e9
    scene.objects['urban_canyon_take2_3_cropped_outliers_cropped_mesh'].material = "concrete"
    scene.tx_array = PlanarArray(num_rows=1,
                            num_cols=1, #2,
                            vertical_spacing=0.5, #0.7,
                            horizontal_spacing=0.5,
                            pattern="dipole", #"tr38901",
                            polarization="V") #"VH")

    # Configure antenna array for all receivers
    scene.rx_array = PlanarArray(num_rows=1,
                            num_cols=1,
                            vertical_spacing=0.5,
                            horizontal_spacing=0.5,
                            pattern="dipole",
                            polarization="V") #"cross")

    # Create transmitter
    tx = Transmitter(name="tx",
                position=TX_POSITION,
                orientation=[0,0,0])
    scene.add(tx)

    # Create a receiver
    rx = Receiver(name="rx",
            position=positions[i],
            orientation=[0,0,0])
    scene.add(rx)

    # TX points towards RX
    tx.look_at(rx)
    scene_loaded = time.perf_counter()
    print(f"scene load time:    {scene_loaded - start} seconds")

    start = time.perf_counter()

    # paths_simple = scene.compute_paths(
    #                             check_scene=False
    #                             )
    # paths_simple_time = time.perf_counter()
    # print(f"simple path compute time:       {paths_simple_time - start} seconds")

    # paths_diff = scene.compute_paths(
    #                             check_scene=False,
    #                             diffraction=True
    #                             )
    # paths_diff_time = time.perf_counter()
    # print(f"diffraction path compute time:  {paths_diff_time - paths_simple_time} seconds")


    # paths_scatter = scene.compute_paths(
    #                             check_scene=False,
    #                             scattering=True
    #                             )
    # paths_scatter_time = time.perf_counter()
    # print(f"scatter path compute time:      {paths_scatter_time - paths_diff_time} seconds")

    paths_complete = scene.compute_paths(
                                check_scene=False,
                                los=True,
                                reflection=True,
                                diffraction=True,
                                scattering=True
                                )
    paths_time = time.perf_counter()
    # print(f"path complete compute time:     {paths_time - paths_scatter_time} seconds")
    print(f"path complete compute time:     {paths_time - start} seconds")

    paths_complete.normalize_delays = False

    # Determine subcarrier frequencies
    rg = ResourceGrid(num_ofdm_symbols=1,
                    fft_size=52,
                    dc_null = True,
                    cyclic_prefix_length=20,
                    #   pilot_pattern = "kronecker",
                    #   pilot_ofdm_symbol_indices = [2, 8],
                    subcarrier_spacing=5e6) #30e3)

    frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)

    # Squeeze useless dimensions
    # [num_time_steps, fft_size]
    # h = tf.squeeze(cir_to_ofdm_channel(frequencies, *paths.cir(), normalize=True))

    # Reshape to [1, num_subcarriers]
    h_sim = tf.reshape(cir_to_ofdm_channel(frequencies, *paths_complete.cir()), [1, -1])

    # scene.preview(paths=paths_diff)

    resolution = [480*8,270*8]
    my_cam = Camera("my_cam", position=[CENTER_X - 20, CENTER_Y - 30, CENTER_Z + 35], look_at=[CENTER_X, CENTER_Y, CENTER_Z])
    scene.add(my_cam)
    scene.render_to_file(camera="my_cam",
                        filename=f"{IMAGE_FOLDER}/simple_paths_{i}.png",
                        resolution=resolution,
                        show_paths=True,
                        num_samples=16,
                        paths=paths_complete,
                        fov=90) # Render scene with paths to file

    csi_readings = np.append(csi_readings, watts_to_dbm(tf.abs(h_sim.numpy().flatten())))
    # rg.show()
    # plt.show()
    # plt.ylim([0, 30])
    plt.plot(np.arange(-int(frequencies.shape[0]/2), int(frequencies.shape[0]/2)), watts_to_dbm(tf.abs(h_sim.numpy().flatten()))) #   h_sim.numpy().flatten())), label="Simulated")
    plt.xlabel("Subcarrier index")
    plt.ylabel("Channel gain")
    plt.title("Channel frequency response")
    # # plt.plot(frequencies/1e6, tf.abs(h_sim.numpy().flatten()))
    plt.show()

np.save(f"csi_readings_{SCENE_NAME}.npy", csi_readings)
