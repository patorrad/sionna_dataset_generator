import numpy as np
import sionna
from sionna.channel.utils import subcarrier_frequencies, cir_to_ofdm_channel
from sionna.ofdm import ResourceGrid
from algorithim import Params, Algorithm, Capon
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera, AntennaArray, Antenna
import pandas as pd


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

CENTER_X, CENTER_Y, CENTER_Z = (27, 1, -3)
SCENE_NAME = "canyon"
IMAGE_FOLDER = f"images/{SCENE_NAME}"

def dbm_to_watts(dbm):
    return 10. ** ((dbm-30)/10)

def watts_to_dbm(watts):
    epsilon = 1e-10  # Small constant to avoid log(0)
    return 10 * np.log10(watts + epsilon) + 30

trajectories = np.load('trajectories.npy')
headings = np.load('headings.npy')

print(len(trajectories[0]))
TX_POSITION = np.array([CENTER_X, CENTER_Y, CENTER_Z])
data = []

for i in range(len(trajectories)):
    RX_POSITIONS = trajectories[i]
    hdg = headings[i]
    print("HDG: " + str(hdg))
    for j in range(len(RX_POSITIONS)):
        RX_POSITION = RX_POSITIONS[j]
        print("RX_POSITION " + str(RX_POSITION))
        scene = load_scene(f"models/{SCENE_NAME}.xml")
        scene.frequency = 2.462e9
        scene.objects['urban_canyon_take2_3_cropped_outliers_cropped_mesh'].material = "concrete"
        scene.synthetic_array = False
        
        RelativeAtennas = AntennaArray(antenna=Antenna("dipole", "V"), 
                            positions=tf.Variable([[0.0, 0.0, 0.0], [0.0, -0.06, 0.0], [0.0, -0.03, 0.0], [0.0, -0.09, 0.0]]))
        
        scene.tx_array = RelativeAtennas 
        scene.rx_array = RelativeAtennas

        # Create transmitter
        tx = Transmitter(name="tx",
                    position=TX_POSITION,
                    orientation=[0,0,0])
        scene.add(tx)

        # Create a receiver
        # changing the yaw of the reciever
        rx = Receiver(name="rx",
                position=RX_POSITION,
                orientation=[hdg,0,0])
        scene.add(rx)

        # TX points towards RX    theta_r = np.squeeze(paths.theta_r)

        tx.look_at(rx)

        paths_complete = scene.compute_paths(
                                    check_scene=False,
                                    los=True,
                                    reflection=True,
                                    diffraction=True,
                                    scattering=True
                                    )

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

        # Reshape to [1, num_subcarriers]
        h_sim = tf.reshape(cir_to_ofdm_channel(frequencies, *paths_complete.cir()), [1, -1])

        # Implementing Capon Algorithim
        params = Params(RelativeAtennas.positions[:, :2])
        # channel is 155, bandwidth is 80
        AOA=np.array([Capon(params, 155, 80, h_sim).evaluate()])
        print(AOA)
        csi_reading = watts_to_dbm(tf.abs(h_sim.numpy().flatten()))


        # data.append(np.array([TX_POSITION, RX_POSITION, AOA]))
        data.append([TX_POSITION, RX_POSITION, AOA, csi_reading])


        # Append the data as rows (each row contains TX, RX, AOA, and csi_reading)
        data.append([TX_POSITION[0], TX_POSITION[1], TX_POSITION[2],  # TX_POSITION X, Y, Z
                    RX_POSITION[0], RX_POSITION[1], RX_POSITION[2],  # RX_POSITION X, Y, Z
                    AOA[0], csi_reading])  # AOA, csi_reading (flattened scalars)

# Convert data to a Pandas DataFrame
df = pd.DataFrame(data, columns=['TX_X', 'TX_Y', 'TX_Z',  # Column names for TX_POSITION
                                'RX_X', 'RX_Y', 'RX_Z',  # Column names for RX_POSITION
                                'AOA', 'CSI_Reading'])  # Column names for AOA and csi_reading

# Display the DataFrame
# print(df)
# Save the DataFrame to a CSV file
df.to_csv('data.csv', index=False)  # index=False prevents saving the DataFrame index as a column