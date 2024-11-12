from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from typing_extensions import override

import tensorflow as tf
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

# from ..constants import C_SPEED, F0, USABLE_SUBCARRIER_FREQUENCIES

# if TYPE_CHECKING:
#     # avoids circular import
#     from .aoa_node import Params

SMALLEST_NORMAL = np.finfo(np.float64).smallest_normal

class Params:
    """Parameters for the AoaNode.

    Attributes:
        theta_min: Min value of theta/AoA samples (radians).
        theta_max: Max value of theta/AoA samples (radians).
        theta_count: Number of theta/AoA samples.
        tau_min: Min value of tau/ToF samples (seconds).
        tau_max: Max value of tau/ToF samples (seconds).
        tau_count: Number of tau/ToF samples.
        buffer_size: Size of circular CSI buffer.
        rate: Target processing/publish rate (Hz).
        algo: Name of the direction-finding algorithm to use.
        profiles: Which, if any, of 1D and 2D profiles to compute.
    """

    def __init__(self, rx_position):
        # algorithm
        self.theta_min = -1.56
        self.theta_max = 1.56
        self.theta_count = 360
        self.tau_min =  -10
        self.tau_max = 40
        self.tau_count = 100
        self.rx_position = np.array(rx_position)

        # <param name="algo" type="string" value="capon" />
        # <param name="profiles" type="int" value="1" />
        # <param name="rate" type="double" value="30" />
        # <param name="theta_min" type="double" value="-1.2217304764" />
        # <param name="theta_max" type="double" value="1.2217304764" />
        # <!-- <param name="theta_min" type="double" value="-1.6" />
        # <param name="theta_max" type="double" value="1.6" /> -->
        # <param name="theta_count" type="int" value="360" />
        # <param name="buffer_size" type="int" value="3" />
        # <rosparam param="rx_position"> [[0.0, 0.0], [0.0, -0.06], [0.0, -0.03], [0.0, -0.09]] </rosparam>

        # self.theta_min = rospy.get_param("~theta_min", -np.pi / 2)
        # self.theta_max = rospy.get_param("~theta_max", np.pi / 2)
        # self.theta_count = rospy.get_param("~theta_count", 180)
        # self.tau_min = rospy.get_param("~tau_min", -10)
        # self.tau_max = rospy.get_param("~tau_max", 40)
        # self.tau_count = rospy.get_param("~tau_count", 100)
        # self.buffer_size = rospy.get_param("~buffer_size", 20)
        # self.rx_position = np.array(rospy.get_param("~rx_position"))

        # ros
        # self.rate = rospy.get_param("~rate", 20)
        # self.algo = rospy.get_param("~algo", "capon")
        # self.profiles = rospy.get_param("~profiles", 3)


class Algorithm:
    """Angle of Arrival (AoA) and Time of Flight (ToF) extraction algorithm.

    Note that many aspects of the algorithm depend on the channel and bandwidth of the
    provided CSI matrices. Create a new algorithm instance for each channel/bandwidth
    combination. See the README for algorithm explanations. Variables are named to
    resemble the math.

    Attributes:
        name: the name of the algorithm. Used to match string to class.
        channel: the WiFi channel of the provided CSI matrices.
        bandwidth: the WiFi bandwidth of the provided CSI matrices.
        theta_samples: theta_count values between theta_min and theta_max in radians.
        tau_samples: tau_count values between tau_min and tau_max in nanoseconds.
    """

    name: ClassVar[str]

    # channel is 155, bandwidth is 80
    def __init__(self, params: Params, channel, bandwidth):
        self.C_SPEED = 299792458  # Speed of light in m/s
        self.F0 = 2.462e9  # Center frequency in Hz
        self.USABLE_SUBCARRIER_FREQUENCIES = 52  # Number of subcarriers
        self.params = params
        self.channel = channel
        self.bandwidth = bandwidth
        theta_min = self.params.theta_min
        theta_max = self.params.theta_max
        theta_count = self.params.theta_count
        self.theta_samples = np.linspace(theta_min, theta_max, theta_count)
        tau_min = self.params.tau_min
        tau_max = self.params.tau_max
        tau_count = self.params.tau_count
        self.tau_samples = np.linspace(tau_min, tau_max, tau_count)

    def csi_callback(self, new_csi: np.ndarray):
        """Processes a new CSI matrix.

        AoA and ToF extraction are separated from this callback function so the
        algorithms can be run at a different rate from the data collection rate.

        Args:
            new_csi: Complex matrix with shape (n_sub, n_rx, n_tx).
        """
        raise NotImplementedError()

    def evaluate(self) -> np.ndarray:
        """Extracts AoA and ToF information.

        Returns:
            A 2d profile with shape (theta_count, tau_count) where larger values
            correspond to increased likelihood that the AoA/ToF combination is correct.
        """
        raise NotImplementedError()

    def aoa_steering_vector(self):
        """Creates an AoA steering vector based on this instance's params.

        Returns:
            An ndarray with shape (n_rx, theta_count).
        """
        # calculate wave number
        k = 2 * np.pi * self.F0 / self.C_SPEED
        # expand dims so broadcasting works later
        dx, dy = np.expand_dims(self.params.rx_position.T, axis=2)
        # print("DX" + str(dx))
        # print("DY" + str(dy))
        # column j corresponds to steering vector for j'th theta sample
        A = np.repeat(np.expand_dims(self.theta_samples, axis=0), len(dx), axis=0)
        A = np.exp(1j * k * (dx * np.cos(A) + dy * np.sin(A)))
        # A now has shape (n_rx, theta_count)
        return A


class Capon(Algorithm):
    """Capon beamforming, also known as Minimum Variance Distortionless Response."""

    name = "capon"

    def __init__(self, params: Params, channel, bandwidth, csi):
        super().__init__(params, channel, bandwidth)
        # self.buffer = CircularBuffer(maxlen=params.buffer_size)
        self.A = self.aoa_steering_vector()  # (n_rx, theta_count)
        # print("A" + str(self.A))
        self.csi_data = None

        # Set dimensions based on message data
        self.n_sub = 52
        self.n_rx = 4
        self.n_tx = 4

        # Extract CSI data
        self.csi_data = csi

    def map_sigmoid(self, x, c=None, w=None):
        """Uses a sigmoid function to map x from (-infinity, infinity) to (0, 1).

        Params:
            x: The value or values to map.
            c: Center of the sigmoid (where it changes concavity). Optional.
            w: Width, or distance between the values mapped to 0.25 and 0.75. Optional.
        """
        if c is None:
            c = 0
        if w is None:
            w = 3
        return 1 / (1 + np.exp(-2 * np.log(3) / w * (x - c)))


    @override
    def evaluate(self):
        # C = np.swapaxes(self.buffer.asarray(), 0, 1)  # (n_rx, n_sub, k)
        # R = np.cov(np.reshape(C, (self.n_rx, -1), order="F"))
        
        self.csi_data = np.reshape(self.csi_data, (self.n_rx, -1), order="F")
        R = np.cov(self.csi_data)

        # use least_squares solve for stability when computing profile
        profile = np.zeros(self.params.theta_count)
        for i in range(self.params.theta_count):
            a = self.A[:, i]  # (n_rx)
            x, *_ = np.linalg.lstsq(R, a, rcond=None)
            profile[i] = 1 / np.real(a.conj().T @ x)
        
        profile_data = np.reshape(np.atleast_2d(profile), (self.params.theta_count, 1))

        # the average directivity of the array is ideally the isotropic gain
        isotropic = np.mean(profile_data)
        # convert directivity of array into dBi
        profile_dBi = 10 * np.log10(profile_data / isotropic)
        # map dBi to [0, 1), prioritizing content around 0 dBi
        profile_norm = self.map_sigmoid(profile_dBi)

        # Create radial plot
        # plt.figure(figsize=(8, 8))
        # ax = plt.subplot(111, projection='polar')

        # # Plot the evaluation profile
        # ax.plot(self.theta_samples, profile_norm, label="Evaluation Profile")


        # # Add labels and legend
        # ax.set_title("Evaluation Profile in Radial Plot", va='bottom')
        # ax.set_theta_zero_location("N")  # Zero angle at the top (North)
        # ax.set_theta_direction(-1)  # Clockwise angle direction

        # # Show plot
        # plt.legend()
        # plt.show()
        
        Evaluation = np.array([profile_norm]) # truncated for brevity
        Evaluation = Evaluation.flatten()  # Flatten to a 1D array if it's 2D

        # Find the index of the maximum value
        max_index = np.argmax(Evaluation)

        # Get the corresponding angle
        max_angle = self.theta_samples[max_index]

        # return in radians but can do np.degrees(max_angle) to convert to degrees
        return max_angle
