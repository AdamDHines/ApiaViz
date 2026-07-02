"""Route following navigation agent matching ant_route_following_test.m Section 3."""

import numpy as np
import torch
from tqdm import tqdm

from mbant.config import NetworkConfig, NavigationConfig, ImageConfig
from mbant.network import MushroomBodyNetwork
from mbant.renderer import render_panorama
from mbant.preprocessing import preprocess_single_for_network, to_torch_input


class NavigationAgent:
    """Agent that navigates a route using a trained mushroom body network.

    Implements the navigation test loop from ant_route_following_test.m (Section 3):
    1. Start at feeder location
    2. At each step: scan ±60° around current heading, select minimum EN response
    3. Move 0.1m forward in selected direction
    4. If >0.2m off-route, replace at nearest forward route point (count as error)
    5. Continue until within 0.2m of nest
    """

    def __init__(
        self,
        network: MushroomBodyNetwork,
        nav_config: NavigationConfig = None,
        img_config: ImageConfig = None,
    ):
        if nav_config is None:
            nav_config = NavigationConfig()
        if img_config is None:
            img_config = ImageConfig()

        self.network = network
        self.nav_config = nav_config
        self.img_config = img_config

    def navigate(
        self,
        img_pos: np.ndarray,
        heading: np.ndarray,
        world_data: dict,
        verbose: bool = True,
    ) -> dict:
        """Run the full navigation test.

        Args:
            img_pos: Training route positions, shape (numPos+1, 2) in meters.
            heading: Route headings, shape (numPos,) in degrees.
            world_data: Dict with 'X', 'Y', 'Z', 'colp' arrays.

        Returns:
            Navigation results dict with:
                - step_record: (3, num_steps) heading, EN response, scan type
                - current_position: (num_steps+1, 2) trajectory
                - error_rate: float
                - error_location: (num_errors, 2) error positions
                - EN_response: list of EN scan arrays per step
                - perf_measure: (route_length,) binary error flags
        """
        nav = self.nav_config
        X, Y, Z, colp = (
            world_data["X"],
            world_data["Y"],
            world_data["Z"],
            world_data["colp"],
        )

        numPos = len(heading)
        trained_route = img_pos.copy()
        route_length = int(np.ceil((len(trained_route) - 1) * 10))

        feeder = trained_route[0].copy()
        nest = trained_route[-1].copy()

        # Error tracking
        perf_measure = np.zeros(route_length)
        route_distance = np.zeros(numPos)
        error_location = []
        record_pos = []

        # Navigation state
        num_scan_img = nav.num_scan_img
        EN_pool = np.zeros(num_scan_img)
        step_record = np.zeros((3, route_length))
        current_position = np.zeros((route_length, 2))
        current_position[0] = feeder

        moving_direction = None
        current_pos = 0  # 0-indexed (MATLAB: current_pos = 1)

        EN_response_list = []
        step_count = 0  # 0-indexed (MATLAB: step_count = 1)

        iterator = range(route_length)
        if verbose:
            iterator = tqdm(iterator, desc="Navigation", unit="step")

        for _ in iterator:
            if moving_direction is None:
                # Naive scan centered on correct heading
                correct_heading = heading[current_pos]
                EN_scan = np.zeros(num_scan_img)

                for i_scan in range(num_scan_img):
                    temp_pos = current_position[step_count]
                    temp_heading = (
                        correct_heading
                        + nav.scan_range / 2
                        - i_scan * nav.scan_step
                    )
                    raw_img = render_panorama(
                        temp_pos[0], temp_pos[1], nav.eye_height,
                        temp_heading, X, Y, Z, colp,
                        nav.hfov, nav.resolution,
                    )
                    input_vec = preprocess_single_for_network(
                        raw_img, self.network.config.C_I_PN_var,
                        self.img_config.resize_shape,
                    )
                    input_tensor = to_torch_input(
                        input_vec, self.network.device
                    )
                    result = self.network.simulate(input_tensor, reward=0)
                    EN_count = result["spike_time_EN"].sum().item()
                    EN_scan[i_scan] = EN_count

                EN_response_list.append(EN_scan.copy())

                # Select heading with minimum EN response
                index = int(np.argmin(EN_scan))
                step_record[0, step_count] = (
                    correct_heading + nav.scan_range / 2 - index * nav.scan_step
                )
                step_record[1, step_count] = EN_scan[index]
                step_record[2, step_count] = 1  # naive scan

                # Move forward
                heading_rad = np.radians(step_record[0, step_count])
                moving_direction = np.array(
                    [np.cos(heading_rad), np.sin(heading_rad)]
                )
                current_position[step_count + 1] = (
                    current_position[step_count] + nav.step_size * moving_direction
                )
            else:
                # Normal scan centered on previous heading
                previous_heading = step_record[0, step_count - 1]
                step_record[2, step_count] = 1

                for i_scan in range(num_scan_img):
                    temp_heading = (
                        previous_heading
                        + nav.scan_range / 2
                        - nav.scan_step * i_scan
                    )
                    raw_img = render_panorama(
                        current_position[step_count, 0],
                        current_position[step_count, 1],
                        nav.eye_height, temp_heading,
                        X, Y, Z, colp, nav.hfov, nav.resolution,
                    )
                    input_vec = preprocess_single_for_network(
                        raw_img, self.network.config.C_I_PN_var,
                        self.img_config.resize_shape,
                    )
                    input_tensor = to_torch_input(
                        input_vec, self.network.device
                    )
                    result = self.network.simulate(input_tensor, reward=0)
                    EN_count = result["spike_time_EN"].sum().item()
                    EN_pool[i_scan] = EN_count

                EN_response_list.append(EN_pool.copy())

                index = int(np.argmin(EN_pool))
                step_record[1, step_count] = EN_pool[index]
                step_record[0, step_count] = (
                    previous_heading + nav.scan_range / 2 - nav.scan_step * index
                )

                heading_rad = np.radians(step_record[0, step_count])
                moving_direction = np.array(
                    [np.cos(heading_rad), np.sin(heading_rad)]
                )
                current_position[step_count + 1] = (
                    current_position[step_count] + nav.step_size * moving_direction
                )

            # Check if reached the nest
            current_distance = np.linalg.norm(
                nest - current_position[step_count + 1]
            )
            if current_distance <= nav.dis_threshold:
                break

            # Check distance to route
            for i in range(numPos):
                route_distance[i] = np.sqrt(
                    (current_position[step_count + 1, 0] - img_pos[i, 0]) ** 2
                    + (current_position[step_count + 1, 1] - img_pos[i, 1]) ** 2
                )

            dis_value = np.min(route_distance)
            ind_pos = int(np.argmin(route_distance))
            record_pos.append(ind_pos)

            # Check finish condition
            if ind_pos >= numPos - 1:
                break

            if dis_value > nav.dis_threshold:
                max_record = max(record_pos) if record_pos else 0

                if max_record < numPos - 2:
                    # Place back on route
                    if ind_pos < max_record:
                        next_pos = min(max_record + 1, numPos - 1)
                        current_position[step_count + 2] = img_pos[next_pos]
                        record_pos.append(max_record + 1)
                        current_pos = max_record + 1
                    else:
                        next_pos = ind_pos + round(nav.step_size * 10)
                        if next_pos >= numPos:
                            next_pos = numPos - 1
                        current_position[step_count + 2] = img_pos[next_pos]
                        record_pos.append(ind_pos + 1)
                        current_pos = ind_pos + 1

                    perf_measure[step_count] = 1
                    error_location.append(
                        current_position[step_count + 2].copy()
                    )
                    step_count += 2
                    moving_direction = None
                else:
                    break
            else:
                step_count += 1

        # Crop results
        current_position = current_position[: step_count + 2]
        step_record = step_record[:, : step_count + 2]

        error_rate = np.sum(perf_measure) / max(step_count, 1)
        change_angle = np.diff(heading)

        return {
            "step_record": step_record,
            "current_position": current_position,
            "dis_threshold": nav.dis_threshold,
            "error_rate": error_rate,
            "EN_response": EN_response_list,
            "error_location": np.array(error_location) if error_location else np.empty((0, 2)),
            "perf_measure": perf_measure,
            "change_angle": change_angle,
            "feeder": feeder,
            "nest": nest,
            "trained_route": trained_route,
        }
