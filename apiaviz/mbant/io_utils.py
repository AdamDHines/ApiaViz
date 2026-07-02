"""I/O utilities for loading .mat data files."""

import numpy as np
import scipy.io as sio


def load_ant_data(filepath: str = "./antview/AntData.mat") -> dict:
    """Load ant route data from AntData.mat.

    Returns:
        Dictionary mapping ant name (e.g., 'Ant1') to route data.
        Each ant entry contains:
            - 'routes': dict mapping route name (e.g., 'Route1') to route data
            - 'available_routes': sorted list of available route numbers
    """
    mat = sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)

    ants = {}
    for i in range(1, 16):
        key = f"Ant{i}"
        if key in mat:
            ant_struct = mat[key]
            inward_route_data = ant_struct.InwardRouteData
            routes = {}
            available_routes = []
            for attr in dir(inward_route_data):
                if not attr.startswith("Route"):
                    continue
                route_num = int(attr[5:])
                route = getattr(inward_route_data, attr)
                control_points = np.array(route.One_cm_control_points)
                # Some .mat files may not have headings
                headings = None
                if hasattr(route, "One_cm_control_points_headings"):
                    headings = np.array(route.One_cm_control_points_headings)
                routes[attr] = {
                    "control_points": control_points,
                    "headings": headings,
                }
                available_routes.append(route_num)
            ants[key] = {
                "routes": routes,
                "available_routes": sorted(available_routes),
            }
    return ants


def load_world_data(filepath: str = "./antview/world5000_gray.mat") -> dict:
    """Load world geometry from world5000_gray.mat.

    Returns:
        Dictionary with:
            - 'X': (numPolygons, verticesPerPolygon) x-coordinates
            - 'Y': (numPolygons, verticesPerPolygon) y-coordinates
            - 'Z': (numPolygons, verticesPerPolygon) z-coordinates
            - 'colp': (numPolygons, verticesPerPolygon) color values
    """
    mat = sio.loadmat(filepath)
    return {
        "X": np.array(mat["X"], dtype=np.float64),
        "Y": np.array(mat["Y"], dtype=np.float64),
        "Z": np.array(mat["Z"], dtype=np.float64),
        "colp": np.array(mat["colp"], dtype=np.float64),
    }


def prepare_route(
    ant_data: dict,
    img_separation: int = 10,
    step_size: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Prepare a route for training: compute positions, headings, and adjust positions.

    Matches ant_route_following_test.m lines 35-68:
        1. Convert from cm to m
        2. Sample every img_separation cm
        3. Compute headings between consecutive positions
        4. Round headings to nearest even integer
        5. Recompute positions based on adjusted headings

    Args:
        ant_data: Dict with 'control_points' (N, 2) in cm.
        img_separation: Sampling interval in cm.
        step_size: Step size in meters.

    Returns:
        (img_pos, heading, numPos) where:
            img_pos: (numPos+1, 2) positions in meters
            heading: (numPos,) headings in degrees
            numPos: number of training images
    """
    temp_route = ant_data["control_points"] / 100.0  # cm -> m

    img_limit = (len(temp_route) // img_separation) * img_separation
    img_pos = temp_route[0:img_limit:img_separation, :].copy()
    numPos = len(img_pos) - 1

    # Compute headings
    heading = np.zeros(numPos)
    for i in range(numPos):
        heading[i] = np.degrees(
            np.arctan2(
                img_pos[i + 1, 1] - img_pos[i, 1],
                img_pos[i + 1, 0] - img_pos[i, 0],
            )
        )

    # Round to nearest even integer
    # MATLAB: if mod(floor(heading), 2) == 0 → floor, else → ceil
    for i in range(numPos):
        floor_h = np.floor(heading[i])
        if int(floor_h) % 2 == 0:
            heading[i] = floor_h
        else:
            heading[i] = np.ceil(heading[i])

    # Recompute positions based on adjusted headings
    for i in range(numPos):
        img_pos[i + 1, :] = img_pos[i, :] + np.array(
            [np.cos(np.radians(heading[i])), np.sin(np.radians(heading[i]))]
        ) * step_size

    return img_pos, heading, numPos
