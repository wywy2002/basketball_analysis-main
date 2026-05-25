import os
import sys
import pathlib
import numpy as np

folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path,"../"))
from utils import measure_distance

class PlayerKalmanFilter:
    def __init__(self, dt=1/30.0):
        self.x = np.zeros((4, 1))
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.P = np.eye(4) * 1000
        self.R = np.array([[1.0, 0], [0, 1.0]]) * 15.0
        self.Q = np.eye(4) * 0.1

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[:2].flatten()

    def update(self, measurement):
        z = np.array(measurement).reshape(2, 1)
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
        return self.x[:2].flatten()

class SpeedAndDistanceCalculator():
    def __init__(self, width_in_pixels, height_in_pixels, width_in_meters, height_in_meters):
        self.width_in_pixels = width_in_pixels
        self.height_in_pixels = height_in_pixels
        self.width_in_meters = width_in_meters
        self.height_in_meters = height_in_meters

    def calculate_distance(self, tactical_player_positions):
        previous_players_position = {}
        output_distances = []
        player_filters = {}
        for frame_number, tactical_player_position_frame in enumerate(tactical_player_positions):
            output_distances.append({})
            for player_id, current_player_position in tactical_player_position_frame.items():
                if player_id not in player_filters:
                    player_filters[player_id] = PlayerKalmanFilter()
                    player_filters[player_id].x[:2] = np.array(current_player_position).reshape(2, 1)
                player_filters[player_id].predict()
                filtered_position = player_filters[player_id].update(current_player_position).tolist()
                if player_id in previous_players_position:
                    meter_distance = self.calculate_meter_distance(previous_players_position[player_id], filtered_position)
                    output_distances[frame_number][player_id] = meter_distance
                previous_players_position[player_id] = filtered_position
        return output_distances

    def calculate_meter_distance(self, previous_pixel_position, current_pixel_position):
        p_x, p_y = previous_pixel_position
        c_x, c_y = current_pixel_position
        pm_x, pm_y = p_x * self.width_in_meters / self.width_in_pixels, p_y * self.height_in_meters / self.height_in_pixels
        cm_x, cm_y = c_x * self.width_in_meters / self.width_in_pixels, c_y * self.height_in_meters / self.height_in_pixels
        return measure_distance((cm_x, cm_y), (pm_x, pm_y)) * 0.4

    def calculate_speed(self, distances, fps=30):
        speeds = []
        window_size = 5
        for frame_idx in range(len(distances)):
            speeds.append({})
            for player_id in distances[frame_idx].keys():
                start_frame = max(0, frame_idx - (window_size * 3) + 1)
                total_distance, frames_present, last_f = 0, 0, None
                for i in range(start_frame, frame_idx + 1):
                    if player_id in distances[i]:
                        if last_f is not None:
                            total_distance += distances[i][player_id]
                            frames_present += 1
                        last_f = i
                if frames_present >= window_size:
                    time_h = (frames_present / fps) / 3600
                    speeds[frame_idx][player_id] = (total_distance / 1000) / time_h if time_h > 0 else 0
                else:
                    speeds[frame_idx][player_id] = 0
        return speeds