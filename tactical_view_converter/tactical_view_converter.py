import os
import sys
import pathlib
import numpy as np
import cv2
from copy import deepcopy

folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path, "../"))
from utils import get_foot_position, measure_distance


class TacticalViewConverter:
    def __init__(self, court_image_path):
        self.court_image_path = court_image_path
        self.width = 300
        self.height = 161

        self.actual_width_in_meters = 28
        self.actual_height_in_meters = 15

        self.key_points = [
            (0, 0),
            (0, int((0.91 / self.actual_height_in_meters) * self.height)),
            (0, int((5.18 / self.actual_height_in_meters) * self.height)),
            (0, int((10 / self.actual_height_in_meters) * self.height)),
            (0, int((14.1 / self.actual_height_in_meters) * self.height)),
            (0, int(self.height)),
            (int(self.width / 2), self.height),
            (int(self.width / 2), 0),
            (int((5.79 / self.actual_width_in_meters) * self.width),
             int((5.18 / self.actual_height_in_meters) * self.height)),
            (int((5.79 / self.actual_width_in_meters) * self.width),
             int((10 / self.actual_height_in_meters) * self.height)),
            (self.width, int(self.height)),
            (self.width, int((14.1 / self.actual_height_in_meters) * self.height)),
            (self.width, int((10 / self.actual_height_in_meters) * self.height)),
            (self.width, int((5.18 / self.actual_height_in_meters) * self.height)),
            (self.width, int((0.91 / self.actual_height_in_meters) * self.height)),
            (self.width, 0),
            (int(((self.actual_width_in_meters - 5.79) / self.actual_width_in_meters) * self.width),
             int((5.18 / self.actual_height_in_meters) * self.height)),
            (int(((self.actual_width_in_meters - 5.79) / self.actual_width_in_meters) * self.width),
             int((10 / self.actual_height_in_meters) * self.height)),
        ]

        self.last_valid_homography_matrix = None

    def validate_keypoints(self, keypoints_list):
        keypoints_list = deepcopy(keypoints_list)

        for frame_idx, frame_keypoints in enumerate(keypoints_list):
            frame_keypoints_data = frame_keypoints.xy.tolist()[0]

            detected_indices = [i for i, kp in enumerate(frame_keypoints_data) if kp[0] > 0 and kp[1] > 0]
            if len(detected_indices) < 3:
                continue

            invalid_keypoints = []
            for i in detected_indices:
                if frame_keypoints_data[i][0] == 0 and frame_keypoints_data[i][1] == 0:
                    continue

                other_indices = [idx for idx in detected_indices if idx != i and idx not in invalid_keypoints]
                if len(other_indices) < 2:
                    continue

                j, k = other_indices[0], other_indices[1]

                d_ij = measure_distance(frame_keypoints_data[i], frame_keypoints_data[j])
                d_ik = measure_distance(frame_keypoints_data[i], frame_keypoints_data[k])

                t_ij = measure_distance(self.key_points[i], self.key_points[j])
                t_ik = measure_distance(self.key_points[i], self.key_points[k])

                if t_ij > 0 and t_ik > 0:
                    prop_detected = d_ij / d_ik if d_ik > 0 else float('inf')
                    prop_tactical = t_ij / t_ik if t_ik > 0 else float('inf')

                    error = abs((prop_detected - prop_tactical) / prop_tactical)

                    if error > 0.8:
                        keypoints_list[frame_idx].xy[0][i] *= 0
                        keypoints_list[frame_idx].xyn[0][i] *= 0
                        invalid_keypoints.append(i)

        return keypoints_list

    def transform_players_to_tactical_view(self, keypoints_list, player_tracks):
        tactical_player_positions = []

        for frame_idx, (frame_keypoints, frame_tracks) in enumerate(zip(keypoints_list, player_tracks)):
            tactical_positions = {}
            frame_keypoints_data = frame_keypoints.xy.tolist()[0]

            if frame_keypoints_data is None or len(frame_keypoints_data) == 0:
                tactical_player_positions.append(tactical_positions)
                continue

            valid_indices = [i for i, kp in enumerate(frame_keypoints_data) if kp[0] > 0 and kp[1] > 0]

            current_h_matrix = None

            if len(valid_indices) >= 4:
                source_points = np.array([frame_keypoints_data[i] for i in valid_indices], dtype=np.float32)
                target_points = np.array([self.key_points[i] for i in valid_indices], dtype=np.float32)

                try:
                    raw_h_matrix, status = cv2.findHomography(source_points, target_points, cv2.RANSAC, 5.0)

                    if raw_h_matrix is not None:
                        if self.last_valid_homography_matrix is not None:
                            test_pt = np.array([[[960.0, 540.0]]], dtype=np.float32)
                            pt_new = cv2.perspectiveTransform(test_pt, raw_h_matrix)[0][0]
                            pt_old = cv2.perspectiveTransform(test_pt, self.last_valid_homography_matrix)[0][0]
                            distance_diff = np.sqrt((pt_new[0] - pt_old[0]) ** 2 + (pt_new[1] - pt_old[1]) ** 2)

                            if distance_diff > self.width * 0.4:
                                current_h_matrix = self.last_valid_homography_matrix
                            else:
                                current_h_matrix = raw_h_matrix
                                self.last_valid_homography_matrix = raw_h_matrix
                        else:
                            current_h_matrix = raw_h_matrix
                            self.last_valid_homography_matrix = raw_h_matrix
                    else:
                        current_h_matrix = self.last_valid_homography_matrix
                except:
                    current_h_matrix = self.last_valid_homography_matrix
            else:
                current_h_matrix = self.last_valid_homography_matrix

            if current_h_matrix is None:
                tactical_player_positions.append(tactical_positions)
                continue

            for player_id, player_data in frame_tracks.items():
                bbox = player_data["bbox"]
                player_pos = np.array([[get_foot_position(bbox)]], dtype=np.float32)

                try:
                    transformed_pt = cv2.perspectiveTransform(player_pos, current_h_matrix)
                    tx, ty = transformed_pt[0][0]
                    if tx < -20 or tx > self.width + 20 or ty < -20 or ty > self.height + 20:
                        continue
                    tactical_positions[player_id] = [float(tx), float(ty)]
                except:
                    continue

            tactical_player_positions.append(tactical_positions)
        return tactical_player_positions