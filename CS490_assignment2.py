#
# file   CS490_assignment2.py 
# brief  Purdue University Fall 2022 CS490 robotics Assignment2 - 
#        Motion model and landmark detection
# date   2022-08-18
#

from helper_functions import *
from matplotlib import pyplot as plt
import numpy as np


# robot robot_t class
class robot_robot_t:
    def __init__(self, x, y, theta):
        self.x_ = x
        self.y_ = y
        self.theta_ = theta
        self.lidar_ = None
        self.landmark_ = None
        self.pairs_ = None
        self.wall_pairs_ = None

    def add_lidar_scan(self, lidar):
        self.lidar_ = lidar

    def add_detected_landmark(self, landmark):
        self.landmark_ = landmark

    def add_landmark_pairs(self, pairs):
        self.pairs_ = pairs


# Question1
# ****************************************************************************************
def location_reader(filename=None):
    gt_location = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            x, y = map(float, line.strip().split())
            gt_location.append((x, y))
    return gt_location


def robot_motion_reader(filename=None):
    robot_motion = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            v, w = map(float, line.strip().split())
            robot_motion.append((v, w))
    return robot_motion


def motion_model_calculation(robot_motion):
    DT = 0.5
    robot_robot_t_list = []
    prev_x, prev_y, prev_theta = 1, 0, 0
    for (v, w) in robot_motion:
        robot_robot_t_list.append(robot_robot_t(prev_x, prev_y, prev_theta))  # store prev frame
        a = w * DT
        dist = 0
        R = 0
        if a != 0:
            R = v * DT / a
        dist = v * DT if R == 0 else R * math.sin(a / 2)
        dx = dist * math.cos(a / 2)
        dy = dist * math.sin(a / 2)
        cos_prev_theta = math.cos(prev_theta)
        sin_prev_theta = math.sin(prev_theta)
        prev_x += dx * cos_prev_theta - dy * sin_prev_theta
        prev_y += dx * sin_prev_theta + dy * cos_prev_theta
        prev_theta += a

    return robot_robot_t_list


# ****************************************************************************************

# Question 2
# ****************************************************************************************
def lidar_scan_reader(robot_robot_t_list, filename=None):
    data = np.loadtxt(filename)

    min_angle = data[:, 0]
    max_angle = data[:, 1]
    delta_angle = data[:, 2]
    lasers = data[:, 3:]

    # lasers[lasers == 10.0] = 3.5

    for i in range(len(robot_robot_t_list)):
        for laser_idx in range(len(lasers[i])):
            if lasers[i][laser_idx] == 10.0:
                lasers[i][laser_idx] = 3.5
        robot_robot_t_list[i].add_lidar_scan(
            (min_angle[i], max_angle[i], delta_angle[i], lasers[i].tolist())
        )
    return robot_robot_t_list


def calculate_angle(min_angle, delta_angle, n):
    return min_angle + delta_angle * n


def landmark_detection(robot_robot_t_list):
    VIS = True
    target_t = [1, 100, 400]
    D_THRESHOLD = 0.24
    OFFSET = 0.15
    for t, robot_t in enumerate(robot_robot_t_list):
        robot_x = robot_t.x_
        robot_y = robot_t.y_
        robot_theta = robot_t.theta_
        min_angle, max_angle, delta_angle, scans = robot_t.lidar_

        gradients = []
        drop_idx = []
        rise_idx = []
        for i in range(1, len(scans) - 1):
            l = scans[i - 1]
            r = scans[i + 1]
            if l > D_THRESHOLD and r > D_THRESHOLD:
                deri = (r - l) / 2.0
                gradients.append(deri)
                if deri <= -0.22:
                    drop_idx.append(i)
                elif deri >= 0.22:
                    rise_idx.append(i)
            else:
                gradients.append(0)

        if VIS and t in target_t:
            plt.plot(gradients)
            plt.title("Tick {} gradients".format(t))
            plt.show(block=True)

        drop_bound = len(drop_idx) - 1
        rise_bound = len(rise_idx) - 1
        landmarks = []
        while rise_bound >= 0 and drop_bound >= 0:
            if drop_idx[drop_bound] < rise_idx[rise_bound]:
                while rise_bound > 0 and drop_idx[drop_bound] < rise_idx[rise_bound - 1]:
                    rise_bound -= 1

                start = drop_idx[drop_bound]
                end = rise_idx[rise_bound] + 1

                start_angle = calculate_angle(min_angle, delta_angle, start)
                end_angle = calculate_angle(start_angle, delta_angle, end - start)
                avg_angle = start_angle + (end_angle - start_angle) / 2 # find middle

                avg_dist = sum(scans[j] + OFFSET for j in range(start, end)) / (
                            end - start)
                x = robot_t.x_ + avg_dist * math.cos(avg_angle + robot_theta)
                y = robot_t.y_ + avg_dist * math.sin(avg_angle + robot_theta)

                landmarks.append((x, y))
                rise_bound -= 1
            drop_bound -= 1

        if rise_bound >= 0:
            avg_angle = 0
            avg_dist = 0
            num_rays = len(scans) - drop_idx[-1] + rise_idx[0] + 1

            # barrier might be between 2pi and 0
            end_angles = np.array([calculate_angle(min_angle, delta_angle, i) for i in range(drop_idx[-1], len(scans))])
            start_angles = np.array([calculate_angle(min_angle, delta_angle, i) for i in range(rise_idx[0] + 1)])
            angles = np.concatenate((end_angles, start_angles))
            angles[angles > math.pi] -= 2 * math.pi

            end_dists = np.array([scans[i] for i in range(drop_idx[-1], len(scans))])
            start_dist = np.array([scans[i] for i in range(rise_idx[0] + 1)])
            dists = np.concatenate((end_dists, start_dist))

            avg_angle = np.mean(angles)
            avg_dist = np.mean(dists) + OFFSET

            x = robot_t.x_ + avg_dist * math.cos(avg_angle + robot_theta)
            y = robot_t.y_ + avg_dist * math.sin(avg_angle + robot_theta)
            landmarks.append((x, y))

        robot_t.add_detected_landmark(landmarks)

        if VIS and t in target_t:
            # also visualize the robot and lasers
            plt.scatter(robot_t.x_, robot_t.y_, s=20, marker='o', color='green')
            for i, laser in enumerate(scans):
                angle = min_angle + i * delta_angle + robot_t.theta_
                lx = robot_t.x_ + laser * math.cos(angle)
                ly = robot_t.y_ + laser * math.sin(angle)
                plt.scatter(lx, ly, color='blue', marker='o', s=3)
            x, y = zip(*landmarks)
            plt.scatter(x, y, s=10, color="red")
            plt.title("landmark at time step: {}".format(t))
            plt.xlim(0, 6)
            plt.ylim(3, -3)
            plt.show(block=True)


# ****************************************************************************************

# Question 3
# ****************************************************************************************
def pair_landmarks(robot_robot_t_list):
    VIS = True
    target_t = [1, 100, 400]
    for t, robot_t in enumerate(robot_robot_t_list):
        pairs = []
        for i, (landmark_x, landmark_y) in enumerate(robot_t.landmark_):
            closest = 0
            min_dist = math.sqrt(math.pow(landmark_x - gt_landmark[0][0], 2) + math.pow(landmark_y - gt_landmark[0][1], 2))
            for j, (gt_x, gt_y) in enumerate(gt_landmark):
                dist = math.sqrt(math.pow(landmark_x - gt_x, 2) + math.pow(landmark_y - gt_y, 2))
                if dist < min_dist:
                    min_dist = dist
                    closest = j
            if min_dist <= 1:
                pairs.append((i, closest))
        robot_t.add_landmark_pairs(pairs)
        if VIS and t in target_t:
            for pair in pairs:
                gt_x, gt_y = gt_landmark[pair[1]]
                landmark_x, landmark_y = robot_t.landmark_[pair[0]]
                plt.scatter([gt_x, landmark_x], [gt_y, landmark_y], s=10)
            plt.scatter(robot_t.x_, robot_t.y_, s=20, marker='o', color='green')
            plt.title("Tick {} landmark pairing".format(t))
            plt.show(block=True)



# ****************************************************************************************


if __name__ == '__main__':
    # you can add visualization functions but do not change the existsing code
    # please check what you need to implement for each function in the handout

    # Question1
    # ************************************************************************************
    gt_location = location_reader('location.txt')
    robot_motion = robot_motion_reader('robot_motion.txt')
    robot_robot_t_list = motion_model_calculation(robot_motion)
    # ************************************************************************************
    # draw_robot_trajectory(robot_robot_t_list, gt_location)

    # Question2
    # ************************************************************************************
    lidar_data = lidar_scan_reader(robot_robot_t_list, 'lidar_scan.txt')
    landmark_detection(robot_robot_t_list)
    # ************************************************************************************

    # Question3
    # ************************************************************************************
    pair_landmarks(robot_robot_t_list)
    # ************************************************************************************
