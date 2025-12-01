#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
from typing import Optional, Tuple, Set

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped

# --- 参数配置 ---
UNKNOWN = -1
FREE_THRESH = 65      
MIN_GOAL_DIST = 0.6         # 最小距离：离太近的不去（防止原地打转）
SCORE_DENSITY_WEIGHT = 0.5  # 密度权重：越高越喜欢去开阔的房间/路口
SAMPLE_SIZE = 50            # 从前沿点中采样多少个作为候选（减少计算量，增加随机性）

class FrontierExplorerSmooth(Node):
    def __init__(self):
        super().__init__('frontier_explorer_smooth')

        # QoS 配置 (关键：适配 SLAM 地图)
        map_qos = QoSProfile(
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, map_qos)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        self.navigator = BasicNavigator()
        self.get_logger().info('Nav2 Ready.')

        self.map: Optional[OccupancyGrid] = None
        self.odom_x: Optional[float] = None
        self.odom_y: Optional[float] = None

        self.pending = False
        self.last_goal_xy: Optional[Tuple[float, float]] = None
        self.blocked_goals: Set[Tuple[int, int]] = set() # 存像素坐标
        
        # 黑名单清理计数器 (每隔一段时间清空一次黑名单，防止死局)
        self.clear_blacklist_counter = 0

        # 定时器：1.0秒一次（频率不用太高，太高会让机器人动作抽搐）
        self.timer = self.create_timer(1.0, self.tick)
        self.get_logger().info('>>> Smooth Explorer Started (Density Based) <<<')

    def map_cb(self, msg: OccupancyGrid):
        if self.map is None:
             self.get_logger().info(f'Map received: {msg.info.width}x{msg.info.height}')
        self.map = msg

    def odom_cb(self, msg: Odometry):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y

    def tick(self):
        if self.map is None or self.odom_x is None:
            return

        # 1. 检查任务状态
        if self.pending:
            if self.navigator.isTaskComplete():
                result = self.navigator.getResult()
                self.get_logger().info(f'Nav Result: {result}')
                
                # 无论成功失败，都把目标附近区域加入黑名单，迫使机器人换地方
                if self.last_goal_xy:
                    self._add_to_blacklist_area(self.last_goal_xy)
                
                self.pending = False
                self.last_goal_xy = None
            else:
                # 【关键优化】如果正在走，且离目标还很远，不要频繁打断！
                # 这能极大解决“动作不平稳”的问题
                if self.last_goal_xy:
                    dx = self.last_goal_xy[0] - self.odom_x
                    dy = self.last_goal_xy[1] - self.odom_y
                    dist = math.hypot(dx, dy)
                    # 如果距离目标 > 0.5米，且 Nav2 还在正常跑，就让它跑，别打扰它
                    if dist > 0.5:
                        return 

        # 2. 处理地图数据
        info = self.map.info
        h, w = info.height, info.width
        grid = np.asarray(self.map.data, dtype=np.int8).reshape((h, w))

        # 3. 挑选最佳目标 (密度 + 距离)
        goal_node = self.pick_best_frontier(grid)
        
        if goal_node is None:
            self.get_logger().warn('No valid frontiers found. Exploring might be done.')
            # 可选：尝试清空黑名单重试
            if len(self.blocked_goals) > 0:
                 self.get_logger().info('Clearing blacklist to retry...')
                 self.blocked_goals.clear()
            return

        gx, gy = goal_node
        goal_pose = self.pixel_to_pose(gx, gy)
        
        # 记录浮点坐标用于计算距离
        self.last_goal_xy = (goal_pose.pose.position.x, goal_pose.pose.position.y)
        
        self.get_logger().info(f'>>> Heading to Room/Frontier: ({self.last_goal_xy[0]:.2f}, {self.last_goal_xy[1]:.2f})')
        self.navigator.goToPose(goal_pose)
        self.pending = True

    def pick_best_frontier(self, grid: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        算法核心：
        1. 找出所有边缘点。
        2. 随机采样 N 个点。
        3. 对每个点打分：Score = (周围前沿点数量 * 权重) - 距离。
        4. 选分数最高的。
        这样机器人会优先去“前沿密集”的地方（通常是房间入口或走廊尽头），而不是贴墙走。
        """
        h, w = grid.shape
        free = (grid >= 0) & (grid <= FREE_THRESH)
        unknown = (grid == UNKNOWN)

        # 找边缘 (Fast Numpy Roll)
        adj_unknown = (np.roll(unknown, 1, axis=0) | 
                       np.roll(unknown, -1, axis=0) | 
                       np.roll(unknown, 1, axis=1) | 
                       np.roll(unknown, -1, axis=1))
        
        frontier_mask = free & adj_unknown
        
        # 获取所有前沿点坐标
        cy, cx = np.where(frontier_mask)
        if cx.size == 0: return None

        # 机器人像素坐标
        info = self.map.info
        res = info.resolution
        origin_x = info.origin.position.x
        origin_y = info.origin.position.y
        robot_px = int((self.odom_x - origin_x) / res)
        robot_py = int((self.odom_y - origin_y) / res)

        # --- 第一次过滤：去掉太近的点 和 黑名单点 ---
        dx = cx - robot_px
        dy = cy - robot_py
        dist2 = dx*dx + dy*dy
        min_pix_sq = (MIN_GOAL_DIST / res) ** 2
        
        valid_indices = []
        for i in range(len(cx)):
            # 距离过滤
            if dist2[i] < min_pix_sq:
                continue
            # 黑名单过滤
            if (cx[i], cy[i]) in self.blocked_goals:
                continue
            valid_indices.append(i)
        
        if not valid_indices:
            return None
            
        cx = cx[valid_indices]
        cy = cy[valid_indices]
        
        # --- 核心：采样评分 (Sampling & Scoring) ---
        # 为了速度，不要计算所有点，随机选 SAMPLE_SIZE 个点来打分
        num_candidates = len(cx)
        if num_candidates > SAMPLE_SIZE:
            indices = np.random.choice(num_candidates, SAMPLE_SIZE, replace=False)
            sample_cx = cx[indices]
            sample_cy = cy[indices]
        else:
            sample_cx = cx
            sample_cy = cy

        best_score = -float('inf')
        best_goal = None

        # 半径：用来计算密度的范围 (比如 0.5米 内有多少个前沿点)
        radius_pix = int(0.5 / res) 

        for i in range(len(sample_cx)):
            tx, ty = sample_cx[i], sample_cy[i]
            
            # 1. 计算距离代价 (Cost)
            # 简单的曼哈顿距离比欧几里得快，够用了
            dist_robot = abs(tx - robot_px) + abs(ty - robot_py)
            
            # 2. 计算密度奖励 (Reward)
            # 统计该点周围 radius_pix 范围内有多少个 frontier 点
            # 利用 numpy 切片快速统计
            y_min = max(0, ty - radius_pix)
            y_max = min(h, ty + radius_pix)
            x_min = max(0, tx - radius_pix)
            x_max = min(w, tx + radius_pix)
            
            # 截取周围的小方块
            local_block = frontier_mask[y_min:y_max, x_min:x_max]
            density = np.count_nonzero(local_block)
            
            # 3. 综合打分
            # Score = (密度 * 权重) - (距离 * 1.0)
            # 距离单位是像素，密度也是像素数，需要调整权重
            score = (density * SCORE_DENSITY_WEIGHT * 5.0) - dist_robot
            
            if score > best_score:
                best_score = score
                best_goal = (tx, ty)

        return best_goal

    def _add_to_blacklist_area(self, goal_xy: Tuple[float, float]):
        """把物理坐标周围一圈加入黑名单"""
        info = self.map.info
        res = info.resolution
        gx = int((goal_xy[0] - info.origin.position.x) / res)
        gy = int((goal_xy[1] - info.origin.position.y) / res)
        
        # 封锁 3x3 或 5x5 区域，防止像素抖动重选
        radius = 2
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                self.blocked_goals.add((gx+dx, gy+dy))

    def pixel_to_pose(self, gx: int, gy: int) -> PoseStamped:
        info = self.map.info
        res = info.resolution
        x = info.origin.position.x + (gx + 0.5) * res
        y = info.origin.position.y + (gy + 0.5) * res

        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.orientation.w = 1.0
        return pose

def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorerSmooth()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()