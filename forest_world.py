"""密林避障物理世界：边界、柱林、动态障碍与 2D LiDAR。"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple
import numpy as np
import pybullet as p


@dataclass
class ForestMapConfig:

    arena_size: float = 50.0
    wall_height: float = 3.0
    start_goal_min_dist: float = 7.0
    start_goal_max_dist: float = 10.0
    safe_zone_radius: float = 5.0
    flight_altitude: float = 1.0

    static_count: int = 0
    static_radius_min: float = 0.5
    static_radius_max: float = 1.5
    pillar_height: float = 3.0
    min_pillar_gap: float = 2.0

    dynamic_count: int = 0
    dynamic_radius: float = 1.0
    dynamic_mass: float = 500.0
    dynamic_speed: float = 1.5

    lidar_rays: int = 24
    lidar_max_range: float = 10.0
    lidar_z_offset: float = 0.0
    lidar_origin_offset: float = 0.15


@dataclass
class ForestWorld:
    client_id: int
    cfg: ForestMapConfig = field(default_factory=ForestMapConfig)

    start_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    goal_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))

    _wall_ids: List[int] = field(default_factory=list)
    _static_ids: List[int] = field(default_factory=list)
    _dynamic_ids: List[int] = field(default_factory=list)
    _pillar_records: List[Tuple[float, float, float]] = field(default_factory=list)

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        self._clear_scene()
        self._pillar_records.clear()
        self._build_walls()
        self._sample_start_goal()
        self._spawn_static_pillars()
        self._spawn_dynamic_pillars()
        return self.start_pos.copy(), self.goal_pos.copy()

    def scan_lidar(self, drone_pos: Sequence[float], drone_yaw: float, drone_id: int) -> np.ndarray:
        angles = np.linspace(0.0, 2.0 * math.pi, self.cfg.lidar_rays, endpoint=False) + drone_yaw
        offset = self.cfg.lidar_origin_offset
        ray_length = self.cfg.lidar_max_range - offset
        z = drone_pos[2] + self.cfg.lidar_z_offset

        origins, ends = [], []
        for ang in angles:
            dx, dy = math.cos(ang), math.sin(ang)
            origins.append([drone_pos[0] + offset * dx, drone_pos[1] + offset * dy, z])
            ends.append([drone_pos[0] + self.cfg.lidar_max_range * dx, drone_pos[1] + self.cfg.lidar_max_range * dy, z])

        results = p.rayTestBatch(origins, ends, physicsClientId=self.client_id)
        out = np.ones(self.cfg.lidar_rays, dtype=np.float32)
        for i, res in enumerate(results):
            hit_id, hit_frac = res[0], res[2]
            if hit_id != -1 and hit_id != drone_id:
                actual = hit_frac * ray_length + offset
                out[i] = actual / self.cfg.lidar_max_range
        return out

    def advance_dynamics(self) -> None:
        ground_z = self.cfg.pillar_height / 2.0
        for uid in self._dynamic_ids:
            vel, _ = p.getBaseVelocity(uid, physicsClientId=self.client_id)
            v_xy = np.array([vel[0], vel[1]], dtype=np.float64)
            speed = float(np.linalg.norm(v_xy))
            if speed < self.cfg.dynamic_speed * 0.9:
                if speed > 0.05:
                    v_xy = v_xy / speed * self.cfg.dynamic_speed
                else:
                    ang = np.random.uniform(0, 2 * math.pi)
                    v_xy = np.array([math.cos(ang), math.sin(ang)]) * self.cfg.dynamic_speed
                p.resetBaseVelocity(uid, linearVelocity=[float(v_xy[0]), float(v_xy[1]), 0.0], angularVelocity=[0.0, 0.0, 0.0], physicsClientId=self.client_id)
            pos, quat = p.getBasePositionAndOrientation(uid, physicsClientId=self.client_id)
            rpy = p.getEulerFromQuaternion(quat)
            if abs(pos[2] - ground_z) > 0.01 or max(abs(rpy[0]), abs(rpy[1])) > 0.01:
                p.resetBasePositionAndOrientation(uid, [pos[0], pos[1], ground_z], p.getQuaternionFromEuler([0.0, 0.0, 0.0]), physicsClientId=self.client_id)

    def set_difficulty(self, static_count: int, dynamic_count: int, max_goal_dist: float) -> None:
        self.cfg.static_count = static_count
        self.cfg.dynamic_count = dynamic_count
        self.cfg.start_goal_max_dist = max_goal_dist
        self.cfg.start_goal_min_dist = max_goal_dist * 0.7

    def _clear_scene(self) -> None:
        alive = {p.getBodyUniqueId(i, physicsClientId=self.client_id) for i in range(p.getNumBodies(self.client_id))}
        for uid in self._wall_ids + self._static_ids + self._dynamic_ids:
            if uid in alive:
                try:
                    p.removeBody(uid, physicsClientId=self.client_id)
                except Exception:
                    pass
        self._wall_ids.clear()
        self._static_ids.clear()
        self._dynamic_ids.clear()

    def _build_walls(self) -> None:
        half = self.cfg.arena_size / 2.0
        h = self.cfg.wall_height
        thick = 1.0
        specs = [
            ([0, half + thick / 2, h / 2], [half + thick, thick / 2, h / 2]),
            ([0, -half - thick / 2, h / 2], [half + thick, thick / 2, h / 2]),
            ([half + thick / 2, 0, h / 2], [thick / 2, half + thick, h / 2]),
            ([-half - thick / 2, 0, h / 2], [thick / 2, half + thick, h / 2]),
        ]
        for pos, ext in specs:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=ext, physicsClientId=self.client_id)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=ext, rgbaColor=[0.5, 0.5, 0.5, 0.8], physicsClientId=self.client_id)
            uid = p.createMultiBody(0, col, vis, basePosition=pos, physicsClientId=self.client_id)
            self._wall_ids.append(uid)

    def _sample_start_goal(self) -> None:
        bound = self.cfg.arena_size / 2.0 - self.cfg.safe_zone_radius
        while True:
            sx, sy = np.random.uniform(-bound, bound, 2)
            gx, gy = np.random.uniform(-bound, bound, 2)
            dist = math.hypot(gx - sx, gy - sy)
            if self.cfg.start_goal_min_dist <= dist <= self.cfg.start_goal_max_dist:
                self.start_pos = np.array([sx, sy, self.cfg.flight_altitude])
                self.goal_pos = np.array([gx, gy, self.cfg.flight_altitude])
                break

    def _valid_xy(self, x: float, y: float, radius: float) -> bool:
        if math.hypot(x - self.start_pos[0], y - self.start_pos[1]) < self.cfg.safe_zone_radius + radius:
            return False
        if math.hypot(x - self.goal_pos[0], y - self.goal_pos[1]) < self.cfg.safe_zone_radius + radius:
            return False
        bound = self.cfg.arena_size / 2.0
        if abs(x) + radius > bound or abs(y) + radius > bound:
            return False
        for cx, cy, cr in self._pillar_records:
            if math.hypot(x - cx, y - cy) < radius + cr + self.cfg.min_pillar_gap:
                return False
        return True

    def _create_pillar(self, x: float, y: float, radius: float, dynamic: bool) -> int:
        h = self.cfg.pillar_height
        mass = self.cfg.dynamic_mass if dynamic else 0.0
        color = [0.8, 0.2, 0.2, 1.0] if dynamic else [0.2, 0.6, 0.2, 1.0]
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=h, physicsClientId=self.client_id)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=h, rgbaColor=color, physicsClientId=self.client_id)
        uid = p.createMultiBody(mass, col, vis, basePosition=[x, y, h / 2], physicsClientId=self.client_id)
        return uid

    def _spawn_static_pillars(self) -> None:
        bound = self.cfg.arena_size / 2.0
        for _ in range(self.cfg.static_count):
            for _ in range(100):
                x, y = np.random.uniform(-bound, bound, 2)
                r = np.random.uniform(self.cfg.static_radius_min, self.cfg.static_radius_max)
                if self._valid_xy(x, y, r):
                    uid = self._create_pillar(x, y, r, dynamic=False)
                    self._static_ids.append(uid)
                    self._pillar_records.append((x, y, r))
                    break

    def _spawn_dynamic_pillars(self) -> None:
        sx, sy = self.start_pos[0], self.start_pos[1]
        gx, gy = self.goal_pos[0], self.goal_pos[1]
        for _ in range(self.cfg.dynamic_count):
            for _ in range(100):
                t = np.random.uniform(0.15, 0.85)
                x = sx + t * (gx - sx) + np.random.uniform(-5.0, 5.0)
                y = sy + t * (gy - sy) + np.random.uniform(-5.0, 5.0)
                r = self.cfg.dynamic_radius
                if self._valid_xy(x, y, r):
                    uid = self._create_pillar(x, y, r, dynamic=True)
                    p.changeDynamics(uid, -1, lateralFriction=0.0, spinningFriction=0.0, rollingFriction=0.0, restitution=1.0, linearDamping=0.0, angularDamping=0.0, physicsClientId=self.client_id)
                    ang = np.random.uniform(0, 2 * math.pi)
                    spd = self.cfg.dynamic_speed
                    p.resetBaseVelocity(uid, linearVelocity=[spd * math.cos(ang), spd * math.sin(ang), 0.0], physicsClientId=self.client_id)
                    self._dynamic_ids.append(uid)
                    self._pillar_records.append((x, y, r))
                    break
