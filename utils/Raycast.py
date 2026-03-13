import numpy as np
import pybullet as p

class RaySensor:
    def __init__(self, num_rays=32, max_range=5.0, use_3d=False, z_levels=None):
        self.num_rays = num_rays
        self.max_range = max_range
        self.use_3d = use_3d

        # For 3D sensing, cast rings at multiple elevations.
        if z_levels is None:
            z_levels = [-0.3, 0.0, 0.3] if use_3d else [0.0]
        self.z_levels = z_levels

        self.local_dirs = self._build_local_directions()

    def _build_local_directions(self):
        dirs = []
        for z in self.z_levels:
            for i in range(self.num_rays):
                yaw = 2.0 * np.pi * i / self.num_rays
                d = np.array([np.cos(yaw), np.sin(yaw), z], dtype=np.float32)
                d = d / np.linalg.norm(d)
                dirs.append(d)
        return np.array(dirs, dtype=np.float32)  # shape: (K, 3)

    def get_observation(self, drone_id, client_id=0, visualize=False):
        pos, orn = p.getBasePositionAndOrientation(drone_id, physicsClientId=client_id)
        pos = np.array(pos, dtype=np.float32)

        # Rotation matrix from drone local frame -> world frame
        rot = np.array(p.getMatrixFromQuaternion(orn, physicsClientId=client_id),
                       dtype=np.float32).reshape(3, 3)

        world_dirs = (rot @ self.local_dirs.T).T  # (K, 3)

        # Small offset so rays don't start exactly inside the drone body
        ray_start_offset = 0.05
        ray_from = pos + ray_start_offset * world_dirs
        ray_to = pos + self.max_range * world_dirs

        results = p.rayTestBatch(
            ray_from.tolist(),
            ray_to.tolist(),
            physicsClientId=client_id
        )

        distances = []
        hits = []

        for i, r in enumerate(results):
            hit_body_uid = r[0]
            hit_fraction = r[2]
            hit_position = np.array(r[3], dtype=np.float32)

            if hit_body_uid == -1:
                dist = self.max_range
                hit = 0.0
                end_pt = ray_to[i]
            else:
                dist = hit_fraction * self.max_range
                hit = 1.0
                end_pt = hit_position

            distances.append(dist)
            hits.append(hit)

            if visualize:
                color = [1, 0, 0] if hit_body_uid != -1 else [0, 1, 0]
                p.addUserDebugLine(
                    ray_from[i],
                    end_pt,
                    color,
                    lifeTime=0.05,
                    physicsClientId=client_id
                )

        distances = np.array(distances, dtype=np.float32)
        hits = np.array(hits, dtype=np.float32)

        # Normalize distance to [0, 1]
        distances_norm = distances / self.max_range

        # Two common options:
        # 1) return only normalized distances
        # 2) return distances + hit mask
        obs = np.concatenate([distances_norm, hits], axis=0)
        return obs