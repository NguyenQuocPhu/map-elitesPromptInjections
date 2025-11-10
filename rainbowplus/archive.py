import numpy as np
from typing import Dict  
import random          
from copy import deepcopy

def cosine_distance_batch(X, Y):
    """Tính khoảng cách Cosine giữa hai mảng vector."""
    # 1 - (X . Y) / (||X|| * ||Y||)
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)
    
    # Tránh chia cho 0 nếu có vector zero
    X_norm[X_norm == 0] = 1e-12
    Y_norm[Y_norm == 0] = 1e-12
    
    dot_product = X @ Y.T
    similarity = dot_product / (X_norm @ Y_norm.T)
    
    # Đảm bảo similarity trong khoảng [-1, 1] do lỗi số thực
    similarity = np.clip(similarity, -1.0, 1.0)
    
    return 1.0 - similarity

def _compute_distance_to_centroids(b_vector, centroids):
    """Tìm centroid gần nhất cho một vector hành vi."""
    if len(centroids) == 0:
        return np.inf, -1
    
    # b_vector cần là 2D array để tính batch
    if b_vector.ndim == 1:
        b_vector = b_vector.reshape(1, -1)
        
    distances = cosine_distance_batch(b_vector, centroids)
    c_id = np.argmin(distances)
    return distances[0, c_id], c_id

class GrowingArchive:
    def __init__(self, n_cells: int,  n_behavior_dim: int, fitness_threshold: float):
        self.n_cells = n_cells
        self.fitness_threshold = fitness_threshold

        self.centroids = np.empty((n_cells, n_behavior_dim), dtype = np.float32)
        self.elites : Dict[int, Dict] = {}
        self.elites_backup : Dict[int, Dict] = {}
        self.n_centroids = 0
        self.dmin = np.inf
    def _compute_dmin(self):
        if self.n_centroids < 2:
            self.dmin = np.inf
            return
        active_centroids = self.centroids[:self.n_centroids]
        distances = cosine_distance_batch(active_centroids, active_centroids)

        np.fill_diagonal(distances, np.inf)
        self.dmin = np.min(distances)

        self.c_id_neighbors = np.argsort(distances, axis = 1)
        self.d_neighbors = np.array([distances[i][self.c_id_neighbors[i]] for i in range(self.n_centroids)])
    def _set_new_elite(self, cell_id: int, evaluation: Dict, is_backup: bool = True):
        self.elites[cell_id] = deepcopy(evaluation)
        if is_backup:
            self.elites_backup = deepcopy(evaluation)
    def __apply_repair(self, pruned_cell_id: int):
        active_centroids = self.centroids[:self.n_centroids]
        keys_to_check = list(self.elites.keys())
        if keys_to_check in active_centroids:
            keys_to_check.remove(pruned_cell_id)
        if not keys_to_check:
            return

        behaviors_to_check = np.array([self.elites[k]["behavior"]] for k in keys_to_check)
        distances = cosine_distance_batch(behaviors_to_check, active_centroids)
        new_cell_ids = np.argmin(distances, axis = 1)
        for i, old_cell_id in enumerate(keys_to_check):
            new_cell_id = new_cell_ids[i]

            if new_cell_id != old_cell_id:
                self.elites[old_cell_id] = deepcopy(self.elites_backup[old_cell_id])
    def add_evaluation(self, evaluation: Dict):
        if evaluation["fitness"] < self.fitness_threshold:
            return
        b_vector = evaluation["behavior"].reshape(1, -1)
        active_centroids = self.centroids[:self.n_centroids]

        if self.n_centroids < self.n_cells:
            new_cell_id = self.n_centroids
            self._set_new_elites(new_cell_id, evaluation, is_backup = True)
            self.n_centroids += 1
            if self.n_centroids == self.n_cells:
                self._compute_dmin()
            return
        
        d_to_nearest, cell_id = _compute_distance_to_centroids(b_vector, active_centroids)
        if d_to_nearest > self.dmin:
            centroid_A = np.argmin(self.d_neighbors[:, 0])
            centroid_B = self.c_id_neighbors[centroid_A, 0]

            dist_A_to_neighbor_2 = self.d_neighbors[centroid_A, 1]
            dist_B_to_neighbor_2 = self.d_neighbors[centroid_B, 1]

            if dist_A_to_neighbor_2 < dist_B_to_neighbor_2:
                pruned_cell_id = centroid_A
            else:
                pruned_cell_id = centroid_B
            self.centroids[pruned_cell_id] = b_vector
            self._set_new_elite(pruned_cell_id, evaluation, is_backup = True)
            
            self._compute_dmin()
            self.__apply_repair(pruned_cell_id)
            return
        
        if evaluation["fitness"] > self.elites[cell_id]["fitness"]:
            self._set_new_elite(cell_id, evaluation, is_backup = False)
    def sample_parent(self):
        if not self.elites:
            return None
        
        random_key = random.choice(list(self.elites.keys()))
        return self.elites[random_key]