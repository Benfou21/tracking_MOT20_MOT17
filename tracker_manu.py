import numpy as np
from scipy.optimize import linear_sum_assignment
import math



class Tracker:
    def __init__(self):
        self.center_points = {} 
        self.id_count = 0  

    def update(self, objects_rect):
        objects_bbs_ids = []
       
        detections = np.array([(x + w/2, y + h/2) for x, y, w, h in objects_rect])

        
        tracked_ids = list(self.center_points.keys())
        tracked_centers = np.array(list(self.center_points.values()))

        
        cost_matrix = np.zeros((len(detections), len(tracked_centers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(tracked_centers):
                cost_matrix[d, t] = np.linalg.norm(det - trk)

       
        row_inds, col_inds = linear_sum_assignment(cost_matrix)

        assigned_detections = set()
        assigned_tracks = set()

        for d, t in zip(row_inds, col_inds):
            if cost_matrix[d, t] < 35:  
                object_id = tracked_ids[t]
                self.center_points[object_id] = detections[d]  
                objects_bbs_ids.append([*objects_rect[d], object_id])
                assigned_detections.add(d)
                assigned_tracks.add(t)

        
        for d, det in enumerate(detections):
            if d not in assigned_detections:
                self.center_points[self.id_count] = det
                objects_bbs_ids.append([*objects_rect[d], self.id_count])
                self.id_count += 1

    
        self.center_points = {object_id: self.center_points[object_id] for t, object_id in enumerate(tracked_ids) if t in assigned_tracks}

        return objects_bbs_ids
