import collections
import cv2
import numpy as np

class JointStateHistory:
    """
    A helper class for receivers to reassemble and work with joint state histories
    transmitted over ZMQ.
    """
    def __init__(self, maxlen=400, warn_on_discontinuity=True):
        # 400 states is roughly 4 seconds at 100Hz
        self.history = collections.deque(maxlen=maxlen)
        self.last_state_number = None
        self.warn_on_discontinuity = warn_on_discontinuity

    def add_states(self, states_list):
        """
        Adds a list of joint state dictionaries to the history buffer.
        Validates state_number sequence if available to check for gaps.
        """
        for state in states_list:
            state_num = state.get('state_number')
            if state_num is not None:
                if self.last_state_number is not None:
                    # Duplicate check
                    if state_num <= self.last_state_number:
                        continue
                    # Gap check
                    if state_num > self.last_state_number + 1:
                        if self.warn_on_discontinuity:
                            print(f"Warning: Discontinuity detected in joint states. Expected state {self.last_state_number + 1}, got {state_num}.")
                self.last_state_number = state_num
            self.history.append(state)

    def get_closest_state(self, target_timestamp, timestamp_key='monotonic_timestamp'):
        """
        Linear search to find the state whose timestamp is closest to target_timestamp.
        """
        if not self.history:
            return None
            
        min_diff = float('inf')
        closest = None
        
        for state in self.history:
            ts = state.get(timestamp_key)
            if ts is not None:
                diff = abs(ts - target_timestamp)
                if diff < min_diff:
                    min_diff = diff
                    closest = state
                    
        return closest

    def get_window(self, start_time, end_time, timestamp_key='monotonic_timestamp'):
        """
        Returns all states that fall within the [start_time, end_time] interval.
        """
        return [s for s in self.history if s.get(timestamp_key) is not None and start_time <= s[timestamp_key] <= end_time]

    def get_history_list(self):
        """Returns the current assembled history as a standard python list."""
        return list(self.history)


def draw_history_graphs(history, width=800, height=200):
    """
    Renders continuous 2D line plots of the gripper's pos_pct and effort
    for visualization of the sliding history window.
    """
    graph_img = np.zeros((height, width, 3), dtype=np.uint8)
    if len(history) < 2:
        return graph_img
        
    times = [s.get('monotonic_timestamp') for s in history]
    # Filter out any states without a monotonic timestamp
    valid_indices = [i for i, t in enumerate(times) if t is not None]
    if len(valid_indices) < 2:
        return graph_img
        
    times = [times[i] for i in valid_indices]
    pos = [history[i]['gripper']['pos_pct'] for i in valid_indices]
    eff = [history[i]['gripper']['effort'] for i in valid_indices]
    
    t0, t1 = times[0], times[-1]
    if t1 == t0: return graph_img
    
    def get_x(t):
        return min(width-1, max(0, int(((t - t0) / (t1 - t0)) * width)))
        
    def draw_curve(values, color, y_min, y_max, y_offset, h_scale, label):
        pts = []
        val_range = y_max - y_min
        if val_range == 0: val_range = 1.0
        
        for i in range(len(values)):
            x = get_x(times[i])
            normalized = (values[i] - y_min) / val_range
            y = min(y_offset + h_scale - 1, max(y_offset, int(y_offset + h_scale - (normalized * h_scale))))
            pts.append((x, y))
            
        pts = np.array(pts, np.int32)
        cv2.polylines(graph_img, [pts], False, color, 1, cv2.LINE_AA)
        
        # Draw background rect for text to make it readable
        text = f"{label} ({values[-1]:.1f})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(graph_img, (5, y_offset + 5), (5 + tw + 10, y_offset + 5 + th + 10), (0,0,0), -1)
        cv2.putText(graph_img, text, (10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # pos_pct usually 0 to 100
    pos_min, pos_max = 0.0, 100.0
    if pos:
        pm = max(pos)
        if pm > 100.0: pos_max = pm
    
    draw_curve(pos, (0, 255, 255), pos_min, pos_max, 0, height // 2, "pos_pct")
    
    # effort scale
    eff_min = min(-10, min(eff))
    eff_max = max(10, max(eff))
    draw_curve(eff, (255, 0, 255), eff_min, eff_max, height // 2, height // 2, "effort")
    
    zero_y = int(height//2 + (height//2) - ((0 - eff_min) / (eff_max - eff_min)) * (height//2))
    zero_y = min(height - 1, max(height // 2, zero_y))
    cv2.line(graph_img, (0, zero_y), (width, zero_y), (100, 100, 100), 1)
    
    return graph_img
