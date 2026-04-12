import numpy as np


class RealTrafficLoader:
    def __init__(self, data_path, num_traffic_nodes):
        data = np.load(data_path)
        self.data_key = self._resolve_key(data)
        traffic = self._extract_series(data[self.data_key])
        self.raw_traffic = traffic[:, :num_traffic_nodes].astype(np.float32)

        self.mean = self.raw_traffic.mean(axis=0, keepdims=True)
        self.std = self.raw_traffic.std(axis=0, keepdims=True) + 1e-6
        self.normalized_traffic = (self.raw_traffic - self.mean) / self.std

    def _resolve_key(self, data):
        for key in ("truth", "data", "arr_0", "prediction"):
            if key in data:
                return key
        raise ValueError("Unknown dataset format")

    def _extract_series(self, array):
        if array.ndim == 2:
            return array

        if array.ndim == 3:
            # METR-LA exports are often [horizon, time, nodes].
            if array.shape[0] <= 24 and array.shape[1] > array.shape[0]:
                return array[0]

            # Fallback for [time, nodes, features].
            if array.shape[-1] <= 4:
                return array[:, :, 0]

        raise ValueError(f"Unsupported traffic data shape: {array.shape}")

    def get_normalized_series(self):
        return self.normalized_traffic.copy()

    def get_raw_series(self):
        return self.raw_traffic.copy()

    def denormalize_traffic(self, values):
        return values * self.std.squeeze(0) + self.mean.squeeze(0)
