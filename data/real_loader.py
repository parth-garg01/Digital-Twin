import numpy as np

class RealTrafficLoader:
    def __init__(self, data_path):
        
        data = np.load(data_path)

        print(data.files)  # debug

        if 'data' in data:
            self.X = data['data']

        elif 'arr_0' in data:
            self.X = data['arr_0']

        elif 'truth' in data:   # ✅ ADD THIS
            self.X = data['truth']

        else:
            raise ValueError("Unknown dataset format")
        self.X = self.X[:, :10, :]
        self.mean = self.X.mean()
        self.std = self.X.std() + 1e-6

        self.X = (self.X - self.mean) / self.std
     
        self.t = 0

    def get_next(self):
        if self.t >= len(self.X):
            self.t = 0
        
        step = self.X[self.t]
        self.t += 1
        
        # ✅ denormalize each node
        traffic = {
            f"T{i}": float(step[i][0] * self.std + self.mean)
            for i in range(len(step))
        }
        
        return traffic