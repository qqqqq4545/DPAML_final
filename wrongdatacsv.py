import numpy as np
import pandas as pd

# 生成假數據
def generate_fake_data(num_samples=1000, sequence_length=10, image_size=(64, 64, 3)):
    X = np.random.rand(num_samples, sequence_length, image_size[0] * image_size[1] * image_size[2])
    y_action = np.random.randint(4, size=(num_samples, 1))  # 動作標記（0: climbing, 1: lay, 2: 正常, 3: 口鼻遮掩）
    y_pacifier = np.random.randint(2, size=(num_samples, 1))  # 遮掩物是否為奶嘴（0: 無遮掩或非奶嘴, 1: 奶嘴）
    y_active = np.random.randint(2, size=(num_samples, 1))  # 是否在活動（0: 靜止, 1: 活動）
    return X, y_action, y_pacifier, y_active

# 生成數據
X, y_action, y_pacifier, y_active = generate_fake_data()

# 保存數據到 numpy 文件
np.savez('fake_lstm_data.npz', X=X, y_action=y_action, y_pacifier=y_pacifier, y_active=y_active)
