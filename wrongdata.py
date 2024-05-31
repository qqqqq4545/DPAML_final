import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from sklearn.model_selection import train_test_split

# 載入數據
data = np.load('fake_lstm_data.npz')
X = data['X']
y_action = data['y_action']
y_pacifier = data['y_pacifier']
y_active = data['y_active']

# 分割訓練和測試數據
X_train, X_test, y_train_action, y_test_action = train_test_split(X, y_action, test_size=0.2, random_state=42)
_, _, y_train_pacifier, y_test_pacifier = train_test_split(X, y_pacifier, test_size=0.2, random_state=42)
_, _, y_train_active, y_test_active = train_test_split(X, y_active, test_size=0.2, random_state=42)

# 建立多輸出LSTM模型
input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
x = LSTM(50, activation='relu')(input_layer)

action_output = Dense(4, activation='softmax', name='action_output')(x)
pacifier_output = Dense(1, activation='sigmoid', name='pacifier_output')(x)
active_output = Dense(1, activation='sigmoid', name='active_output')(x)

model = Model(inputs=input_layer, outputs=[action_output, pacifier_output, active_output])

# 編譯模型
model.compile(optimizer='adam', 
              loss={'action_output': 'sparse_categorical_crossentropy', 
                    'pacifier_output': 'binary_crossentropy', 
                    'active_output': 'binary_crossentropy'}, 
              metrics={'action_output': 'accuracy', 
                       'pacifier_output': 'accuracy', 
                       'active_output': 'accuracy'})

model.summary()

# # 訓練模型
# model.fit(X_train, [y_train_action, y_train_pacifier, y_train_active], epochs=20, validation_data=(X_test, [y_test_action, y_test_pacifier, y_test_active]))

# # 保存模型
# model.save('multi_output_lstm_baby_monitor_model.h5')
