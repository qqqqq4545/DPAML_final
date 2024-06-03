import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, TimeDistributed, LSTM, Dense
from tensorflow.keras.models import Model

# 设置图像文件夹路径
image_dir = 'C:/baby_monitor/img'  # 请确保路径正确
image_size = (64, 64, 3)  # 确保包含通道数
categories = ['pacifier', 'cover']

# 检查目录和文件名
def check_directory(image_dir, categories):
    if os.path.exists(image_dir):
        print(f"Directory '{image_dir}' exists.")
        for category in categories:
            category_path = os.path.join(image_dir, category)
            if os.path.exists(category_path):
                print(f"Category directory '{category_path}' exists.")
                for img_name in os.listdir(category_path):
                    img_path = os.path.join(category_path, img_name)
                    print(f"File: {img_path}")
            else:
                print(f"Category directory '{category_path}' does not exist.")
    else:
        print(f"Directory '{image_dir}' does not exist.")

# 调用检查函数
check_directory(image_dir, categories)

# 创建数据集
def load_dataset(image_dir, image_size, categories):
    images = []
    labels = []
    for category in categories:
        category_path = os.path.join(image_dir, category)
        label = categories.index(category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            try:
                img = load_img(img_path, target_size=image_size[:2])
                img_array = img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(label)
            except PermissionError as e:
                print(f"PermissionError: {e}")
            except Exception as e:
                print(f"Error: {e}")
    return np.array(images), np.array(labels)

X, y = load_dataset(image_dir, image_size, categories)

# 将标签转换为one-hot编码
y = to_categorical(y, num_classes=len(categories))

# 分割训练和测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 确定输入形状
sequence_length = 1  # 单张图像
input_shape = (sequence_length, image_size[0], image_size[1], image_size[2])
input_layer = Input(shape=input_shape)

# 使用TimeDistributed包裹CNN部分
cnn = TimeDistributed(Conv2D(32, (3, 3), activation='relu'))(input_layer)
cnn = TimeDistributed(MaxPooling2D((2, 2)))(cnn)
cnn = TimeDistributed(Conv2D(64, (3, 3), activation='relu'))(cnn)
cnn = TimeDistributed(MaxPooling2D((2, 2)))(cnn)
cnn = TimeDistributed(Flatten())(cnn)

# LSTM部分
lstm = LSTM(50, activation='relu')(cnn)

# 输出层
output = Dense(len(categories), activation='softmax')(lstm)

model = Model(inputs=input_layer, outputs=output)

# 编译模型
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# 重新调整输入数据形状以适应LSTM模型
X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)

# 训练模型
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# 保存模型
model.save('lrcn_baby_monitor_model.h5')
