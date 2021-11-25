# IMPORT MODULES
import keras.backend
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Dense, GlobalAveragePooling2D, BatchNormalization, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.models import Model
from keras_applications.mobilenet_v3 import MobileNetV3Small

# -----------------------------1.客製化模型--------------------------------
# 載入keras模型(更換輸出類別數)
model = MobileNetV3Small(include_top=False,
                         weights='imagenet',
                         input_shape=(512, 512, 3),
                         backend=keras.backend,
                         layers=keras.layers,
                         models=keras.models,
                         utils=keras.utils
                         )

x = model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(6, activation='softmax', name='predictions')(x)
model = Model(input=model.input, outputs=predictions)

# 編譯模型
model.compile(optimizer=Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print('\n' + 'Your own model compiled')
print(model.summary())


# -----------------------------2.設置callbacks-----------------------------
# 設定earlystop條件
estop = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

# 設定模型儲存條件
checkpoint = ModelCheckpoint('MobileV3_checkpoint_v4.h5', verbose=1,
                             monitor='val_accuracy', save_best_only=True,
                             mode='max')

# 設定lr降低條件(0.001 → 0.0002 → 0.00004 → ......)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, mode='min', verbose=1,
                              min_lr=1e-5)


# -----------------------------3.設置訓練集--------------------------------
# 設定ImageDataGenerator參數(路徑、批量、圖片尺寸)
train_dir = './AOI/split_train_and_vali_balance/train/'
valid_dir = './AOI/split_train_and_vali_balance/vali/'
batch_size = 8
target_size = (512, 512)

# 設定批量生成器
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=15,
                                   width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.5, vertical_flip=True, horizontal_flip=True,
                                   fill_mode="nearest")
val_datagen = ImageDataGenerator(rescale=1./255)

# 讀取資料集+批量生成器，產生每epoch訓練樣本
train_generator = train_datagen.flow_from_directory(train_dir, target_size=target_size, batch_size=batch_size)
valid_generator = val_datagen.flow_from_directory(valid_dir, target_size=target_size, batch_size=batch_size)


# -----------------------------4.開始訓練模型------------------------------
# 重新訓練權重
history = model.fit_generator(train_generator,
                              epochs=500, verbose=1,
                              steps_per_epoch=train_generator.samples//batch_size,
                              validation_data=valid_generator,
                              validation_steps=valid_generator.samples//batch_size,
                              callbacks=[checkpoint, estop, reduce_lr])
                              # class_weight=class_weights)

# -----------------------------5.儲存模型，並記錄學習曲線------------------------------
# 儲存模型
model.save('./MobileV3_v4.h5')
print('已儲存MobileV3_v4.h5')

# 畫出學習曲線
acc = history.history['accuracy']
epochs = range(1, len(acc) + 1)
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend(loc='lower right')
plt.grid()
plt.savefig('./acc.png')
plt.show()

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc='upper right')
plt.grid()
plt.savefig('loss.png')
plt.show()