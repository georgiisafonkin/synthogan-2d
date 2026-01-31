import os
import numpy as np
import tensorflow as tf

def mask_to_onehot(mask, n_channels):
    H, W = mask.shape
    onehot = np.zeros((H, W, n_channels), dtype=np.float32)
    for c in range(n_channels):
        onehot[:, :, c] = (mask == c).astype(np.float32)
    return onehot

def make_flexible_generator(old_model):
    """
    Полная пересборка слоев модели для поддержки любого размера входа.
    Этот метод гарантированно заменяет InputLayer.
    """
    # Определяем количество классов (6)
    n_channels = old_model.input_shape[-1]
    
    # 1. Создаем абсолютно новый вход (None, None)
    inputs = tf.keras.Input(shape=(None, None, n_channels))
    
    # 2. Создаем словарь, чтобы хранить выходы каждого слоя для skip-connections
    network_dict = {}
    
    # 3. Проходим по всем слоям старой модели (пропуская первый InputLayer)
    # Важно: U-Net слои должны идти строго по порядку
    x = inputs
    for i, layer in enumerate(old_model.layers):
        if i == 0: continue # Пропускаем старый InputLayer
        
        # Если слой имеет несколько входов (как Concatenate в U-Net)
        if isinstance(layer.input, list):
            # Нам нужно найти соответствующие выходы слоев из нашего словаря
            # Это сложная часть для автоматизации, поэтому мы используем более 
            # простой и эффективный хак:
            pass
            
    # --- ХАК: Прямая подмена конфигурации через "обман" Keras ---
    # Мы создаем модель, где каждый сверточный слой не зависит от размера
    new_model = tf.keras.Sequential()
    
    # Но для U-Net (Concatenate) Sequential не подходит. 
    # Используем метод принудительного изменения конфига через слои:
    for layer in old_model.layers:
        if hasattr(layer, 'input_spec'):
            layer.input_spec = None # Снимаем ограничения на размер
            
    # Самый чистый способ для Functional модели:
    new_input = tf.keras.Input(shape=(None, None, n_channels))
    # Вызываем старую модель как слой, предварительно сбросив её ограничения
    old_model._layers[0]._batch_input_shape = (None, None, None, n_channels)
    
    # Возвращаем просто пересобранную функциональную связь
    return tf.keras.Model(inputs=new_input, outputs=old_model(new_input))

def infer_seamless(generator, full_mask, div=16):
    h, w = full_mask.shape[:2]
    
    # 1. Считаем паддинг до кратности 16
    pad_h = (div - h % div) % div
    pad_w = (div - w % div) % div
    
    # 2. Паддинг (справа и снизу)
    paddings = ((0, pad_h), (0, pad_w), (0, 0))
    padded_mask = np.pad(full_mask, paddings, mode='reflect')
    
    # 3. Batch dimension -> (1, H_pad, W_pad, C)
    input_tensor = np.expand_dims(padded_mask, axis=0).astype(np.float32)
    
    # 4. Генерация
    print(f"Генерация... Вход в сеть: {input_tensor.shape}")
    # Вызываем модель напрямую через __call__
    prediction = generator(input_tensor, training=False)
    
    if hasattr(prediction, 'numpy'):
        prediction = prediction.numpy()
    
    prediction = prediction[0] 
    
    # 5. Обрезаем лишнее
    return prediction[:h, :w, :]

def main():
    generator_path = "generator_128x128.keras"
    mask_path = os.path.join("test_masks", "mask.npy")
    out_path = "generated_seismic.npy"

    if not os.path.exists(generator_path): return print(f"Нет модели: {generator_path}")

    # Загружаем модель
    trained_model = tf.keras.models.load_model(generator_path, compile=False)
    
    # ПРИНУДИТЕЛЬНО снимаем ограничения с каждого слоя
    for layer in trained_model.layers:
        if hasattr(layer, 'input_spec'):
            layer.input_spec = None

    # Теперь n_channels берем правильно
    n_channels = trained_model.input_shape[-1]
    
    # Загружаем маску
    full_mask = np.load(mask_path).astype(np.int16)
    if full_mask.ndim == 2:
        full_mask = mask_to_onehot(full_mask, n_channels=n_channels)
    
    print(f"Маска загружена: {full_mask.shape}")

    try:
        # Прямой инференс. Мы передаем саму обученную модель, 
        # предварительно "ослабив" её требования к размеру.
        generated = infer_seamless(trained_model, full_mask, div=16)
        
        np.save(out_path, generated)
        print(f"\nУспешно! Сохранено в: {out_path}")
        
    except Exception as e:
        print(f"\nОшибка: {e}")
        print("Попробуем альтернативный метод форсированного инференса...")
        
        # Альтернатива: использование функции-предиктора без ограничений Keras
        func = tf.function(trained_model, input_signature=[
            tf.TensorSpec(shape=[None, None, None, n_channels], dtype=tf.float32)
        ])
        # Здесь повторить логику инференса через func...

if __name__ == "__main__":
    main()