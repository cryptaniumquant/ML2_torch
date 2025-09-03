import numpy as np
import pandas as pd
import talib
import matplotlib.pyplot as plt
import os
import sys
from indicators_create4train_mat_MN_3_time import AiFeatures

def visualize_kama_labels(close, high, low, labels=None, W=60, thr=1.8, window_start=0, window_size=1000, title="KAMA Slope/ATR Labels"):
    """
    Визуализирует цены, KAMA и метки BUY/HOLD/SELL.
    
    :param close: Массив цен закрытия
    :param high: Массив максимальных цен
    :param low: Массив минимальных цен
    :param labels: Массив меток (если None, будут рассчитаны с помощью calculate_labels)
    :param W: Окно KAMA
    :param thr: Порог для ADX
    :param window_start: Начало окна для визуализации
    :param window_size: Размер окна для визуализации
    :param title: Заголовок графика
    """
    # Если метки не предоставлены, рассчитываем их
    if labels is None:
        # Реализуем вручную алгоритм расчета меток на основе угла наклона KAMA, нормированного ATR
        # Вычисляем KAMA с окном W
        kama = talib.KAMA(close, W)
        
        # Вычисляем наклон (slope) KAMA - приращение за один бар
        slope = np.diff(kama, prepend=kama[0])  # Используем numpy.diff и добавляем первый элемент для сохранения размера
        
        # Вычисляем ATR за то же окно
        atr = talib.ATR(high, low, close, W)
        
        # Вычисляем силу движения как отношение модуля наклона к ATR
        strength = np.abs(slope) / atr
        # Заменяем NaN и inf на 0
        strength = np.nan_to_num(strength, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Инициализируем метки как HOLD (1)
        labels = np.ones(len(close))
        
        # BUY: если сила > порог и наклон положительный
        labels[(strength > thr) & (slope > 0)] = 2
        
        # SELL: если сила > порог и наклон отрицательный
        labels[(strength > thr) & (slope < 0)] = 0
    
    # Проверяем распределение классов
    total_samples = len(labels)
    buy_count = np.sum(labels == 2)
    sell_count = np.sum(labels == 0)
    hold_count = np.sum(labels == 1)
    
    buy_percent = buy_count / total_samples * 100
    sell_percent = sell_count / total_samples * 100
    hold_percent = hold_count / total_samples * 100
    
    print(f"Распределение классов:")
    print(f"BUY:  {buy_count} ({buy_percent:.2f}%)")
    print(f"HOLD: {hold_count} ({hold_percent:.2f}%)")
    print(f"SELL: {sell_count} ({sell_percent:.2f}%)")
    
    # Проверяем, достаточно ли BUY и SELL меток
    if buy_percent < 10 or sell_percent < 10:
        print(f"ВНИМАНИЕ: Процент BUY ({buy_percent:.2f}%) или SELL ({sell_percent:.2f}%) меньше рекомендуемых 10%")
        print(f"Возможно, стоит скорректировать порог (текущий: {thr})")
    
    # Создаем массив индексов для отображения
    indices = np.arange(len(close))
    
    # Определяем окно для визуализации
    end_idx = min(window_start + window_size, len(close))
    window_indices = indices[window_start:end_idx]
    window_close = close[window_start:end_idx]
    window_labels = labels[window_start:end_idx]
    
    # Вычисляем KAMA для отображения
    kama = talib.KAMA(close, W)
    window_kama = kama[window_start:end_idx]
    
    # Создаем фигуру и оси
    plt.figure(figsize=(14, 8))
    
    # Строим цену закрытия
    plt.plot(window_indices, window_close, color='gray', alpha=0.5, label='Close Price')
    
    # Строим KAMA
    plt.plot(window_indices, window_kama, color='blue', linewidth=2, label=f'KAMA({W})')
    
    # Отмечаем точки в соответствии с метками
    buy_mask = window_labels == 2
    sell_mask = window_labels == 0
    
    if np.any(buy_mask):
        buy_indices = window_indices[buy_mask]
        plt.scatter(buy_indices, window_close[buy_mask], color='green', marker='^', s=100, label='BUY')
    
    if np.any(sell_mask):
        sell_indices = window_indices[sell_mask]
        plt.scatter(sell_indices, window_close[sell_mask], color='red', marker='v', s=100, label='SELL')
    
    # Добавляем заголовок и легенду
    plt.title(f"{title} (W={W}, thr={thr})")
    plt.xlabel('Индекс')
    plt.ylabel('Цена')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Добавляем аннотацию с распределением классов
    plt.figtext(0.01, 0.01, 
               f"BUY: {buy_percent:.2f}%, HOLD: {hold_percent:.2f}%, SELL: {sell_percent:.2f}%", 
               fontsize=10)
    
    # Показываем график
    plt.tight_layout()
    plt.savefig(f'kama_adx_labels_W{W}_thr{thr}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return labels

def load_data_and_visualize(file_path, W=60, thr=25, window_start=0, window_size=1000):
    """
    Загружает данные из CSV файла и визуализирует метки на основе KAMA.
    
    :param file_path: Путь к CSV файлу с данными
    :param W: Окно KAMA
    :param thr: Порог для ADX
    :param window_start: Начало окна для визуализации
    :param window_size: Размер окна для визуализации
    """
    # Загружаем данные
    print(f"Загрузка данных из {file_path}...")
    df = pd.read_csv(file_path)
    
    # Проверяем наличие необходимых столбцов
    required_columns = ['close', 'high', 'low']
    for col in required_columns:
        if col not in df.columns:
            # Пробуем найти столбцы с заглавной буквы
            cap_col = col.capitalize()
            if cap_col in df.columns:
                df[col] = df[cap_col]
            else:
                raise ValueError(f"В файле отсутствует столбец {col} или {cap_col}")
    
    # Преобразуем данные в numpy массивы
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    print(f"Загружено {len(close)} точек данных")
    
    # Визуализируем с разными параметрами
    print("\n1. Визуализация с параметрами по умолчанию:")
    visualize_kama_labels(close, high, low, W=W, thr=thr, window_start=window_start, window_size=window_size)
    
    # Попробуем разные пороги, если распределение классов не оптимально
    if thr == 1.8:  # Если используется порог по умолчанию
        print("\n2. Визуализация с пониженным порогом (thr=1.5):")
        visualize_kama_labels(close, high, low, W=W, thr=1.5, window_start=window_start, window_size=window_size)
        
        print("\n3. Визуализация с повышенным порогом (thr=2.0):")
        visualize_kama_labels(close, high, low, W=W, thr=2.0, window_start=window_start, window_size=window_size)

if __name__ == "__main__":
    # Пример использования
    file_path = 'data\SOLUSDT_1m_20220131_to_20250724.csv'  # Путь к вашему файлу с данными
    load_data_and_visualize(file_path, W=60, thr=0.11, window_start=0, window_size=5000)
