import talib
import numpy as np
import pandas as pd
import time
from typing import Optional
from datetime import datetime
from numba import njit, prange, float64, int64
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import bisect
import pickle
import json
import os
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


class AiFeatures:
    def __init__(self):
        # Количество баров, когда все индикаторы посчитаны и готовы к работе
        self.firstValidValue: int = 0

        # Списки с ценовыми данными
        self.Open: Optional[np.ndarray] = None
        self.High: Optional[np.ndarray] = None
        self.Low: Optional[np.ndarray] = None
        self.Close: Optional[np.ndarray] = None
        self.Volume: Optional[np.ndarray] = None

        # Индикаторы

        #region Трендовость (3 шт)
        # Calculate SMA for different periods
        self.SMA_60: Optional[np.ndarray] = None  # SMA с периодом 60 минут
        self.SMA_240: Optional[np.ndarray] = None  # SMA с периодом 240 минут (4 часа)
        self.SMA_1440: Optional[np.ndarray] = None  # SMA с периодом 1440 минут (1 день)
        self.ROC_10: Optional[np.ndarray] = None
        self.ROC_40: Optional[np.ndarray] = None
        self.ROC_240: Optional[np.ndarray] = None
        #endregion

        #region Angles
        # Calculate angles (3 шт.) between current point and different past periods
        self.ANGLE_SMA_60_100: Optional[np.ndarray] = None  # Угол наклона SMA с периодом 60 минут измеренный по отношению к 100 барам назад
        self.ANGLE_SMA_240_400: Optional[np.ndarray] = None  # Угол наклона SMA с периодом 240 минут измеренный по отношению к 400 барам назад
        self.ANGLE_SMA_1440_2400: Optional[np.ndarray] = None  # Угол наклона SMA с периодом 1440 минут измеренный по отношению к 2400 барам назад
        #endregion

        #region ROC
        # Calculate (19 шт.) price percent increments (RocP100) for each Fibonacci period
        # Короткосрочные периоды (9 шт.): от минуты до 55 минут
        self.ROCP100_Fib_1: Optional[np.ndarray] = None  # На сколько % изменилась цена по отношению к цене 1 бар назад (около 1 минуты)
        self.ROCP100_Fib_2: Optional[np.ndarray] = None  # На сколько процентов изменилась цена по отношению к цене 2 бара назад (около 2 минуты)
        self.ROCP100_Fib_3: Optional[np.ndarray] = None  # На сколько процентов изменилась цена по отношению к цене 3 бара назад (около 3 минуты)
        self.ROCP100_Fib_5: Optional[np.ndarray] = None  # На сколько процентов изменилась цена по отношению к цене 5 баров назад (около 5 минут)
        self.ROCP100_Fib_8: Optional[np.ndarray] = None  # На сколько процентов изменилась цена по отношению к цене 8 баров назад (около 8 минут)
        self.ROCP100_Fib_13: Optional[np.ndarray] = None  # На сколько процентов изменилась цена по отношению к цене 13 баров назад (около 13 минут)
        self.ROCP100_Fib_21: Optional[np.ndarray] = None  # На сколько процентов изменилась цена по отношению к цене 21 бара назад (около 21 минуты)
        self.ROCP100_Fib_34: Optional[np.ndarray] = None  # На сколько процентов изменилась цена по отношению к цене 34 бара назад (около 34 минуты)
        self.ROCP100_Fib_55: Optional[np.ndarray] = None  # На сколько процентов изменилась цена по отношению к цене 55 баров назад (около 55 минут)

        # Среднесрочные периоды (6 шт.): от часа до 16 часов
        self.ROCP100_Fib_89: Optional[np.ndarray] = None  # На сколько процентов изменилась цена по отношению к цене 89 баров назад (около 1 час 29 минут)
        self.ROCP100_Fib_144: Optional[np.ndarray] = None  # На сколько процентов изменилась цена по отношению к цене 144 бара назад (около 2 часа 24 минуты)
        self.ROCP100_Fib_233: Optional[np.ndarray] = None  # На сколько процентов изменилась цена по отношению к цене 233 бара назад (около 3 часа 53 минуты)
        self.ROCP100_Fib_377: Optional[np.ndarray] = None  # На сколько процентов изменилась цена по отношению к цене 377 баров назад (около 6 часов 17 минут)
        self.ROCP100_Fib_610: Optional[np.ndarray] = None  # На сколько процентов изменилась цена по отношению к цене 610 баров назад (около 10 часов 10 минут)
        self.ROCP100_Fib_987: Optional[np.ndarray] = None  # На сколько процентов изменилась цена по отношению к цене 987 баров назад (около 16 часов 27 минут)

        # Долгосрочные периоды (4 шт.) от дня до 5 дней
        self.ROCP100_Fib_1597: Optional[np.ndarray] = None  # На сколько процентов изменилась цена по отношению к цене 1597 баров назад (около 1 день 4 часа)
        self.ROCP100_Fib_2584: Optional[np.ndarray] = None  # На сколько процентов изменилась цена по отношению к цене 2584 бара назад (около 1 день 19 часов)
        self.ROCP100_Fib_4181: Optional[np.ndarray] = None  # На сколько процентов изменилась цена по отношению к цене 4181 бара назад (около 2 дня 22 часа)
        self.ROCP100_Fib_6765: Optional[np.ndarray] = None  # На сколько процентов изменилась цена по отношению к цене 6765 баров назад (около 4 дня 13 часов)
        #endregion

        #region ATR
        # Calculate ATR for different periods
        self.ATR_14: Optional[np.ndarray] = None  # ATR с периодом 14
        self.ATR_64: Optional[np.ndarray] = None  # ATR с периодом 64
        self.ATR_384: Optional[np.ndarray] = None  # ATR с периодом 384
        #endregion

        #region StDevPct
        # Calculate StDevPct for different periods
        self.StDevPct_60: Optional[np.ndarray] = None  # StDevPct с периодом 60
        self.StDevPct_240: Optional[np.ndarray] = None  # StDevPct с периодом 240
        self.StDevPct_1440: Optional[np.ndarray] = None  # StDevPct с периодом 1440
        #endregion

        #region AdOsc
        # Calculate (3 шт.) AdOsc for different periods
        self.AdOsc_14: Optional[np.ndarray] = None  # AdOsc с периодом 14
        self.AdOsc_64: Optional[np.ndarray] = None  # AdOsc с периодом 64
        self.AdOsc_384: Optional[np.ndarray] = None  # AdOsc с периодом 384
        #endregion

        #region Rsi
        # Calculate (3 шт.) Rsi for different periods
        self.Rsi_14: Optional[np.ndarray] = None  # Rsi с периодом 14
        self.Rsi_64: Optional[np.ndarray] = None  # Rsi с периодом 64
        self.Rsi_384: Optional[np.ndarray] = None  # Rsi с периодом 384
        #endregion

        #region StochF_K
        # Calculate (3 шт.) StochF_K for different periods
        self.StochF_K_14: Optional[np.ndarray] = None  # StochF_K с периодом 14
        self.StochF_K_64: Optional[np.ndarray] = None  # StochF_K с периодом 64
        self.StochF_K_384: Optional[np.ndarray] = None  # StochF_K с периодом 384
        #endregion

        #region VolatilityPercentage
        # Calculate (1 шт.) VolatilityPercentage for different periods
        self.VolatilityPercentage: Optional[np.ndarray] = None  # VolatilityPercentage
        #endregion

        #region VolumeRatio
        # Calculate (1 шт.) VolumeRatio for different periods
        self.VolumeRatio: Optional[np.ndarray] = None  # VolumeRatio
        #endregion

    @staticmethod
    def Sma(prices: np.ndarray, period: int) -> np.ndarray:
        """
        Вычисляет простую скользящую среднюю (SMA) для заданного периода.

        :param prices: Массив цен.
        :param period: Период для расчета SMA.
        :return: Массив значений SMA.
        """
        # Convert input array to np.float64 (double) type to avoid TA-Lib errors
        prices_double = prices.astype(np.float64)
        return talib.SMA(prices_double, period)

    @staticmethod
    def ROC(prices: np.ndarray, period: int) -> np.ndarray:
        """
        Вычисляет простую скользящую среднюю (SMA) для заданного периода.

        :param prices: Массив цен.
        :param period: Период для расчета SMA.
        :return: Массив значений SMA.
        """
        return talib.ROC(prices, period)

    @staticmethod
    def CalculateAngle(priceChange: float, period: int) -> float:
        """
        Вычисляет угол наклона на основе изменения цены и периода.
        Угол наклона в диапазоне от -90 до 90 градусов позволяет легко интерпретировать направление изменения цены.
        Положительные углы (от 0 до 90 градусов) указывают на рост цены.
        Отрицательные углы (от 0 до -90 градусов) указывают на снижение цены.
        Угол 0 градусов указывает на отсутствие изменения цены.

        :param priceChange: Изменение цены.
        :param period: Период, за который изменилась цена.
        :return: Угол наклона в градусах.
        :raises ValueError: Если период меньше или равен нулю.
        """
        # Проверка на корректность периода
        if period <= 0:
            raise ValueError("Период должен быть положительным числом.")

        # Вычисляем угол наклона в радианах
        angleInRadians = np.arctan(priceChange / period)

        # Преобразуем угол в градусы
        angleInDegrees = angleInRadians * (180.0 / np.pi)

        # Ограничиваем угол значениями от -90 до 90, чтобы избежать некорректных интерпретаций
        if angleInDegrees < -90:
            angleInDegrees = -90
        if angleInDegrees > 90:
            angleInDegrees = 90

        # Возвращаем угол наклона в градусах
        return angleInDegrees

    @staticmethod
    def CalculateAngleArray(prices: Optional[np.ndarray], period: int, usePctChange: bool = True) -> Optional[np.ndarray]:
        """
        Вычисляет угол наклона между текущей точкой и точкой, сдвинутой на заданный период,
        используя либо процентное изменение, либо изменение в пунктах.
        Угол наклона в диапазоне от -90 до 90 градусов позволяет легко интерпретировать направление изменения цены.
        Положительные углы (от 0 до 90 градусов) указывают на рост цены.
        Отрицательные углы (от 0 до -90 градусов) указывают на снижение цены.
        Угол 0 градусов указывает на отсутствие изменения цены.

        :param prices: Массив цен, для которых вычисляется угол наклона.
        :param period: Период, на который сдвигаются цены для вычисления предыдущих значений.
        :param usePctChange: Флаг, указывающий, использовать ли процентное изменение цены (true) или изменение в пунктах (false). По умолчанию true.
        :return: Массив углов наклона в градусах.
        :raises ValueError: Если период меньше или равен нулю.
        :raises ValueError: Если массив предыдущих цен является None или содержит нулевые значения.
        """
        # Проверка на null для массива цен
        if prices is None:
            return None

        # Проверка на корректность периода
        if period <= 0:
            raise ValueError("Период должен быть положительным числом.")

        changes: Optional[np.ndarray]

        # Выбираем метод вычисления изменения цены в зависимости от параметра usePctChange
        if usePctChange:
            # Вычисляем процентное изменение цен
            changes = AiFeatures.RocP100(prices, period)
        else:
            # Вычисляем изменение цены в пунктах
            changes = AiFeatures.PriceChangePunkt(prices, period)

        # Проверка на null для массива изменений
        if changes is None:
            raise ValueError("Не удалось получить массив изменений цен.")

        # Вычисляем угол наклона для каждого изменения
        angles = np.array([AiFeatures.CalculateAngle(change, period) for change in changes])

        return angles

    @staticmethod
    def PriceChangePunkt(prices: Optional[np.ndarray], period: int) -> Optional[np.ndarray]:
        """
        Вычисляет изменение цены в пунктах по сравнению с ценой несколько баров назад.
        Формула: price - prevPrice (Индикаторы импульса)

        :param prices: Массив цен, для которых вычисляется изменение в пунктах.
        :param period: Период, на который сдвигаются цены для вычисления предыдущих значений.
        :return: Массив изменений цен в пунктах.
        :raises ValueError: Если период меньше или равен нулю.
        :raises ValueError: Если массив предыдущих цен является None или содержит нулевые значения.
        """
        # Проверка на null для массива цен
        if prices is None:
            return None

        # Проверка на корректность периода
        if period <= 0:
            raise ValueError("Период должен быть положительным числом.")

        # Получаем массив предыдущих цен, сдвинутых на заданный период
        prevPrices = np.roll(prices, period)

        # Проверка на null для массива предыдущих цен
        if prevPrices is None:
            raise ValueError("Не удалось получить массив предыдущих цен.")

        # Вычисляем изменение цены в пунктах для каждой цены
        priceChangePunkt = prices - prevPrices

        # Возвращаем массив изменений цен в пунктах
        return priceChangePunkt

    @staticmethod
    def RocP100(prices: Optional[np.ndarray], period: int) -> Optional[np.ndarray]:
        """
        Вычисляет процентное изменение цены (Rate of Change Percentage) относительно предыдущей цены.
        Формула: (price - prevPrice) / prevPrice * 100.0 (Индикаторы импульса)

        :param prices: Массив цен, для которых вычисляется процентное изменение.
        :param period: Период, на который сдвигаются цены для вычисления предыдущих значений.
        :return: Массив процентных изменений цен.
        :raises ValueError: Если период меньше или равен нулю.
        :raises ValueError: Если массив предыдущих цен является None или содержит нулевые значения.
        """
        # Проверка на null для массива цен
        if prices is None:
            return None

        # Проверка на корректность периода
        if period <= 0:
            raise ValueError("Период должен быть положительным числом.")

        # Получаем массив предыдущих цен, сдвинутых на заданный период
        prevPrices = np.roll(prices, period)

        # Проверка на null для массива предыдущих цен
        if prevPrices is None:
            raise ValueError("Не удалось получить массив предыдущих цен.")

        # Вычисляем процентное изменение для каждой цены
        with np.errstate(divide='ignore', invalid='ignore'):
            rocP100 = np.where(prevPrices != 0, (prices - prevPrices) / prevPrices * 100.0, 0.0)

        # Возвращаем массив процентных изменений
        return rocP100

    @staticmethod
    def Atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """
        Вычисляет ATR (Average True Range) для заданного периода.

        :param high: Массив высоких цен.
        :param low: Массив низких цен.
        :param close: Массив закрывающих цен.
        :param period: Период для расчета ATR.
        :return: Массив значений ATR.
        """
        return talib.ATR(high, low, close, timeperiod=period)*100/close

    @staticmethod
    def StDevPct(prices: np.ndarray, period: int) -> np.ndarray:
        """
        Вычисляет стандартное отклонение в процентах для заданного периода.

        :param prices: Массив цен.
        :param period: Период для расчета стандартного отклонения.
        :return: Массив значений стандартного отклонения в процентах.
        """
        return talib.STDDEV(prices, timeperiod=period)*100/prices

    @staticmethod
    def AdOsc(high: np.ndarray, low: np.ndarray, close: np.ndarray,
              volume: np.ndarray,  fast_period: int, slowperiod:int) -> np.ndarray:
        """
        Вычисляет AdOsc (Average Directional Oscillator) для заданного периода.

        :param high: Массив высоких цен.
        :param low: Массив низких цен.
        :param close: Массив закрывающих цен.
        :param period: Период для расчета AdOsc.
        :return: Массив значений AdOsc.
        """
        # Convert all input arrays to np.float64 (double) type to avoid TA-Lib errors
        high_double = high.astype(np.float64)
        low_double = low.astype(np.float64)
        close_double = close.astype(np.float64)
        volume_double = volume.astype(np.float64)
        return talib.ADOSC(high_double, low_double, close_double, volume_double, fast_period, slowperiod)

    @staticmethod
    def Rsi(close: np.ndarray, period: int) -> np.ndarray:
        """
        Вычисляет RSI (Relative Strength Index) для заданного периода.

        :param high: Массив высоких цен.
        :param low: Массив низких цен.
        :param close: Массив закрывающих цен.
        :param period: Период для расчета RSI.
        :return: Массив значений RSI.
        """
        return talib.RSI(close, timeperiod=period)

    @staticmethod
    def StochF_K(high: np.ndarray, low: np.ndarray, close: np.ndarray, fastKPeriod: int) -> np.ndarray:
        """
        Вычисляет StochF_K (Stochastic Oscillator K) для заданного периода.

        :param high: Массив высоких цен.
        :param low: Массив низких цен.
        :param close: Массив закрывающих цен.
        :param fastKPeriod: Быстрый период K.
        :param slowKPeriod: Медленный период K.
        :param fastDMAType: Тип скользящей средней для быстрого периода K.
        :return: Массив значений StochF_K.
        """
        return talib.STOCHF(high, low, close, fastKPeriod)[0]

   

    @staticmethod
    def VolatilityPercentage_def(volatilityValues: np.ndarray, prices: np.ndarray) -> np.ndarray:
        """
        Процентное изменение волатильности.

        :param volatilityValues: Список значений волатильности.
        :param prices: Список цен.
        :return: Список значений, представляющих процентное изменение волатильности.
        """
        # Проверка на null для списков
        if volatilityValues is None or prices is None:
            return None

        # Проверка на равенство длин списков
        if len(volatilityValues) != len(prices):
            raise ValueError("Длины списков должны совпадать.")

        # Используем векторизованные операции для вычисления процентного изменения волатильности
        percentageVolatility = np.zeros_like(prices, dtype=float)

        # Находим индексы, где prices не равны нулю
        non_zero_indices = prices != 0

        # Вычисляем процентное изменение волатильности только для не нулевых значений prices
        percentageVolatility[non_zero_indices] = (volatilityValues[non_zero_indices] / prices[non_zero_indices]) * 100.0

        return percentageVolatility

    @staticmethod
    def VolumeRatio_def(volumes: np.ndarray, period: int) -> np.ndarray:
        """
        Вычисляет отношение текущего объема к простому скользящему среднему объема (SMA) для заданного периода.

        :param volumes: Массив объемов, для которых вычисляется отношение.
        :param period: Период для расчета скользящего среднего.
        :return: Массив отношений текущего объема к SMA объема.
        :raises ValueError: Если период меньше или равен нулю.
        :raises ValueError: Если массив объемов является None.
        """
        # Проверка на null для массива объемов
        if volumes is None:
            return None

        # Проверка на корректность периода
        if period <= 0:
            raise ValueError("Период должен быть положительным числом.")

        # Рассчитываем простое скользящее среднее (SMA) объема
        maVol = AiFeatures.Sma(volumes, period)

        # Проверка на null для массива SMA объема
        if maVol is None:
            raise ValueError("Не удалось получить массив SMA объема.")

        # Вычисляем отношение текущего объема к SMA объема для каждого бара
        volumeRatio = np.where(maVol == 0.0, 1.0, volumes / maVol)

        return volumeRatio

    @staticmethod
    def PriceChannelPosition_def(high: np.ndarray, low: np.ndarray,
                                 close: np.ndarray, period: int) -> np.ndarray:
        """
        Индикатор положения цены относительно ценовых каналов.

        :param highestHigh: Верхний канал (ряд данных).
        :param lowestLow: Нижний канал (ряд данных).
        :param close: Цена закрытия (ряд данных).
        :return: Возвращает массив значений индикатора, показывающих положение цены относительно каналов.
        """

        rolling_max = pd.Series(high).rolling(window=period).max()
        rolling_min = pd.Series(low).rolling(window=period).min()

        # Calculate normalized position (0 = at minimum, 1 = at maximum)
        PriceChannelPosition = np.array((pd.Series(close) - rolling_min) / (rolling_max - rolling_min))

        return PriceChannelPosition

    @staticmethod
    def HourOfDay(date: np.ndarray) -> np.ndarray:

        # Преобразование данных в тип datetime64, если это необходимо
        if date.dtype != 'datetime64[ns]':
            date = pd.to_datetime(date).to_numpy(dtype='datetime64[ns]')

        # Получение часов из даты
        HourOfDay = pd.Series(date).dt.hour

        # Возвращаем массив значений индикатора
        return HourOfDay.to_numpy()

    @staticmethod
    def TimeOfDay(date: np.ndarray) -> np.ndarray:

        # Преобразование данных в тип datetime64, если это необходимо
        if date.dtype != 'datetime64[ns]':
            date = pd.to_datetime(date).to_numpy(dtype='datetime64[ns]')

        # Получение часов из даты
        date_hour = pd.Series(date).dt.hour

        # Бининг часов в категории времени суток
        TimeOfDay = pd.cut(date_hour,
                           bins=[-np.inf, 5, 11, 17, 23],
                           labels=[0, 1, 2, 3])

        # Возвращаем массив значений индикатора
        return TimeOfDay.to_numpy()

    @staticmethod
    def DayOfWeek(date: np.ndarray) -> np.ndarray:

        # Получение баров финансового инструмента
        if date.dtype != 'datetime64[ns]':
            date_dt = pd.to_datetime(date).to_numpy(dtype='datetime64[ns]')
        else:
            date_dt = date

        # Вычисление номера дня недели для каждого бара и преобразование в float
        day_of_week = pd.Series(date_dt).dt.weekday.astype(float)

        # Возвращаем массив значений индикатора
        return day_of_week.to_numpy() + 1

    @staticmethod
    def MoonPhase(date: np.ndarray) -> np.ndarray:
        # Вычисление количества дней с 1 января 2000 года
        reference_date = datetime(2000, 1, 1, 12, 0, 0)
        days_since_reference = (pd.to_datetime(date) - reference_date).total_seconds() / 86400

        # Вычисление фазы Луны
        moon_phase = (days_since_reference % 29.53058867) / 29.53058867

        # Нормализация фазы Луны в диапазоне от 0 до 1
        moon_phase = np.where(moon_phase < 0, moon_phase + 1, moon_phase)

        return moon_phase

    @staticmethod
    def DayOfMonth(date: np.ndarray) -> np.ndarray:
        # Проверка, что date является массивом numpy
        if not isinstance(date, np.ndarray):
            raise ValueError("Input date must be a numpy array")

        # Преобразование данных в тип datetime64, если это необходимо
        if date.dtype != 'datetime64[ns]':
            date = pd.to_datetime(date).to_numpy(dtype='datetime64[ns]')

        # Преобразование в Series для использования атрибута dt
        date_series = pd.Series(date)

        # Получение дня месяца
        day_of_month = date_series.dt.day.astype(float)

        # Возвращаем массив значений индикатора
        return day_of_month.to_numpy()

    @staticmethod
    def DayOfYear(date: np.ndarray) -> np.ndarray:
        # Проверка, что date является массивом numpy
        if not isinstance(date, np.ndarray):
            raise ValueError("Input date must be a numpy array")

        # Преобразование данных в тип datetime64, если это необходимо
        if date.dtype != 'datetime64[ns]':
            date = pd.to_datetime(date).to_numpy(dtype='datetime64[ns]')

        # Преобразование в Series для использования атрибута dt
        date_series = pd.Series(date)

        # Получение дня года
        day_of_year = date_series.dt.dayofyear.astype(float)

        # Возвращаем массив значений индикатора
        return day_of_year.to_numpy()

    @staticmethod
    @njit
    def _create_fibonacci_selection(indicator: np.ndarray, oldestBar: int, currentBar: int,
                                    initialStepCount: int) -> np.ndarray:
        if currentBar == -1 or currentBar >= len(indicator):
            currentBar = len(indicator) - 1
        if currentBar < oldestBar:
            raise ValueError("currentBar должен быть больше или равен oldestBar.")

        selection = []
        step = 1
        count = 0
        while currentBar - step >= oldestBar and count < initialStepCount:
            selection.append(indicator[currentBar - step])
            step += 1
            count += 1

        lastIndex = currentBar - step + 1
        step1, step2 = 1, 1
        while lastIndex - step1 >= oldestBar:
            selection.append(indicator[lastIndex - step1])
            step1, step2 = step2, step1 + step2

        return np.array(selection)

    @staticmethod
    def RankValueForBarFibonacci(indicator: np.ndarray, maxRank: int,
                                 highValueHighRank: bool = True,
                                 barNumb: int = -1,
                                 initialStepCount: int = 100) -> float:
        if barNumb == -1:
            barNumb = len(indicator) - 1  # Берем последний бар, если не передан

        window = AiFeatures._create_fibonacci_selection(indicator, oldestBar=0,
                                                        currentBar=barNumb,
                                                        initialStepCount=initialStepCount)

        if len(window) == 0:
            return 0.0  # Защита от пустого массива

        window_sorted = np.sort(window)
        rankIndex = np.searchsorted(window_sorted, indicator[barNumb], side='left')
        rank = (rankIndex / len(window_sorted)) * maxRank
        return rank if highValueHighRank else maxRank - rank


    def get_daily_ohlc(self, dates_array, high, low, close):
        """
        Преобразует минутные данные в дневные OHLC.
        
        :param dates_array: Массив datetime64 дат
        :param high: Массив максимумов
        :param low: Массив минимумов
        :param close: Массив цен закрытия
        :return: Кортеж (daily_high, daily_low, daily_close)
        """
        # Создаем DataFrame с данными
        df = pd.DataFrame({
            'date': dates_array,
            'high': high,
            'low': low,
            'close': close
        })
        
        # Конвертируем дату в datetime если это не datetime64
        if not np.issubdtype(df['date'].dtype, np.datetime64):
            df['date'] = pd.to_datetime(df['date'])
        
        # Группируем по дате (убираем время) и агрегируем
        daily_df = df.groupby(df['date'].dt.date).agg({
            'high': 'max',
            'low': 'min',
            'close': 'last'  # берем последнее значение дня
        })
        
        return daily_df['high'].values, daily_df['low'].values, daily_df['close'].values

    def calculate_labels(self, close_prices: np.ndarray, high_prices: np.ndarray, low_prices: np.ndarray, 
                        barrier_pct: float = 0.025, barrier_horizon: int = 60*6,
                        W: int = 60, thr: float = 0.11) -> tuple:
        """
        Рассчитывает метки для 3 классов на основе наклона KAMA, нормированного по ATR,
        с дополнительным усилением трендовых периодов и применением барьеров для фильтрации сигналов.
        
        BUY (2): Когда сила наклона KAMA > порог и наклон положительный
        SELL (0): Когда сила наклона KAMA > порог и наклон отрицательный
        HOLD (1): Все остальные случаи
        
        :param close_prices: Массив цен закрытия
        :param high_prices: Массив высоких цен
        :param low_prices: Массив низких цен
        :param barrier_pct: Процент барьера для фильтрации сигналов
        :param barrier_horizon: Горизонт в барах для проверки достижения барьера
        :param W: Окно KAMA
        :param thr: Базовый порог для силы наклона
        :return: Кортеж из двух массивов меток (labels, _)
        """
        
        # Вычисляем KAMA с окном W
        kama = talib.KAMA(close_prices, W)
        
        # Вычисляем наклон (slope) KAMA - приращение за один бар
        slope = np.diff(kama, prepend=kama[0])
        
        # Вычисляем ATR за то же окно
        atr = talib.ATR(high_prices, low_prices, close_prices, W)
        
        # Вычисляем силу движения как отношение модуля наклона к ATR
        strength = np.abs(slope) / atr
        strength = np.nan_to_num(strength, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Инициализируем метки как HOLD (1)
        labels = np.ones(len(close_prices))
        
        # Базовая разметка по порогу силы/знаку наклона
        labels[(strength > thr) & (slope > 0)] = 2  # BUY
        labels[(strength > thr) & (slope < 0)] = 0  # SELL

        # Пост-фильтр барьеров с горизонтом: BUY требует High >= entry*(1+barrier_pct), SELL требует Low <= entry*(1-barrier_pct)
        # в пределах barrier_horizon баров вперёд, иначе -> HOLD
        if barrier_pct is not None and barrier_horizon is not None and barrier_pct > 0 and barrier_horizon > 0:
            buy_idxs = np.where(labels == 2)[0]
            if buy_idxs.size > 0:
                buy_hit = np.zeros(buy_idxs.size, dtype=bool)
                for j, i in enumerate(buy_idxs):
                    entry = close_prices[i]
                    target = entry * (1.0 + barrier_pct)
                    start = i + 1
                    end = min(i + 1 + barrier_horizon, len(high_prices))
                    if start < end:
                        buy_hit[j] = np.nanmax(high_prices[start:end]) >= target
                labels[buy_idxs[~buy_hit]] = 1  # HOLD

            sell_idxs = np.where(labels == 0)[0]
            if sell_idxs.size > 0:
                sell_hit = np.zeros(sell_idxs.size, dtype=bool)
                for j, i in enumerate(sell_idxs):
                    entry = close_prices[i]
                    target = entry * (1.0 - barrier_pct)
                    start = i + 1
                    end = min(i + 1 + barrier_horizon, len(low_prices))
                    if start < end:
                        sell_hit[j] = np.nanmin(low_prices[start:end]) <= target
                labels[sell_idxs[~sell_hit]] = 1  # HOLD
        
        return labels, None
    
    



    def process_indicators(self, ohlc_data: pd.DataFrame, lags: list, output_dir: str, 
                          W: int = 60, thr: float = 0.11, barrier_pct: float = 0.025, 
                          barrier_horizon: int = 60*6) -> None:
        """
        Обрабатывает индикаторы и сохраняет результаты.
        
        :param ohlc_data: DataFrame с данными OHLCV
        :param lags: Список лагов для матриц индикаторов
        :param output_dir: Директория для сохранения результатов
        :param W: Окно KAMA
        :param thr: Базовый порог для силы наклона
        :param barrier_pct: Процент барьера для фильтрации сигналов
        :param barrier_horizon: Горизонт в барах для проверки достижения барьера
        """
        # Измеряем время выполнения
        start_time = time.time()
        
        # Вычисляем все индикаторы
        indicators_df = self.calculate_all_indicators(ohlc_data)
        indicators_time = time.time()
        print(f"Calculating indicators took {indicators_time - start_time:.2f} seconds")
        
        # Создаем матрицы индикаторов
        result_df, all_matrices, valid_indices = self.create_indicator_matrices(indicators_df, lags)
        matrices_time = time.time()
        print(f"Creating indicator matrices took {matrices_time - indicators_time:.2f} seconds")
        print(f"Valid matrices: {len(all_matrices)} out of {len(ohlc_data)}")


        # Получаем дневные OHLC из минутных данных
        dates_array = ohlc_data.index.values

        
        # Добавляем метки в result_df, а не в indicators_df
        # Используем новую логику разметки с барьерами
        labels, _ = self.calculate_labels(self.Close, self.High, self.Low, 
                                         barrier_pct=barrier_pct, barrier_horizon=barrier_horizon,
                                         W=W, thr=thr)
        result_df['ClassLabel'] = labels  # Используем одно поле для 3-классовой разметки
        
        # Добавляем OHLCV данные в result_df
        result_df['Close'] = self.Close
        result_df['Open'] = self.Open
        result_df['High'] = self.High
        result_df['Low'] = self.Low
        result_df['Volume'] = self.Volume
        result_df['Date'] = dates_array

        # Создаем массив меток для валидных матриц
        valid_labels = np.array([labels[idx] for idx in valid_indices], dtype=np.float32)
        valid_dates = np.array([dates_array[idx] for idx in valid_indices])
        
        # Сохраняем матрицы в numpy формате
        matrices_array = np.array(all_matrices, dtype=np.float32)
        
        # Сохраняем матрицы в файл
        np.save(os.path.join(output_dir, 'btc_indicator_matrices_MN_3.npy'), matrices_array)
        
        # Сохраняем метки и даты в один файл
        np.savez_compressed(
            os.path.join(output_dir, 'btc_labels_and_dates_MN_3.npz'),
            class_labels=valid_labels,
            dates=valid_dates
        )
        
        # Сохраняем индексы валидных матриц для возможного последующего анализа
        np.save(os.path.join(output_dir, 'btc_valid_indices_3.npy'), np.array(valid_indices))
        
        # Сохраняем базовую информацию в CSV для справки
        result_df.to_csv(
            os.path.join(output_dir, 'btc_indicator_matrices_info_MN_3.csv'),
            index=True,
            chunksize=25000  # Размер чанка в строках
        )
        
        save_time = time.time()
        print(f"Saving results took {save_time - matrices_time:.2f} seconds")
        print(f"Total processing time: {save_time - start_time:.2f} seconds")
        print(f"Saved {len(all_matrices)} matrices of size 50x50 in numpy format")
        print(f"Saved labels and dates in 'labels_and_dates.npz'")

    def calculate_all_indicators(self, ohlc_data: pd.DataFrame) -> pd.DataFrame:
        """
        Вычисляет все индикаторы для полного датафрейма.
        
        :param ohlc_data: DataFrame с данными OHLCV
        :return: DataFrame со всеми индикаторами
        """
        # Преобразуем данные в numpy массивы для быстрых вычислений
        self.Open = ohlc_data['open'].values
        self.High = ohlc_data['high'].values
        self.Low = ohlc_data['low'].values
        self.Close = ohlc_data['close'].values
        self.Volume = ohlc_data['volume'].values
        
        # Создаем DataFrame для хранения всех индикаторов
        indicators_df = pd.DataFrame(index=ohlc_data.index)

        # Сначала рассчитываем SMA, так как они нужны для расчета углов
        self.SMA_60 = self.Sma(self.Close, 60)
        self.SMA_240 = self.Sma(self.Close, 240)
        self.SMA_1440 = self.Sma(self.Close, 1440)

        # Углы - используем CalculateAngleArray с уже рассчитанными SMA
        indicators_df['ANGLE_SMA_60_100'] = self.CalculateAngleArray(self.SMA_60, 100)
        indicators_df['ANGLE_SMA_240_400'] = self.CalculateAngleArray(self.SMA_240, 400)
        indicators_df['ANGLE_SMA_1440_2400'] = self.CalculateAngleArray(self.SMA_1440, 2400)

        indicators_df['SMA_60'] = self.SMA_60
        indicators_df['SMA_240'] = self.SMA_240
        indicators_df['SMA_1440'] = self.SMA_1440

        indicators_df['ROCP100_Fib_1'] = self.RocP100(self.Close, 1)
        indicators_df['ROCP100_Fib_2'] = self.RocP100(self.Close, 2)
        indicators_df['ROCP100_Fib_3'] = self.RocP100(self.Close, 3)
        indicators_df['ROCP100_Fib_5'] = self.RocP100(self.Close, 5)
        indicators_df['ROCP100_Fib_8'] = self.RocP100(self.Close, 8)
        indicators_df['ROCP100_Fib_13'] = self.RocP100(self.Close, 13)
        indicators_df['ROCP100_Fib_21'] = self.RocP100(self.Close, 21)
        indicators_df['ROCP100_Fib_34'] = self.RocP100(self.Close, 34)
        indicators_df['ROCP100_Fib_55'] = self.RocP100(self.Close, 55)
        indicators_df['ROCP100_Fib_89'] = self.RocP100(self.Close, 89)
        indicators_df['ROCP100_Fib_144'] = self.RocP100(self.Close, 144)
        indicators_df['ROCP100_Fib_233'] = self.RocP100(self.Close, 233)
        indicators_df['ROCP100_Fib_377'] = self.RocP100(self.Close, 377)
        indicators_df['ROCP100_Fib_610'] = self.RocP100(self.Close, 610)
        indicators_df['ROCP100_Fib_987'] = self.RocP100(self.Close, 987)
        indicators_df['ROCP100_Fib_1597'] = self.RocP100(self.Close, 1597)
        indicators_df['ROCP100_Fib_2584'] = self.RocP100(self.Close, 2584)
        indicators_df['ROCP100_Fib_4181'] = self.RocP100(self.Close, 4181)
        indicators_df['ROCP100_Fib_6765'] = self.RocP100(self.Close, 6765)
        

        # ATR
        indicators_df['ATR_Pct_14'] = self.Atr(self.High, self.Low, self.Close, 14)
        indicators_df['ATR_Pct_64'] = self.Atr(self.High, self.Low, self.Close, 64)
        indicators_df['ATR_Pct_384'] = self.Atr(self.High, self.Low, self.Close, 384)


        # StDevPct
        indicators_df['StDevPct_60'] = self.StDevPct(self.Close, 60)
        indicators_df['StDevPct_240'] = self.StDevPct(self.Close, 240)
        indicators_df['StDevPct_1440'] = self.StDevPct(self.Close, 1440)

        # ROC
        indicators_df['ROC_10'] = self.ROC(self.Close, 10)
        indicators_df['ROC_40'] = self.ROC(self.Close, 40)
        indicators_df['ROC_240'] = self.ROC(self.Close, 240)

        # RSI
        indicators_df['Rsi_14'] = self.Rsi(self.Close, 14)
        indicators_df['Rsi_64'] = self.Rsi(self.Close, 64)
        indicators_df['Rsi_384'] = self.Rsi(self.Close, 384)

        # StochF_K
        indicators_df['StochF_K_14'] = self.StochF_K(self.High, self.Low, self.Close, 14)
        indicators_df['StochF_K_64'] = self.StochF_K(self.High, self.Low, self.Close, 64)
        indicators_df['StochF_K_384'] = self.StochF_K(self.High, self.Low, self.Close, 384)


        # AdOsc
        indicators_df['AdOsc_3_10'] = self.AdOsc(self.High, self.Low, self.Close,
                                     self.Volume, 3, 10)
        indicators_df['AdOsc_12_40'] = self.AdOsc(self.High, self.Low, self.Close,
                                     self.Volume, 12, 40)
        indicators_df['AdOsc_72_240'] = self.AdOsc(self.High, self.Low, self.Close,
                                     self.Volume, 72, 240)


        # Calculate VolatilityPercentage
        # volatilityValues = np.abs(self.High - self.Low)
        # indicators_df['VolatilityPercentage'] = self.VolatilityPercentage_def(volatilityValues, self.Close)

        # Calculate VolumeRatio
        indicators_df['VolumeRatio_60'] = self.VolumeRatio_def(self.Volume, 60)
        indicators_df['VolumeRatio_240'] = self.VolumeRatio_def(self.Volume, 240)
        indicators_df['VolumeRatio_1440'] = self.VolumeRatio_def(self.Volume, 1440)

        # Price Channel
        indicators_df['PriceChannelPosition_60'] = self.PriceChannelPosition_def(self.High, self.Low, self.Close, 60)
        indicators_df['PriceChannelPosition_240'] = self.PriceChannelPosition_def(self.High, self.Low, self.Close, 240)
        indicators_df['PriceChannelPosition_1440'] = self.PriceChannelPosition_def(self.High, self.Low, self.Close, 1440)

        # Сезонные индикаторы
        # Преобразуем индекс в numpy массив datetime64
        dates_array = ohlc_data.index.values
        
        # Вычисляем сезонные индикаторы для всего массива дат
        #indicators_df['TimeOfDay'] = self.TimeOfDay(dates_array)
        indicators_df['HourOfDay'] = self.HourOfDay(dates_array)
        #indicators_df['DayOfWeek'] = self.DayOfWeek(dates_array)
        #indicators_df['DayOfYear'] = self.DayOfYear(dates_array)
        #indicators_df['DayOfMonth'] = self.DayOfMonth(dates_array)
        #indicators_df['MoonPhase'] = self.MoonPhase(dates_array)

        # Нормализация сезонных индикаторов
        indicators_df['HourOfDay'] = (indicators_df['HourOfDay'])/23
        #indicators_df['TimeOfDay'] = (indicators_df['TimeOfDay'])/3
        #indicators_df['DayOfWeek'] = (indicators_df['DayOfWeek']-1)/6
        #indicators_df['DayOfYear'] = self.NormalizeSeasonalIndicator(indicators_df['DayOfYear'].values, 1, 366)
        #indicators_df['DayOfMonth'] = (indicators_df['DayOfMonth']-1)/30
        #indicators_df['MoonPhase'] = self.NormalizeSeasonalIndicator(indicators_df['MoonPhase'].values, 1, 100)

        return indicators_df

    @staticmethod
    @njit(parallel=True)
    def _normalize_values_parallel(values: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        """Нормализует массив значений параллельно"""
        result = np.zeros_like(values)
        for i in prange(len(values)):
            if not np.isnan(values[i]):
                result[i] = (values[i] - min_val) / (max_val - min_val)
        return result

    def create_indicator_matrices(self, indicators_df: pd.DataFrame, lags: list) -> pd.DataFrame:
        """
        Создает матрицы индикаторов с лагами для каждой временной точки.
        Оптимизированная версия с использованием numpy для быстрых вычислений.
        
        :param indicators_df: DataFrame со всеми индикаторами
        :param lags: Список лагов для создания матриц
        :return: DataFrame с матрицами индикаторов
        """
        result_df = pd.DataFrame(index=indicators_df.index)
        
        # Список сезонных индикаторов, которые не нужно нормализовать
        seasonal_indicators = {'TimeOfDay', 'HourOfDay'}
        
        # Размер матрицы
        matrix_size = 50
        
        # Создаем список для хранения всех матриц
        all_matrices = []
        valid_indices = []
        
        def process_timepoint(idx):
            # Создаем пустую матрицу 50x50 с типом float32
            matrix = np.zeros((matrix_size, matrix_size), dtype=np.float32)
            is_valid = True
            
            row_idx = 0
            for column in indicators_df.columns:
                values = indicators_df[column].values
                valid_data = []
                valid_lags = []
                
                for lag in lags:
                    lag_idx = idx - lag
                    if lag_idx >= 0:
                        value = values[lag_idx]
                        if not np.isnan(value):
                            valid_data.append(value)
                            valid_lags.append(lag)
                
                if valid_data:
                    if column in seasonal_indicators:
                        # Для сезонных индикаторов не нормализуем, они уже нормализованы
                        normalized = valid_data
                    else:
                        # Нормализуем остальные индикаторы
                        valid_data = np.array(valid_data)
                        min_val = np.min(valid_data)
                        max_val = np.max(valid_data)
                        
                        if max_val != min_val:
                            normalized = (valid_data - min_val) / (max_val - min_val)
                        else:
                            normalized = np.zeros_like(valid_data)
                
                    # Проверяем, что у нас достаточно данных для заполнения строки
                    if len(normalized) < matrix_size:
                        is_valid = False
                        break
                    
                    # Заполняем строку матрицы
                    matrix[row_idx, :len(normalized[:matrix_size])] = normalized[:matrix_size]
                    row_idx += 1
                    
                    # Если превысили размер матрицы, прекращаем
                    if row_idx >= matrix_size:
                        break
        
            # Проверяем, что матрица полностью заполнена (50x50)
            if row_idx < matrix_size:
                is_valid = False
        
            return matrix, is_valid, idx
        
        # Используем ThreadPoolExecutor для параллельной обработки
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(executor.map(
                process_timepoint, 
                range(len(indicators_df))
            ))
        
        # Фильтруем только валидные матрицы и сохраняем их индексы
        for matrix, is_valid, idx in results:
            if is_valid:
                all_matrices.append(matrix)
                valid_indices.append(idx)
        
        # Сохраняем индексы валидных матриц
        result_df['valid_matrix'] = False
        for idx in valid_indices:
            result_df.iloc[idx, result_df.columns.get_loc('valid_matrix')] = True
        
        return result_df, all_matrices, valid_indices

def read_ohlc_data(file_path: str) -> pd.DataFrame:
    """
    Читает OHLCV данные из CSV файла с форматом:
    bar,Date,Open,High,Low,Close,Volume
    
    Пример строки:
    0,2024-07-01T00:00:00,62766.1,62809.5,62766,62769.1,270.744
    """
    df = pd.read_csv(file_path)
    #df = df.iloc[1550000:]
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Убираем столбец 'bar', так как он нам не нужен ?
    if 'bar' in df.columns:
        df = df.drop('bar', axis=1)
    
    # Приводим названия столбцов к нижнему регистру для единообразия
    df.columns = df.columns.str.lower()
    
    return df

def process_indicators(file_path: str, lags: list, output_dir: str, 
                      W: int = 60, thr: float = 0.11, barrier_pct: float = 0.025, 
                      barrier_horizon: int = 360) -> pd.DataFrame:
    """
    Основная функция для обработки данных и создания матриц индикаторов
    
    :param file_path: Путь к CSV файлу с данными
    :param lags: Список лагов для создания матриц
    :return: DataFrame с матрицами индикаторов
    """
    start_total = time.time()
    
    # Читаем данные
    start_read = time.time()
    ohlc_data = read_ohlc_data(file_path)
    end_read = time.time()
    print(f"Чтение данных: {end_read - start_read:.2f} секунд")
    print(f"Количество строк в данных: {len(ohlc_data)}")
    
    # Создаем экземпляр класса и вычисляем все индикаторы
    start_indicators = time.time()
    ai_features = AiFeatures()
    ai_features.process_indicators(ohlc_data, lags, output_dir, W, thr, barrier_pct, barrier_horizon)
    end_indicators = time.time()
    print(f"Расчет индикаторов: {end_indicators - start_indicators:.2f} секунд")
    print(f"Количество индикаторов: {len(ai_features.calculate_all_indicators(ohlc_data).columns)}")
    
    end_total = time.time()
    print(f"\nОбщее время выполнения: {end_total - start_total:.2f} секунд")
    print(f"Размер выходного DataFrame: {len(ohlc_data)}")

def visualize_prices_and_labels(file_path: str, window_start: int = 0, window_size: int = 1000, 
                               compare_methods: bool = False, W: int = 30, thr: float = 0.08,
                               barrier_pct: float = 0.015, barrier_horizon: int = 60*6):
    """
    Визуализирует цены закрытия, KAMA, и сигналы на основе наклона KAMA, нормированного по ATR.
    Опционально сравнивает старый и новый методы разметки.

    :param file_path: Путь к CSV файлу с данными
    :param window_start: Начальный индекс окна для визуализации
    :param window_size: Размер окна для визуализации
    :param compare_methods: Сравнивать ли старый и новый методы разметки
    """
    # Читаем данные
    print(f"Загрузка данных из {file_path}...")
    ohlc_data = read_ohlc_data(file_path)
    print(f"Загружено {len(ohlc_data)} строк данных")
    
    # Создаем экземпляр класса AiFeatures
    ai_features = AiFeatures()
    
    # Устанавливаем данные OHLCV
    ai_features.Open = ohlc_data['open'].values
    ai_features.High = ohlc_data['high'].values
    ai_features.Low = ohlc_data['low'].values
    ai_features.Close = ohlc_data['close'].values
    ai_features.Volume = ohlc_data['volume'].values
    
    # Даты для визуализации
    dates_array = ohlc_data.index.values
    
    # Рассчитываем KAMA с окном W
    kama = talib.KAMA(ai_features.Close, timeperiod=W)
    
    # Вычисляем наклон (slope) KAMA - приращение за один бар
    slope = np.diff(kama, prepend=kama[0])  # Используем numpy.diff и добавляем первый элемент для сохранения размера
    
    # Вычисляем ATR за то же окно
    atr = talib.ATR(ai_features.High, ai_features.Low, ai_features.Close, W)
    
    # Вычисляем силу движения как отношение модуля наклона к ATR
    strength = np.abs(slope) / atr
    strength = np.nan_to_num(strength, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Инициализируем метки как HOLD (1)
    labels = np.ones(len(ai_features.Close))
    
    # Сохраняем копию меток до применения барьеров для сравнения (если нужно)
    labels_before = labels.copy()
    
    # Используем метод calculate_labels для получения меток с барьерами
    labels, _ = ai_features.calculate_labels(
        ai_features.Close, ai_features.High, ai_features.Low,
        barrier_pct=barrier_pct, barrier_horizon=barrier_horizon,
        W=W, thr=thr
    )
    
    # Для сравнения методов (если включено)
    labels_old = None
    if compare_methods:
        # Создаем метки по старому методу (без барьеров)
        labels_old = np.ones(len(ai_features.Close))
        labels_old[(strength > thr) & (slope > 0)] = 2  # BUY
        labels_old[(strength > thr) & (slope < 0)] = 0  # SELL
    end_idx = min(window_start + window_size, len(ohlc_data))
    start_idx = max(0, window_start)
    # Если окно пустое (например, window_start за пределами длины), сдвигаем на последний доступный диапазон
    if start_idx >= end_idx:
        end_idx = len(ohlc_data)
        start_idx = max(0, end_idx - window_size)
        print(f"Предупреждение: окно визуализации скорректировано на [{start_idx}:{end_idx}] из-за выхода за пределы данных")
    
    # Выбираем данные для визуализации
    window_dates = dates_array[start_idx:end_idx]
    window_close = ai_features.Close[start_idx:end_idx]
    window_labels = labels[start_idx:end_idx]
    window_kama = kama[start_idx:end_idx]
    window_slope = slope[start_idx:end_idx]
    window_atr = atr[start_idx:end_idx]
    
    # Вычисляем силу движения для выбранного окна
    window_strength = np.abs(window_slope) / window_atr
    window_strength = np.nan_to_num(window_strength, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Создаем график с двумя подграфиками
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]})
    
    # Верхний график: цена закрытия и линия KAMA
    ax1.plot(window_dates, window_close, label='Цена закрытия', color='blue', linewidth=1)
    ax1.plot(window_dates, window_kama, label=f'KAMA ({W})', color='green', linewidth=1.2)
    
    # Находим индексы для каждого класса
    buy_indices = np.where(window_labels == 2.0)[0]  # BUY (2)
    hold_indices = np.where(window_labels == 1.0)[0]  # HOLD (1)
    sell_indices = np.where(window_labels == 0.0)[0]  # SELL (0)
    
    # Выводим информацию о количестве найденных сигналов
    print(f"Найдено {len(buy_indices)} BUY-сигналов (2)")
    print(f"Найдено {len(hold_indices)} HOLD-сигналов (1)")
    print(f"Найдено {len(sell_indices)} SELL-сигналов (0)")
    
    # Проверка распределения классов
    total_samples = len(window_labels)
    if total_samples == 0:
        print("Предупреждение: пустое окно визуализации после коррекции; пропускаю расчёт распределения классов")
        buy_percent = sell_percent = hold_percent = 0.0
    else:
        buy_percent = len(buy_indices) / total_samples * 100
        sell_percent = len(sell_indices) / total_samples * 100
        hold_percent = len(hold_indices) / total_samples * 100
    
    print(f"\nРаспределение классов (новый метод):")
    print(f"BUY:  {len(buy_indices)} ({buy_percent:.2f}%)")
    print(f"HOLD: {len(hold_indices)} ({hold_percent:.2f}%)")
    print(f"SELL: {len(sell_indices)} ({sell_percent:.2f}%)")
    
    # Сравнение методов, если включено
    if compare_methods and labels_old is not None:
        window_labels_old = labels_old[start_idx:end_idx]
        buy_indices_old = np.where(window_labels_old == 2.0)[0]
        hold_indices_old = np.where(window_labels_old == 1.0)[0]
        sell_indices_old = np.where(window_labels_old == 0.0)[0]
        
        buy_percent_old = len(buy_indices_old) / total_samples * 100
        sell_percent_old = len(sell_indices_old) / total_samples * 100
        hold_percent_old = len(hold_indices_old) / total_samples * 100
        
        print(f"\nРаспределение классов (старый метод):")
        print(f"BUY:  {len(buy_indices_old)} ({buy_percent_old:.2f}%)")
        print(f"HOLD: {len(hold_indices_old)} ({hold_percent_old:.2f}%)")
        print(f"SELL: {len(sell_indices_old)} ({sell_percent_old:.2f}%)")
        
        print(f"\nИзменения в распределении:")
        print(f"BUY:  {buy_percent - buy_percent_old:+.2f}% ({len(buy_indices) - len(buy_indices_old):+d} сигналов)")
        print(f"SELL: {sell_percent - sell_percent_old:+.2f}% ({len(sell_indices) - len(sell_indices_old):+d} сигналов)")
        print(f"HOLD: {hold_percent - hold_percent_old:+.2f}% ({len(hold_indices) - len(hold_indices_old):+d} сигналов)")
    
    # Предупреждение, если распределение не оптимально
    if buy_percent < 10 or sell_percent < 10:
        print(f"\nВнимание: Классы BUY ({buy_percent:.2f}%) или SELL ({sell_percent:.2f}%) меньше 10%")
        print(f"Возможно, стоит понизить порог (текущий: {thr})")

    
    # Отмечаем точки с BUY-сигналами (2) на верхнем графике
    if len(buy_indices) > 0:
        ax1.scatter(
            [window_dates[i] for i in buy_indices], 
            [window_close[i] for i in buy_indices], 
            color='green', marker='^', s=50, label='BUY (2)'
        )
    
    # Отмечаем точки с SELL-сигналами (0) на верхнем графике
    if len(sell_indices) > 0:
        ax1.scatter(
            [window_dates[i] for i in sell_indices], 
            [window_close[i] for i in sell_indices], 
            color='red', marker='v', s=50, label='SELL (0)'
        )
    
    
    # Нижний график: кумулятивный подсчет позиции по сигналам
    # Преобразуем метки в значения позиций: BUY (2) -> +1, SELL (0) -> -1, HOLD (1) -> 0
    position_values = np.zeros(len(window_labels))
    position_values[window_labels == 2] = 1  # BUY = +1
    position_values[window_labels == 0] = -1  # SELL = -1
    # HOLD остается 0
    
    # Вычисляем кумулятивную сумму
    cumulative_position = np.cumsum(position_values)
    
    # Отображаем кумулятивную позицию
    ax2.plot(window_dates, cumulative_position, color='purple', linewidth=1.5, label='Кумулятивная позиция')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)  # Линия нуля
    
    # Добавляем также сами сигналы для наглядности
    ax2.plot(window_dates, position_values, color='green', linewidth=0.8, alpha=0.5, label='Сигналы (+1/-1/0)')
    
    # Добавляем подписи к осям
    ax2.set_ylabel('Кумулятивная позиция')
    
    # Добавляем текст с итоговой позицией
    final_position = cumulative_position[-1]
    ax2.text(window_dates[0], max(cumulative_position), f'Итоговая позиция: {final_position:.0f}', 
             fontsize=10, color='blue', bbox=dict(facecolor='white', alpha=0.7))
    
    # Добавляем заливку для визуализации будущих цен (только для примера)
    # Выбираем несколько точек с сигналами для демонстрации
    future_window = 1440  # Такой же размер окна, как в calculate_labels
    example_points = []
    
    # Выбираем до 3 точек BUY и до 3 точек SELL для демонстрации
    if len(buy_indices) > 0:
        example_buy_points = buy_indices[:min(3, len(buy_indices))]
        for idx in example_buy_points:
            if idx + future_window < len(window_close):
                example_points.append((idx, 'buy'))
    
    if len(sell_indices) > 0:
        example_sell_points = sell_indices[:min(3, len(sell_indices))]
        for idx in example_sell_points:
            if idx + future_window < len(window_close):
                example_points.append((idx, 'sell'))
    
    # Для каждой выбранной точки показываем будущие цены
    for point_idx, signal_type in example_points:
        # Получаем текущую цену и будущие цены
        current_price = window_close[point_idx]
        future_end = min(point_idx + future_window, len(window_close) - 1)
        future_dates = window_dates[point_idx:future_end]
        future_prices = window_close[point_idx:future_end]
        
        # Считаем процент цен выше/ниже текущей
        higher_count = np.sum(future_prices > current_price)
        lower_count = np.sum(future_prices < current_price)
        higher_percent = higher_count / len(future_prices) if len(future_prices) > 0 else 0
        lower_percent = lower_count / len(future_prices) if len(future_prices) > 0 else 0
        
        # Добавляем аннотацию с процентами
        if signal_type == 'buy':
            ax1.annotate(f'{higher_percent:.1%} выше', 
                        xy=(window_dates[point_idx], current_price * 1.01),
                        xytext=(window_dates[point_idx], current_price * 1.03),
                        arrowprops=dict(facecolor='green', shrink=0.05),
                        color='green', fontsize=8)
        else:  # sell
            ax1.annotate(f'{lower_percent:.1%} ниже', 
                        xy=(window_dates[point_idx], current_price * 0.99),
                        xytext=(window_dates[point_idx], current_price * 0.97),
                        arrowprops=dict(facecolor='red', shrink=0.05),
                        color='red', fontsize=8)
    
    # Настройка верхнего графика
    ax1.set_title(f"Сигналы с фильтром барьеров {barrier_pct*100:.1f}% (High/Low) в горизонте {barrier_horizon} баров", fontsize=14)
    ax1.set_ylabel('Цена', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Настройка нижнего графика
    ax2.set_title('Разница между быстрой и медленной KAMA', fontsize=14)
    ax2.set_xlabel('Дата', fontsize=12)
    ax2.set_ylabel('Разница', fontsize=12)
    # Устанавливаем автоматические границы для лучшей визуализации
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # Форматирование даты на оси X
    date_format = DateFormatter('%Y-%m-%d')
    ax1.xaxis.set_major_formatter(date_format)
    ax2.xaxis.set_major_formatter(date_format)
    plt.xticks(rotation=45)
    
    # Автоматическое выравнивание элементов графика
    plt.tight_layout()
    
    # Сохраняем графики один раз перед показом и закрываем фигуру после
    fig_path1 = 'kama_crossover_future_analysis_visualization.png'
    fig.savefig(fig_path1, dpi=300, bbox_inches='tight')
    output_path = os.path.join(os.path.dirname(file_path), 'kama_crossover_signals_visualization.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Графики сохранены: {fig_path1} и {output_path}")
    
    # Показываем и закрываем фигуру
    plt.show()
    plt.close(fig)
    
    # Возвращаем данные и метки
    return ohlc_data, labels, labels_old if compare_methods and labels_old is not None else labels.copy()

def main(W=60, thr=0.11, barrier_pct=0.025, barrier_horizon=60*6):
    """
    Основная функция обработки данных.
    
    :param W: Окно KAMA
    :param thr: Базовый порог для силы наклона
    :param barrier_pct: Процент барьера для фильтрации сигналов
    :param barrier_horizon: Горизонт в барах для проверки достижения барьера
    """
    start_total = time.time()
    
    # Путь к файлу с данными
    file_path = 'data\SOLUSDT_1m_20210101_to_20250820.csv'
    
    # Список лагов для матриц индикаторов
    lags = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
    100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 
    205, 210, 215, 220, 225, 230, 235, 240, 245]
    
    print(f"Количество лагов: {len(lags)}")
    
    output_dir = 'data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Раскомментируйте следующую строку для запуска визуализации
    # visualize_prices_and_labels(file_path, window_start=0, window_size=1000)
    
    process_indicators(file_path, lags, output_dir,
                              W=W, thr=thr, barrier_pct=barrier_pct, barrier_horizon=barrier_horizon)

if __name__ == '__main__':
    # Определяем все параметры в одном месте
    file_path = 'data\ETHUSDT_1m_20210107_to_20250826.csv'
    
    # Параметры для KAMA и расчета меток
    W = 30                # Окно KAMA
    thr = 0.05            # Порог для силы наклона
    barrier_pct = 0.02   # Процент барьера для фильтрации сигналов
    barrier_horizon = 60*6 # Горизонт в барах для проверки достижения барьера
    
    # Параметры для визуализации
    window_start = 400000
    window_size = 10000
    compare_methods = False
    
    # Запускаем основную функцию обработки данных
    # Раскомментируйте, чтобы запустить
    main(W=W, thr=thr, barrier_pct=barrier_pct, barrier_horizon=barrier_horizon)
    
    # Запускаем визуализацию цен и меток с заданными параметрами
    visualize_prices_and_labels(
        file_path, 
        window_start=window_start, 
        window_size=window_size,
        compare_methods=compare_methods,
        W=W, 
        thr=thr,
        barrier_pct=barrier_pct, 
        barrier_horizon=barrier_horizon
    )