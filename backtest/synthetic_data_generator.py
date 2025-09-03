import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Tuple, Dict
import os
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class OrnsteinUhlenbeckGenerator:
    """
    Генератор синтетических временных рядов на основе процесса Орнштейна-Уленбека
    согласно методологии из статьи "Determining Optimal Trading Rules without Back-testing"
    """
    
    def __init__(self):
        self.parameters = {}
        self.synthetic_dir = Path("data/synthetic")
        
    def estimate_ou_parameters(self, price_series: np.ndarray, symbol: str) -> Tuple[float, float, float]:
        """
        Оценка параметров OU-процесса через OLS-регрессию
        
        Args:
            price_series: Временной ряд цен
            symbol: Символ актива для логирования
            
        Returns:
            Tuple[μ, φ, σ] - параметры OU-процесса
        """
        # Переход к лог-ценам
        P = np.log(price_series)
        
        # Вычисление разностей ΔP_t = P_t - P_{t-1}
        dP = np.diff(P)
        
        # Лаговые цены P_{t-1}
        lagP = P[:-1]
        
        # Центрирование для μ=0 в регрессии
        X = lagP - lagP.mean()
        
        # OLS-регрессия: ΔP_t = θ · (P_{t-1} - μ) + ε_t
        model = sm.OLS(dP, X).fit()
        theta = model.params[0]
        
        # Преобразование коэффициентов
        phi = 1 + theta
        
        # Стандартное отклонение остатков
        residuals = dP - theta * X
        sigma = np.std(residuals, ddof=1)
        
        # Долгосрочное среднее
        mu = lagP.mean()
        
        # Расчет half-life
        if phi > 0 and phi < 1:
            half_life = np.log(2) / np.log(1/phi)
        else:
            half_life = np.inf
            
        print(f"\n{symbol} - Parametry OU-processa:")
        print(f"  mu (dolgosrochnoe srednee): {mu:.6f}")
        print(f"  phi (koefficient avtoregresii): {phi:.6f}")
        print(f"  sigma (volatilnost): {sigma:.6f}")
        print(f"  Half-life: {half_life:.2f} periodov")
        print(f"  R-squared: {model.rsquared:.4f}")
        
        # Сохранение параметров
        self.parameters[symbol] = {
            'mu': mu,
            'phi': phi,
            'sigma': sigma,
            'half_life': half_life,
            'r_squared': model.rsquared,
            'last_price': P[-1]  # Последняя лог-цена для начала симуляции
        }
        
        return mu, phi, sigma
    
    def generate_synthetic_paths(self, symbol: str, n_paths: int = 10000, 
                               path_length: int = 1440, seed: int = None) -> np.ndarray:
        """
        Генерация синтетических ценовых траекторий
        
        Args:
            symbol: Символ актива
            n_paths: Количество траекторий
            path_length: Длина каждой траектории (в барах)
            seed: Сид для воспроизводимости
            
        Returns:
            np.ndarray: Массив синтетических цен [n_paths, path_length]
        """
        if symbol not in self.parameters:
            raise ValueError(f"Параметры для {symbol} не найдены. Сначала вызовите estimate_ou_parameters()")
        
        params = self.parameters[symbol]
        mu, phi, sigma = params['mu'], params['phi'], params['sigma']
        P0 = params['last_price']
        
        if seed is not None:
            np.random.seed(seed)
        
        # Инициализация массива траекторий
        paths = np.zeros((n_paths, path_length + 1))
        paths[:, 0] = P0  # Начальная цена
        
        # Генерация случайных шоков
        shocks = np.random.normal(0, sigma, (n_paths, path_length))
        
        # Генерация траекторий
        for t in range(1, path_length + 1):
            paths[:, t] = mu + phi * (paths[:, t-1] - mu) + shocks[:, t-1]
        
        # Возврат к обычным ценам (exp от лог-цен)
        price_paths = np.exp(paths[:, 1:])  # Исключаем начальную цену
        
        print(f"\n{symbol} - Sgenerirowano {n_paths} traektorij dlinoj {path_length} barov")
        print(f"  Srednyaya cena: {np.mean(price_paths):.2f}")
        print(f"  Std. otklonenie: {np.std(price_paths):.2f}")
        print(f"  Min/Max: {np.min(price_paths):.2f} / {np.max(price_paths):.2f}")
        
        return price_paths
    
    def analyze_and_save_characteristic_paths(self, symbol: str, price_paths: np.ndarray,
                                            base_timestamp: str = "2025-01-01 00:00:00"):
        """
        Анализ и сохранение характерных траекторий (лучшие, нулевые, худшие)
        
        Args:
            symbol: Символ актива
            price_paths: Массив синтетических цен
            base_timestamp: Базовая временная метка
        """
        n_paths, path_length = price_paths.shape
        
        # Расчет итоговой доходности для каждого пути
        initial_prices = price_paths[:, 0]
        final_prices = price_paths[:, -1]
        total_returns = (final_prices - initial_prices) / initial_prices * 100
        
        # Сортировка путей по доходности
        sorted_indices = np.argsort(total_returns)
        
        # Выбор характерных путей
        # 3 лучших (максимальный рост)
        best_indices = sorted_indices[-3:]
        
        # 3 худших (максимальное падение)
        worst_indices = sorted_indices[:3]
        
        # 3 нейтральных (ближе всего к нулевой доходности)
        zero_target = 0.0
        zero_distances = np.abs(total_returns - zero_target)
        neutral_indices = np.argsort(zero_distances)[:3]
        
        # Создание директории для характерных путей
        characteristic_dir = self.synthetic_dir / "characteristic_paths"
        characteristic_dir.mkdir(exist_ok=True)
        
        # Генерация временных меток
        base_time = pd.to_datetime(base_timestamp)
        timestamps = [base_time + timedelta(minutes=i) for i in range(path_length)]
        
        # Функция для сохранения пути
        def save_path(path_idx, category, rank):
            prices = price_paths[path_idx]
            return_pct = total_returns[path_idx]
            
            # Генерация OHLC данных
            noise_factor = 0.001
            opens = prices * (1 + np.random.normal(0, noise_factor, len(prices)))
            highs = prices * (1 + np.abs(np.random.normal(0, noise_factor, len(prices))))
            lows = prices * (1 - np.abs(np.random.normal(0, noise_factor, len(prices))))
            closes = prices
            volumes = np.random.lognormal(10, 1, len(prices))
            
            # Создание DataFrame
            df = pd.DataFrame({
                'Open': opens,
                'High': highs,
                'Low': lows,
                'Close': closes,
                'Volume': volumes,
                'Date': timestamps
            })
            
            # Сохранение файла
            filename = f"{symbol}_{category}_rank_{rank}_return_{return_pct:.2f}pct.csv"
            filepath = characteristic_dir / filename
            df.to_csv(filepath, index=False)
            
            return filepath, return_pct
        
        # Сохранение характерных путей
        saved_paths = {
            'best': [],
            'neutral': [],
            'worst': []
        }
        
        # Лучшие пути
        for i, idx in enumerate(reversed(best_indices)):  # От лучшего к худшему из лучших
            filepath, return_pct = save_path(idx, 'best', i+1)
            saved_paths['best'].append((filepath, return_pct))
        
        # Нейтральные пути
        for i, idx in enumerate(neutral_indices):
            filepath, return_pct = save_path(idx, 'neutral', i+1)
            saved_paths['neutral'].append((filepath, return_pct))
        
        # Худшие пути
        for i, idx in enumerate(worst_indices):
            filepath, return_pct = save_path(idx, 'worst', i+1)
            saved_paths['worst'].append((filepath, return_pct))
        
        # Создание сводного отчета
        report_file = characteristic_dir / f"{symbol}_characteristic_paths_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"Характерные траектории для {symbol}\n")
            f.write(f"Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("СТАТИСТИКА ПО ВСЕМ ПУТЯМ:\n")
            f.write(f"Общее количество путей: {n_paths}\n")
            f.write(f"Средняя доходность: {np.mean(total_returns):.4f}%\n")
            f.write(f"Стандартное отклонение: {np.std(total_returns):.4f}%\n")
            f.write(f"Минимальная доходность: {np.min(total_returns):.4f}%\n")
            f.write(f"Максимальная доходность: {np.max(total_returns):.4f}%\n\n")
            
            f.write("ЛУЧШИЕ ПУТИ (максимальный рост):\n")
            for i, (filepath, return_pct) in enumerate(saved_paths['best']):
                f.write(f"  {i+1}. {filepath.name} - доходность: {return_pct:.4f}%\n")
            
            f.write("\nНЕЙТРАЛЬНЫЕ ПУТИ (близко к нулевому росту):\n")
            for i, (filepath, return_pct) in enumerate(saved_paths['neutral']):
                f.write(f"  {i+1}. {filepath.name} - доходность: {return_pct:.4f}%\n")
            
            f.write("\nХУДШИЕ ПУТИ (максимальное падение):\n")
            for i, (filepath, return_pct) in enumerate(saved_paths['worst']):
                f.write(f"  {i+1}. {filepath.name} - доходность: {return_pct:.4f}%\n")
        
        print(f"\n{symbol} - Harakternyje puti sohraneny:")
        print(f"  Direktoriya: {characteristic_dir}/")
        print(f"  Luchshie puti: {len(saved_paths['best'])}")
        print(f"  Nejtralnyje puti: {len(saved_paths['neutral'])}")
        print(f"  Hudshie puti: {len(saved_paths['worst'])}")
        print(f"  Otchot: {report_file}")
        
        return saved_paths

    def save_synthetic_data(self, symbol: str, price_paths: np.ndarray, 
                          base_timestamp: str = "2025-01-01 00:00:00"):
        """
        Сохранение синтетических данных в CSV файлы
        
        Args:
            symbol: Символ актива
            price_paths: Массив синтетических цен
            base_timestamp: Базовая временная метка
        """
        n_paths, path_length = price_paths.shape
        
        # Создание директории для синтетических данных
        synthetic_dir = "data/synthetic"
        os.makedirs(synthetic_dir, exist_ok=True)
        
        # Генерация временных меток (1-минутные интервалы)
        base_time = pd.to_datetime(base_timestamp)
        timestamps = [base_time + timedelta(minutes=i) for i in range(path_length)]
        
        # Сохранение каждой траектории в отдельный файл
        for path_idx in range(min(100, n_paths)):  # Сохраняем первые 100 траекторий
            # Создание OHLCV данных из цен (упрощенная версия)
            prices = price_paths[path_idx]
            
            # Генерация OHLC из цен закрытия (добавляем небольшой шум)
            noise_factor = 0.001  # 0.1% шума
            opens = prices * (1 + np.random.normal(0, noise_factor, len(prices)))
            highs = prices * (1 + np.abs(np.random.normal(0, noise_factor, len(prices))))
            lows = prices * (1 - np.abs(np.random.normal(0, noise_factor, len(prices))))
            closes = prices
            volumes = np.random.lognormal(10, 1, len(prices))  # Случайные объемы
            
            # Создание DataFrame
            df = pd.DataFrame({
                'Open': opens,
                'High': highs,
                'Low': lows,
                'Close': closes,
                'Volume': volumes,
                'Date': timestamps
            })
            
            # Сохранение в CSV
            filename = f"{synthetic_dir}/{symbol}_synthetic_path_{path_idx:03d}.csv"
            df.to_csv(filename, index=False)
        
        # Сохранение сводного файла с параметрами
        summary_file = f"{synthetic_dir}/{symbol}_generation_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Синтетические данные для {symbol}\n")
            f.write(f"Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Параметры OU-процесса:\n")
            params = self.parameters[symbol]
            for key, value in params.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\nКоличество траекторий: {n_paths}\n")
            f.write(f"Длина траектории: {path_length} баров\n")
            f.write(f"Сохранено файлов: {min(100, n_paths)}\n")
        
        print(f"\n{symbol} - Sinteticheskie dannye sohraneny:")
        print(f"  Direktoriya: {synthetic_dir}/")
        print(f"  Fajlov traektorij: {min(100, n_paths)}")
        print(f"  Svodka: {summary_file}")

def load_crypto_data(filepath: str) -> pd.DataFrame:
    """Загрузка данных криптовалют"""
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def main():
    """Основная функция для генерации синтетических данных"""
    print("=== Generator sinteticheskih vremennyh ryadov ===")
    print("Metodologiya: Ornstein-Uhlenbeck process")
    print("Istochnik: 'Determining Optimal Trading Rules without Back-testing'\n")
    
    # Инициализация генератора
    generator = OrnsteinUhlenbeckGenerator()
    
    # Список файлов данных
    data_files = {
        'BTCUSDT': 'data/BTCUSDT_1m_20210106_to_20250825.csv',
        'ETHUSDT': 'data/ETHUSDT_1m_20210107_to_20250826.csv',
        'SOLUSDT': 'data/SOLUSDT_1m_20210101_to_20250820.csv'
    }
    
    # Параметры генерации
    n_paths = 100  # Количество траекторий
    path_length = 1440*30*3  # Длина траектории (1 день = 1440 минут)
    
    for symbol, filepath in data_files.items():
        print(f"\n{'='*50}")
        print(f"Obrabotka {symbol}")
        print(f"{'='*50}")
        
        try:
            # Загрузка данных
            print(f"Zagruzka dannyh iz {filepath}...")
            df = load_crypto_data(filepath)
            print(f"Zagruzheno {len(df)} zapisej")
            
            # Использование цен закрытия для оценки параметров
            close_prices = df['Close'].values
            
            # Оценка параметров OU-процесса
            print("Ocenka parametrov OU-processa...")
            generator.estimate_ou_parameters(close_prices, symbol)
            
            # Генерация синтетических траекторий
            print("Generaciya sinteticheskih traektorij...")
            synthetic_paths = generator.generate_synthetic_paths(
                symbol=symbol,
                n_paths=n_paths,
                path_length=path_length,
                seed=42  # Для воспроизводимости
            )
            
            # Сохранение данных
            print("Sohranenie sinteticheskih dannyh...")
            generator.save_synthetic_data(symbol, synthetic_paths)
            
            # Анализ и сохранение характерных путей
            print("Analiz i sohranenie harakternyh putej...")
            generator.analyze_and_save_characteristic_paths(symbol, synthetic_paths)
            
        except Exception as e:
            print(f"Oshibka pri obrabotke {symbol}: {str(e)}")
            continue
    
    print(f"\n{'='*50}")
    print("Generaciya zavershena!")
    print("Sinteticheskie dannye sohraneny v direktorii data/synthetic/")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
