import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля графиков
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SyntheticDataVisualizer:
    """
    Класс для визуализации синтетических временных рядов
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.synthetic_dir = self.data_dir / "synthetic"
        self.original_data = {}
        self.synthetic_data = {}
        
    def load_original_data(self, symbol: str) -> pd.DataFrame:
        """Загрузка оригинальных данных"""
        file_mapping = {
            'BTCUSDT': 'BTCUSDT_1m_20210106_to_20250825.csv',
            'ETHUSDT': 'ETHUSDT_1m_20210107_to_20250826.csv', 
            'SOLUSDT': 'SOLUSDT_1m_20210101_to_20250820.csv'
        }
        
        filepath = self.data_dir / file_mapping[symbol]
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    
    def load_synthetic_paths(self, symbol: str, n_paths: int = 5) -> List[pd.DataFrame]:
        """Загрузка нескольких синтетических траекторий"""
        paths = []
        for i in range(n_paths):
            filepath = self.synthetic_dir / f"{symbol}_synthetic_path_{i:03d}.csv"
            if filepath.exists():
                df = pd.read_csv(filepath)
                df['Date'] = pd.to_datetime(df['Date'])
                paths.append(df)
        return paths
    
    def plot_price_comparison(self, symbol: str, n_paths: int = 50, 
                            original_sample_size: int = 1440):
        """
        Сравнение оригинальных и синтетических цен
        """
        # Загрузка данных
        original_df = self.load_original_data(symbol)
        synthetic_paths = self.load_synthetic_paths(symbol, n_paths)
        
        # Выборка из оригинальных данных (последние 1440 точек)
        original_sample = original_df.tail(original_sample_size).copy()
        original_sample = original_sample.reset_index(drop=True)
        
        # Создание графика
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # График 1: Оригинальные данные
        ax1.plot(original_sample.index, original_sample['Close'], 
                color='black', linewidth=2, label='Originalnye dannye', alpha=0.8)
        ax1.set_title(f'{symbol} - Originalnye dannye (poslednie {original_sample_size} barov)', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cena', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # График 2: Синтетические траектории
        colors = plt.cm.Set3(np.linspace(0, 1, len(synthetic_paths)))
        
        for i, (path_df, color) in enumerate(zip(synthetic_paths, colors)):
            ax2.plot(path_df.index, path_df['Close'], 
                    color=color, linewidth=1.5, alpha=0.7,
                    label=f'Sinteticheskij put {i+1}')
        
        ax2.set_title(f'{symbol} - Sinteticheskie traektorii ({len(synthetic_paths)} putej)', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Vremya (minuty)', fontsize=12)
        ax2.set_ylabel('Cena', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Сохранение графика
        output_path = self.synthetic_dir / f"{symbol}_price_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Graf sohranyon: {output_path}")
        
        plt.show()
    
    def plot_returns_distribution(self, symbol: str, n_paths: int = 10):
        """
        Сравнение распределений доходностей
        """
        # Загрузка данных
        original_df = self.load_original_data(symbol)
        synthetic_paths = self.load_synthetic_paths(symbol, n_paths)
        
        # Расчет доходностей для оригинальных данных
        original_returns = np.log(original_df['Close'] / original_df['Close'].shift(1)).dropna()
        
        # Расчет доходностей для синтетических данных
        synthetic_returns = []
        for path_df in synthetic_paths:
            returns = np.log(path_df['Close'] / path_df['Close'].shift(1)).dropna()
            synthetic_returns.extend(returns.values)
        
        # Создание графика
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Гистограммы распределений
        ax1.hist(original_returns, bins=100, alpha=0.7, density=True, 
                color='blue', label='Originalnye dannye')
        ax1.hist(synthetic_returns, bins=100, alpha=0.7, density=True, 
                color='red', label='Sinteticheskie dannye')
        ax1.set_title(f'{symbol} - Raspredelenie dohodnostej', fontweight='bold')
        ax1.set_xlabel('Log-dohodnost')
        ax1.set_ylabel('Plotnost')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        original_sample = np.random.choice(original_returns, size=min(1000, len(original_returns)))
        synthetic_sample = np.random.choice(synthetic_returns, size=min(1000, len(synthetic_returns)))
        
        stats.probplot(original_sample, dist="norm", plot=ax2)
        ax2.set_title(f'{symbol} - Q-Q plot (normalnost)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Сохранение
        output_path = self.synthetic_dir / f"{symbol}_returns_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Graf sohranyon: {output_path}")
        
        plt.show()
        
        # Статистики
        print(f"\n{symbol} - Statistiki dohodnostej:")
        print(f"Originalnye dannye:")
        print(f"  Srednee: {np.mean(original_returns):.6f}")
        print(f"  Std: {np.std(original_returns):.6f}")
        print(f"  Skewness: {stats.skew(original_returns):.4f}")
        print(f"  Kurtosis: {stats.kurtosis(original_returns):.4f}")
        
        print(f"Sinteticheskie dannye:")
        print(f"  Srednee: {np.mean(synthetic_returns):.6f}")
        print(f"  Std: {np.std(synthetic_returns):.6f}")
        print(f"  Skewness: {stats.skew(synthetic_returns):.4f}")
        print(f"  Kurtosis: {stats.kurtosis(synthetic_returns):.4f}")
    
    def plot_volatility_analysis(self, symbol: str, n_paths: int = 5, window: int = 60):
        """
        Анализ волатильности
        """
        # Загрузка данных
        original_df = self.load_original_data(symbol)
        synthetic_paths = self.load_synthetic_paths(symbol, n_paths)
        
        # Расчет скользящей волатильности для оригинальных данных
        original_returns = np.log(original_df['Close'] / original_df['Close'].shift(1))
        original_vol = original_returns.rolling(window=window).std() * np.sqrt(1440)  # Дневная волатильность
        
        # Выборка из оригинальных данных
        sample_size = 1440
        original_vol_sample = original_vol.tail(sample_size)
        
        # Создание графика
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # График 1: Сравнение волатильности
        axes[0].plot(range(len(original_vol_sample)), original_vol_sample, 
                    color='black', linewidth=2, label='Originalnaya volatilnost')
        
        for i, path_df in enumerate(synthetic_paths):
            returns = np.log(path_df['Close'] / path_df['Close'].shift(1))
            vol = returns.rolling(window=window).std() * np.sqrt(1440)
            axes[0].plot(range(len(vol)), vol, alpha=0.7, linewidth=1, 
                        label=f'Sinteticheskij put {i+1}')
        
        axes[0].set_title('Sravnenie volatilnosti', fontweight='bold')
        axes[0].set_ylabel('Dnevnaya volatilnost')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # График 2: Распределение волатильности
        all_synthetic_vol = []
        for path_df in synthetic_paths:
            returns = np.log(path_df['Close'] / path_df['Close'].shift(1))
            vol = returns.rolling(window=window).std() * np.sqrt(1440)
            all_synthetic_vol.extend(vol.dropna().values)
        
        axes[1].hist(original_vol_sample.dropna(), bins=30, alpha=0.7, 
                    density=True, color='blue', label='Originalnye')
        axes[1].hist(all_synthetic_vol, bins=30, alpha=0.7, 
                    density=True, color='red', label='Sinteticheskie')
        axes[1].set_title('Raspredelenie volatilnosti', fontweight='bold')
        axes[1].set_xlabel('Volatilnost')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # График 3: Автокорреляция доходностей
        from statsmodels.tsa.stattools import acf
        
        original_sample_returns = np.log(original_df['Close'].tail(sample_size) / 
                                       original_df['Close'].tail(sample_size).shift(1)).dropna()
        original_acf = acf(original_sample_returns, nlags=50)
        
        axes[2].plot(range(len(original_acf)), original_acf, 
                    color='black', linewidth=2, label='Originalnye', marker='o', markersize=3)
        
        for i, path_df in enumerate(synthetic_paths[:3]):  # Только первые 3 для читаемости
            returns = np.log(path_df['Close'] / path_df['Close'].shift(1)).dropna()
            synthetic_acf = acf(returns, nlags=50)
            axes[2].plot(range(len(synthetic_acf)), synthetic_acf, 
                        alpha=0.7, label=f'Sinteticheskij {i+1}', marker='s', markersize=2)
        
        axes[2].set_title('Avtokorrelyaciya dohodnostej', fontweight='bold')
        axes[2].set_xlabel('Lag')
        axes[2].set_ylabel('ACF')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # График 4: Статистики по путям
        path_stats = []
        for i, path_df in enumerate(synthetic_paths):
            returns = np.log(path_df['Close'] / path_df['Close'].shift(1)).dropna()
            stats_dict = {
                'Path': i+1,
                'Mean': np.mean(returns),
                'Std': np.std(returns),
                'Min': np.min(path_df['Close']),
                'Max': np.max(path_df['Close'])
            }
            path_stats.append(stats_dict)
        
        stats_df = pd.DataFrame(path_stats)
        
        # Барплот статистик
        x = np.arange(len(stats_df))
        width = 0.35
        
        axes[3].bar(x - width/2, stats_df['Mean'] * 10000, width, 
                   label='Mean Return (bp)', alpha=0.7)
        axes[3].bar(x + width/2, stats_df['Std'] * 100, width, 
                   label='Volatility (%)', alpha=0.7)
        
        axes[3].set_title('Statistiki po putyam', fontweight='bold')
        axes[3].set_xlabel('Nomer puti')
        axes[3].set_xticks(x)
        axes[3].set_xticklabels([f'Put {i}' for i in stats_df['Path']])
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Сохранение
        output_path = self.synthetic_dir / f"{symbol}_volatility_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Graf sohranyon: {output_path}")
        
        plt.show()
    
    def create_summary_report(self, symbols: List[str] = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']):
        """
        Создание сводного отчета по всем символам
        """
        fig, axes = plt.subplots(len(symbols), 2, figsize=(15, 5*len(symbols)))
        if len(symbols) == 1:
            axes = axes.reshape(1, -1)
        
        for i, symbol in enumerate(symbols):
            # Загрузка данных
            synthetic_paths = self.load_synthetic_paths(symbol, 10)
            
            # График цен
            colors = plt.cm.tab10(np.linspace(0, 1, len(synthetic_paths)))
            for j, (path_df, color) in enumerate(zip(synthetic_paths[:5], colors)):
                axes[i, 0].plot(path_df.index, path_df['Close'], 
                              color=color, alpha=0.7, linewidth=1,
                              label=f'Put {j+1}' if i == 0 else "")
            
            axes[i, 0].set_title(f'{symbol} - Sinteticheskie ceny', fontweight='bold')
            axes[i, 0].set_ylabel('Cena')
            axes[i, 0].grid(True, alpha=0.3)
            if i == 0:
                axes[i, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Распределение конечных цен
            final_prices = [path_df['Close'].iloc[-1] for path_df in synthetic_paths]
            axes[i, 1].hist(final_prices, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[i, 1].set_title(f'{symbol} - Raspredelenie konechnyh cen', fontweight='bold')
            axes[i, 1].set_xlabel('Konechnaya cena')
            axes[i, 1].set_ylabel('Chastota')
            axes[i, 1].grid(True, alpha=0.3)
            
            # Статистики
            mean_price = np.mean(final_prices)
            std_price = np.std(final_prices)
            axes[i, 1].axvline(mean_price, color='red', linestyle='--', 
                             label=f'Srednee: {mean_price:.2f}')
            axes[i, 1].legend()
        
        plt.tight_layout()
        
        # Сохранение
        output_path = self.synthetic_dir / "summary_report.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Svodnyj otchot sohranyon: {output_path}")
        
        plt.show()

def main():
    """Основная функция для создания визуализаций"""
    print("=== Vizualizaciya sinteticheskih vremennyh ryadov ===\n")
    
    # Инициализация визуализатора
    visualizer = SyntheticDataVisualizer()
    
    # Проверка наличия синтетических данных
    if not visualizer.synthetic_dir.exists():
        print("Oshibka: Direktoriya s sinteticheskimi dannymi ne najdena!")
        print("Snachala zapustite synthetic_data_generator.py")
        return
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    
    # Создание визуализаций для каждого символа
    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"Sozdanie vizualizacij dlya {symbol}")
        print(f"{'='*50}")
        
        try:
            # Сравнение цен
            print("1. Sravnenie cen...")
            visualizer.plot_price_comparison(symbol, n_paths=50)
            
            # Распределение доходностей
            print("2. Analiz dohodnostej...")
            visualizer.plot_returns_distribution(symbol, n_paths=50)
            
            # Анализ волатильности
            print("3. Analiz volatilnosti...")
            visualizer.plot_volatility_analysis(symbol, n_paths=50)
            
        except Exception as e:
            print(f"Oshibka pri sozdanii vizualizacij dlya {symbol}: {str(e)}")
            continue
    
    # Сводный отчет
    print(f"\n{'='*50}")
    print("Sozdanie svodnogo otchota...")
    print(f"{'='*50}")
    
    try:
        visualizer.create_summary_report(symbols)
    except Exception as e:
        print(f"Oshibka pri sozdanii svodnogo otchota: {str(e)}")
    
    print(f"\n{'='*50}")
    print("Vizualizaciya zavershena!")
    print("Grafiki sohraneny v direktorii data/synthetic/")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
