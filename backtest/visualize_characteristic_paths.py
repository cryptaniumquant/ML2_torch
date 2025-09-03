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

class CharacteristicPathsVisualizer:
    """
    Класс для визуализации характерных синтетических траекторий
    (лучшие, нейтральные, худшие пути)
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.synthetic_dir = self.data_dir / "synthetic"
        self.characteristic_dir = self.synthetic_dir / "characteristic_paths"
        
    def load_characteristic_paths(self, symbol: str) -> Dict[str, List[pd.DataFrame]]:
        """Загрузка характерных путей для символа"""
        paths = {
            'best': [],
            'neutral': [],
            'worst': []
        }
        
        for category in ['best', 'neutral', 'worst']:
            for rank in range(1, 4):  # rank 1, 2, 3
                pattern = f"{symbol}_{category}_rank_{rank}_*.csv"
                files = list(self.characteristic_dir.glob(pattern))
                
                if files:
                    filepath = files[0]  # Берем первый найденный файл
                    df = pd.read_csv(filepath)
                    df['Date'] = pd.to_datetime(df['Date'])
                    
                    # Извлекаем доходность из имени файла
                    filename = filepath.name
                    return_str = filename.split('_return_')[1].split('pct.csv')[0]
                    return_pct = float(return_str)
                    df['return_pct'] = return_pct
                    df['category'] = category
                    df['rank'] = rank
                    df['filepath'] = str(filepath)
                    
                    paths[category].append(df)
        
        return paths
    
    def plot_characteristic_comparison(self, symbol: str):
        """
        Сравнение характерных путей для одного символа
        """
        # Загрузка данных
        paths = self.load_characteristic_paths(symbol)
        
        # Проверка наличия данных
        total_paths = sum(len(paths[cat]) for cat in paths)
        if total_paths == 0:
            print(f"Oshibka: Harakternyje puti dlya {symbol} ne najdeny!")
            return
        
        # Создание графика
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{symbol} - Analiz harakternyh traektorij', fontsize=16, fontweight='bold')
        
        # Цвета для категорий
        colors = {
            'best': ['#2E8B57', '#32CD32', '#90EE90'],      # Зеленые оттенки
            'neutral': ['#4682B4', '#87CEEB', '#B0C4DE'],   # Синие оттенки  
            'worst': ['#DC143C', '#FF6347', '#FFA07A']      # Красные оттенки
        }
        
        # График 1: Все траектории
        ax1 = axes[0, 0]
        for category, category_paths in paths.items():
            for i, path_df in enumerate(category_paths):
                color = colors[category][i]
                return_pct = path_df['return_pct'].iloc[0]
                label = f'{category.title()} {i+1}: {return_pct:.2f}%'
                
                ax1.plot(path_df.index, path_df['Close'], 
                        color=color, linewidth=2, alpha=0.8, label=label)
        
        ax1.set_title('Vse harakternyje traektorii', fontweight='bold')
        ax1.set_xlabel('Vremya (minuty)')
        ax1.set_ylabel('Cena')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # График 2: Распределение доходностей
        ax2 = axes[0, 1]
        all_returns = []
        all_categories = []
        
        for category, category_paths in paths.items():
            for path_df in category_paths:
                all_returns.append(path_df['return_pct'].iloc[0])
                all_categories.append(category)
        
        # Создание барплота
        categories = ['worst', 'neutral', 'best']
        cat_returns = {cat: [] for cat in categories}
        
        for ret, cat in zip(all_returns, all_categories):
            cat_returns[cat].append(ret)
        
        x_pos = []
        heights = []
        colors_bar = []
        labels_bar = []
        
        for i, category in enumerate(categories):
            for j, ret in enumerate(cat_returns[category]):
                x_pos.append(i * 4 + j)
                heights.append(ret)
                colors_bar.append(colors[category][j])
                labels_bar.append(f'{category.title()} {j+1}')
        
        bars = ax2.bar(x_pos, heights, color=colors_bar, alpha=0.8)
        ax2.set_title('Dohodnost harakternyh putej', fontweight='bold')
        ax2.set_ylabel('Dohodnost (%)')
        ax2.set_xticks([1, 5, 9])
        ax2.set_xticklabels(['Worst', 'Neutral', 'Best'])
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Добавление значений на столбцы
        for bar, height in zip(bars, heights):
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.3),
                    f'{height:.2f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                    fontsize=9, fontweight='bold')
        
        # График 3: Волатильность путей
        ax3 = axes[1, 0]
        
        for category, category_paths in paths.items():
            for i, path_df in enumerate(category_paths):
                returns = np.log(path_df['Close'] / path_df['Close'].shift(1)).dropna()
                rolling_vol = returns.rolling(window=60).std() * np.sqrt(1440)  # Дневная волатильность
                
                color = colors[category][i]
                return_pct = path_df['return_pct'].iloc[0]
                label = f'{category.title()} {i+1}'
                
                ax3.plot(range(len(rolling_vol)), rolling_vol, 
                        color=color, linewidth=1.5, alpha=0.7, label=label)
        
        ax3.set_title('Volatilnost putej (60-minutnoe okno)', fontweight='bold')
        ax3.set_xlabel('Vremya (minuty)')
        ax3.set_ylabel('Dnevnaya volatilnost')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # График 4: Статистики по категориям
        ax4 = axes[1, 1]
        
        # Расчет статистик
        stats_data = []
        for category, category_paths in paths.items():
            returns_list = []
            volatilities = []
            
            for path_df in category_paths:
                returns = np.log(path_df['Close'] / path_df['Close'].shift(1)).dropna()
                returns_list.extend(returns.values)
                volatilities.append(np.std(returns))
            
            stats_data.append({
                'category': category,
                'mean_return': np.mean(returns_list) * 1440 * 100,  # Дневная доходность в %
                'volatility': np.mean(volatilities) * np.sqrt(1440) * 100,  # Дневная волатильность в %
                'total_return_mean': np.mean([df['return_pct'].iloc[0] for df in category_paths])
            })
        
        # Создание барплота статистик
        categories_order = ['worst', 'neutral', 'best']
        x = np.arange(len(categories_order))
        width = 0.25
        
        mean_returns = [next(s['mean_return'] for s in stats_data if s['category'] == cat) for cat in categories_order]
        volatilities = [next(s['volatility'] for s in stats_data if s['category'] == cat) for cat in categories_order]
        total_returns = [next(s['total_return_mean'] for s in stats_data if s['category'] == cat) for cat in categories_order]
        
        ax4.bar(x - width, mean_returns, width, label='Srednyaya dnevnaya dohodnost (%)', alpha=0.8, color='lightblue')
        ax4.bar(x, volatilities, width, label='Srednyaya dnevnaya volatilnost (%)', alpha=0.8, color='lightcoral')
        ax4.bar(x + width, [tr/30 for tr in total_returns], width, label='Obshaya dohodnost / 30 (%)', alpha=0.8, color='lightgreen')
        
        ax4.set_title('Statistiki po kategoriyam', fontweight='bold')
        ax4.set_ylabel('Znacheniye (%)')
        ax4.set_xticks(x)
        ax4.set_xticklabels([cat.title() for cat in categories_order])
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Сохранение
        output_path = self.characteristic_dir / f"{symbol}_characteristic_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Graf sohranyon: {output_path}")
        
        plt.show()
    
    def plot_all_symbols_comparison(self, symbols: List[str] = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']):
        """
        Сравнение характерных путей по всем символам
        """
        fig, axes = plt.subplots(len(symbols), 3, figsize=(18, 6*len(symbols)))
        if len(symbols) == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Sravnenie harakternyh putej po vsem monetam', fontsize=16, fontweight='bold')
        
        # Цвета для категорий
        colors = {
            'best': '#2E8B57',
            'neutral': '#4682B4', 
            'worst': '#DC143C'
        }
        
        for i, symbol in enumerate(symbols):
            paths = self.load_characteristic_paths(symbol)
            
            # График лучших путей
            ax_best = axes[i, 0]
            for j, path_df in enumerate(paths['best']):
                return_pct = path_df['return_pct'].iloc[0]
                ax_best.plot(path_df.index, path_df['Close'], 
                           color=colors['best'], alpha=0.7 - j*0.2, linewidth=2,
                           label=f'Best {j+1}: {return_pct:.2f}%')
            
            ax_best.set_title(f'{symbol} - Luchshie puti', fontweight='bold')
            ax_best.set_ylabel('Cena')
            ax_best.legend()
            ax_best.grid(True, alpha=0.3)
            
            # График нейтральных путей
            ax_neutral = axes[i, 1]
            for j, path_df in enumerate(paths['neutral']):
                return_pct = path_df['return_pct'].iloc[0]
                ax_neutral.plot(path_df.index, path_df['Close'], 
                              color=colors['neutral'], alpha=0.7 - j*0.2, linewidth=2,
                              label=f'Neutral {j+1}: {return_pct:.2f}%')
            
            ax_neutral.set_title(f'{symbol} - Nejtralnyje puti', fontweight='bold')
            ax_neutral.set_ylabel('Cena')
            ax_neutral.legend()
            ax_neutral.grid(True, alpha=0.3)
            
            # График худших путей
            ax_worst = axes[i, 2]
            for j, path_df in enumerate(paths['worst']):
                return_pct = path_df['return_pct'].iloc[0]
                ax_worst.plot(path_df.index, path_df['Close'], 
                            color=colors['worst'], alpha=0.7 - j*0.2, linewidth=2,
                            label=f'Worst {j+1}: {return_pct:.2f}%')
            
            ax_worst.set_title(f'{symbol} - Hudshie puti', fontweight='bold')
            ax_worst.set_ylabel('Cena')
            ax_worst.legend()
            ax_worst.grid(True, alpha=0.3)
            
            # Добавление подписи оси X только для нижнего ряда
            if i == len(symbols) - 1:
                for ax in [ax_best, ax_neutral, ax_worst]:
                    ax.set_xlabel('Vremya (minuty)')
        
        plt.tight_layout()
        
        # Сохранение
        output_path = self.characteristic_dir / "all_symbols_characteristic_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Svodnyj graf sohranyon: {output_path}")
        
        plt.show()
    
    def create_returns_distribution_analysis(self, symbols: List[str] = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']):
        """
        Анализ распределения доходностей характерных путей
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Analiz dohodnostej harakternyh putej', fontsize=16, fontweight='bold')
        
        all_data = []
        
        # Сбор данных по всем символам
        for symbol in symbols:
            paths = self.load_characteristic_paths(symbol)
            
            for category, category_paths in paths.items():
                for path_df in category_paths:
                    return_pct = path_df['return_pct'].iloc[0]
                    all_data.append({
                        'symbol': symbol,
                        'category': category,
                        'return_pct': return_pct
                    })
        
        df_all = pd.DataFrame(all_data)
        
        # График 1: Распределение по символам
        ax1 = axes[0, 0]
        for symbol in symbols:
            symbol_data = df_all[df_all['symbol'] == symbol]['return_pct']
            ax1.hist(symbol_data, bins=10, alpha=0.7, label=symbol, density=True)
        
        ax1.set_title('Raspredelenie po simvolam', fontweight='bold')
        ax1.set_xlabel('Dohodnost (%)')
        ax1.set_ylabel('Plotnost')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # График 2: Распределение по категориям
        ax2 = axes[0, 1]
        categories = ['worst', 'neutral', 'best']
        colors_cat = ['red', 'blue', 'green']
        
        for category, color in zip(categories, colors_cat):
            cat_data = df_all[df_all['category'] == category]['return_pct']
            ax2.hist(cat_data, bins=5, alpha=0.7, label=category.title(), 
                    color=color, density=True)
        
        ax2.set_title('Raspredelenie po kategoriyam', fontweight='bold')
        ax2.set_xlabel('Dohodnost (%)')
        ax2.set_ylabel('Plotnost')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # График 3: Boxplot по символам и категориям
        ax3 = axes[1, 0]
        
        # Подготовка данных для boxplot
        boxplot_data = []
        labels = []
        
        for symbol in symbols:
            for category in categories:
                data = df_all[(df_all['symbol'] == symbol) & (df_all['category'] == category)]['return_pct']
                if len(data) > 0:
                    boxplot_data.append(data.values)
                    labels.append(f'{symbol}\n{category}')
        
        bp = ax3.boxplot(boxplot_data, labels=labels, patch_artist=True)
        
        # Раскраска boxplot
        colors_box = []
        for symbol in symbols:
            colors_box.extend(['lightcoral', 'lightblue', 'lightgreen'])
        
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.set_title('Boxplot po simvolam i kategoriyam', fontweight='bold')
        ax3.set_ylabel('Dohodnost (%)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # График 4: Сводная таблица статистик
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Создание сводной таблицы
        stats_table = []
        for symbol in symbols:
            symbol_data = df_all[df_all['symbol'] == symbol]
            
            stats_row = [symbol]
            for category in categories:
                cat_data = symbol_data[symbol_data['category'] == category]['return_pct']
                if len(cat_data) > 0:
                    stats_row.append(f'{cat_data.mean():.2f}%')
                else:
                    stats_row.append('N/A')
            
            stats_table.append(stats_row)
        
        # Добавление общей статистики
        stats_table.append(['OVERALL'] + [f'{df_all[df_all["category"] == cat]["return_pct"].mean():.2f}%' 
                                         for cat in categories])
        
        table = ax4.table(cellText=stats_table,
                         colLabels=['Symbol', 'Worst', 'Neutral', 'Best'],
                         cellLoc='center',
                         loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        ax4.set_title('Srednyaya dohodnost po kategoriyam', fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Сохранение
        output_path = self.characteristic_dir / "returns_distribution_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Analiz dohodnostej sohranyon: {output_path}")
        
        plt.show()

def main():
    """Основная функция для создания визуализаций характерных путей"""
    print("=== Vizualizaciya harakternyh putej ===\n")
    
    # Инициализация визуализатора
    visualizer = CharacteristicPathsVisualizer()
    
    # Проверка наличия характерных путей
    if not visualizer.characteristic_dir.exists():
        print("Oshibka: Direktoriya s harakternymi putyami ne najdena!")
        print("Snachala zapustite synthetic_data_generator.py s novoj funkcionalnostyu")
        return
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    
    # Создание визуализаций для каждого символа
    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"Analiz harakternyh putej dlya {symbol}")
        print(f"{'='*50}")
        
        try:
            visualizer.plot_characteristic_comparison(symbol)
        except Exception as e:
            print(f"Oshibka pri analize {symbol}: {str(e)}")
            continue
    
    # Сравнение по всем символам
    print(f"\n{'='*50}")
    print("Sravnenie po vsem simvolam...")
    print(f"{'='*50}")
    
    try:
        visualizer.plot_all_symbols_comparison(symbols)
    except Exception as e:
        print(f"Oshibka pri sravnenii: {str(e)}")
    
    # Анализ распределения доходностей
    print(f"\n{'='*50}")
    print("Analiz raspredeleniya dohodnostej...")
    print(f"{'='*50}")
    
    try:
        visualizer.create_returns_distribution_analysis(symbols)
    except Exception as e:
        print(f"Oshibka pri analize dohodnostej: {str(e)}")
    
    print(f"\n{'='*50}")
    print("Vizualizaciya harakternyh putej zavershena!")
    print("Grafiki sohraneny v direktorii data/synthetic/characteristic_paths/")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
