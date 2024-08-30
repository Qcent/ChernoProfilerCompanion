# https://gist.github.com/TheCherno/31f135eea6ee729ab5f26a6908eb3a5e

import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from scipy import stats
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QTableView, QHBoxLayout
from PyQt5.QtCore import QAbstractTableModel, Qt


# Load JSON data from file
def load_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)


# Convert DataFrame to a Table Model
class PandasModel(QAbstractTableModel):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self._data = data

    def rowCount(self, parent=None):
        return len(self._data)

    def columnCount(self, parent=None):
        return len(self._data.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        value = self._data.iloc[index.row(), index.column()]
        return str(value)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            header = self._data.columns[section]
            if 'dur' in header.lower():
                return header + ' (μs)'  # Append microseconds to duration columns
            return header
        return str(section + 1)


# Create a Matplotlib Figure for the graph
class PlotCanvas(FigureCanvas):
    def __init__(self, parent):
        fig, ax = plt.subplots(figsize=(10, 6))
        super().__init__(fig)
        self.parent = parent
        self.ax = ax
        self.plot()

        # Create toolbar and add it to the layout
        self.toolbar = NavigationToolbar(self, parent)
        self.toolbar.hide()  # Hide initially, will show based on view

    def plot(self):
        self.ax.clear()
        df = pd.DataFrame(self.parent.data['traceEvents'])
        df['relative_ts'] = df['ts'] - df['ts'].min()
        for name, group in df.groupby('name'):
            self.ax.plot(group['relative_ts'], group['dur'], marker='o', linestyle='-', label=name)
        self.ax.set_xlabel('Relative Time (ms)')
        self.ax.set_ylabel('Duration (μs)')
        self.ax.set_title('Function Durations Over Time')
        self.ax.legend()
        self.ax.grid(True)
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self, filepath):
        super().__init__()
        self.setWindowTitle('Cherno Profiler Companion')
        self.resize(1200, 800)  # Set window size

        # Load data
        self.filepath = filepath
        self.data = load_json(filepath)

        df = pd.DataFrame(self.data['traceEvents'])
        df['relative (μs)'] = df['ts'] - df['ts'].min()
        # Specify the desired order of columns
        new_order = ['cat', 'name', 'dur', 'relative (μs)', 'tid', 'ts']
        # Reorder the columns
        df = df[new_order]


        # Create Widgets
        self.table_view = QTableView()
        self.table_model = PandasModel(df.sort_values(by='ts', ascending=True))
        self.table_view.setModel(self.table_model)

        self.plot_canvas = PlotCanvas(self)

        self.stats_view = QTableView()

        stats_df = self.calculate_stats(pd.DataFrame(self.data['traceEvents']))
        self.stats_model = PandasModel(stats_df)
        self.stats_view.setModel(self.stats_model)

        self.show_table_button = QPushButton('Show Table')
        self.show_stats_button = QPushButton('Show Stats')
        self.show_graph_button = QPushButton('Show Graph')

        self.show_table_button.clicked.connect(self.show_table_view)
        self.show_stats_button.clicked.connect(self.show_stats_view)
        self.show_graph_button.clicked.connect(self.show_graph_view)

        # Layout
        layout = QVBoxLayout()

        # Add buttons to the top
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.show_table_button)
        button_layout.addWidget(self.show_stats_button)
        button_layout.addWidget(self.show_graph_button)
        layout.addLayout(button_layout)

        # Add plot canvas and toolbar
        layout.addWidget(self.plot_canvas.toolbar)  # Toolbar added here
        layout.addWidget(self.plot_canvas)
        layout.addWidget(self.table_view)
        layout.addWidget(self.stats_view)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Default view
        self.show_stats_view()

    def show_table_view(self):
        self.plot_canvas.toolbar.hide()
        self.table_view.show()
        self.stats_view.hide()
        self.plot_canvas.hide()
        self.show_stats_button.show()
        self.show_graph_button.show()
        self.show_table_button.hide()
        self.current_view = 'table'

    def show_stats_view(self):
        self.plot_canvas.toolbar.hide()
        self.table_view.hide()
        self.stats_view.show()
        self.plot_canvas.hide()
        self.show_stats_button.hide()
        self.show_graph_button.show()
        self.show_table_button.show()
        self.current_view = 'stats'

    def show_graph_view(self):
        self.table_view.hide()
        self.stats_view.hide()
        self.plot_canvas.show()
        self.plot_canvas.toolbar.show()  # Ensure toolbar is visible
        self.show_stats_button.show()
        self.show_table_button.show()
        self.show_graph_button.hide()
        self.current_view = 'graph'

    def calculate_stats(self, df):
        # Calculate statistics and count of occurrences
        stats_summary = df.groupby('name')['dur'].agg([
            ('Count', 'count'),
            ('Max', 'max'),
            ('Min', 'min'),
            ('Mean', 'mean'),
            ('Median', 'median'),
            ('Mode', lambda x: self.calculate_mode(x))
        ]).reset_index()

        # Update header to denote microseconds
        stats_summary.columns = ['Function', 'Count', 'Max (μs)', 'Min (μs)', 'Mean (μs)', 'Median (μs)', 'Mode (μs)']

        # Align numbers to the right and format them
        for col in stats_summary.columns[1:]:
            if col == 'Count':
                stats_summary[col] = stats_summary[col].apply(lambda x: x)
            else:
                stats_summary[col] = stats_summary[col].apply(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)

        # Sort by Count
        stats_summary = stats_summary.sort_values(by='Count', ascending=False)

        return stats_summary

    def calculate_mode(self, series):
        try:
            mode_result = stats.mode(series, keepdims=False)
            if isinstance(mode_result.mode, np.ndarray) and isinstance(mode_result.count, np.ndarray):
                return mode_result.mode[0] if mode_result.count[0] > 0 else 'No mode'
            else:
                return mode_result.mode if mode_result.count > 0 else 'No mode'
        except Exception as e:
            print(f"Error calculating mode: {e}")
            return 'Error'


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Parse JSON data and generate a spreadsheet and graphs.')
    parser.add_argument('filepath', type=str, help='Path to the JSON file')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    main_window = MainWindow(args.filepath)
    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
