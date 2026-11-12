"""Hàm tiện ích dùng chung cho các controller giao diện (không chứa logic ML)."""
import math

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication, QMessageBox, QTableWidgetItem

BLUE, TEAL, AMBER, GREY = "#1f6feb", "#12b5a6", "#f59e0b", "#9aa5b1"


def apply_light_theme():
    """Luôn dùng giao diện sáng (style Fusion + bảng màu sáng), kể cả khi Windows bật Dark mode.
    Nếu không, Windows tô nền tối cho ô văn bản, danh sách xổ xuống, lịch… trong khi chữ màu tối -> không đọc được."""
    app = QApplication.instance()
    if app is None:
        return
    app.setStyle("Fusion")
    palette = QPalette()
    colors = {
        QPalette.ColorRole.Window: "#f3f6fb", QPalette.ColorRole.WindowText: "#1f2d3d",
        QPalette.ColorRole.Base: "#ffffff", QPalette.ColorRole.AlternateBase: "#f7f9fc",
        QPalette.ColorRole.Text: "#1f2d3d", QPalette.ColorRole.Button: "#ffffff",
        QPalette.ColorRole.ButtonText: "#1f2d3d", QPalette.ColorRole.ToolTipBase: "#ffffff",
        QPalette.ColorRole.ToolTipText: "#1f2d3d", QPalette.ColorRole.PlaceholderText: "#8a96a8",
        QPalette.ColorRole.Highlight: "#1f6feb", QPalette.ColorRole.HighlightedText: "#ffffff",
        QPalette.ColorRole.Link: "#1f6feb", QPalette.ColorRole.BrightText: "#ffffff",
    }
    for role, color in colors.items():
        palette.setColor(role, QColor(color))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.WindowText, QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor("#9aa5b1"))
    app.setPalette(palette)


def fmt_money(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:,.0f} €"


def fmt_pct(value, digits=1):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:+.{digits}f}%"


def showDataIntoTableWidget(table, df, max_rows=None):
    """Đổ DataFrame vào QTableWidget (giống bài SalesMainWindowEx đã học). NaN hiển thị 'N/A'."""
    if max_rows is not None:
        df = df.head(max_rows)
    table.clear()
    table.setRowCount(0)
    table.setColumnCount(len(df.columns))
    table.setHorizontalHeaderLabels([str(c) for c in df.columns])
    for row, values in enumerate(df.itertuples(index=False)):
        table.insertRow(row)
        for col, v in enumerate(values):
            if isinstance(v, float):
                text = "N/A" if math.isnan(v) else f"{v:,.4f}" if abs(v) < 1 else f"{v:,.2f}"
            else:
                text = str(v)
            table.setItem(row, col, QTableWidgetItem(text))
    table.resizeColumnsToContents()


def show_error(parent, message, title="Lỗi"):
    QMessageBox.warning(parent, title, str(message))


class ChartWidget:
    """Figure matplotlib + thanh công cụ, gắn vào một QVBoxLayout đặt sẵn trong Qt Designer."""

    def __init__(self, layout, parent):
        self.figure = Figure(figsize=(6, 3), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, parent)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def new_axes(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        return ax

    def message(self, text):
        ax = self.new_axes()
        ax.axis("off")
        ax.text(0.5, 0.5, text, ha="center", va="center", color="#5b6b7f")
        self.draw()

    def draw(self):
        self.canvas.draw_idle()


class Worker(QThread):
    """Chạy một hàm tốn thời gian ở luồng riêng để giao diện không bị treo."""
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs

    def run(self):
        try:
            self.succeeded.emit(self.fn(*self.args, **self.kwargs))
        except Exception as exc:  # noqa: BLE001 – mọi lỗi được báo lên giao diện
            self.failed.emit(f"{type(exc).__name__}: {exc}")
