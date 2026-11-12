"""Chạy chương trình Store Revenue Forecast & Planning Studio (giống file TestSalesMainWindowEx.py đã học).

PyCharm: chuột phải main.py -> Run.   Dòng lệnh:  python main.py
"""
import sys

from PyQt6.QtWidgets import QApplication, QMainWindow

from ui.MainWindowEx import MainWindowEx

app = QApplication(sys.argv)
myui = MainWindowEx()
myui.setupUi(QMainWindow())
myui.showWindow()
sys.exit(app.exec())
