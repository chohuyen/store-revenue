"""Kiểm thử giao diện ở chế độ offscreen (không cần màn hình)."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
QtWidgets = pytest.importorskip("PyQt6.QtWidgets")


@pytest.fixture(scope="module")
def window():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    from ui.MainWindowEx import MainWindowEx
    w = MainWindowEx()                      # giống cách chạy trong main.py
    w.setupUi(QtWidgets.QMainWindow())
    yield w
    for worker in list(w._workers):
        worker.wait(10000)
    w.MainWindow.close()
    app.processEvents()


def test_generated_file_matches_ui(tmp_path):
    """ui/MainWindow.py phải đúng là kết quả pyuic6 của MainWindow.ui (không sửa tay)."""
    out = tmp_path / "gen.py"
    subprocess.run([sys.executable, "-m", "PyQt6.uic.pyuic", "-x", str(ROOT / "ui/MainWindow.ui"), "-o", str(out)],
                   check=True)
    strip = lambda text: [l for l in text.splitlines() if "Form implementation generated" not in l]  # noqa: E731
    assert strip(out.read_text(encoding="utf-8")) == strip((ROOT / "ui/MainWindow.py").read_text(encoding="utf-8"))


def test_generated_file_has_no_business_logic():
    text = (ROOT / "ui/MainWindow.py").read_text(encoding="utf-8")
    assert "from src" not in text and "import src" not in text and ".fit(" not in text


def test_menu_switches_pages(window):
    buttons = [window.pushButtonMenuDashboard, window.pushButtonMenuInput, window.pushButtonMenuResults,
               window.pushButtonMenuPerformance, window.pushButtonMenuWhatIf, window.pushButtonMenuPlanning,
               window.pushButtonMenuAbout]
    for i, b in enumerate(buttons):
        b.click()
        assert window.stackedWidgetPages.currentIndex() == i


def test_single_prediction_flow(window):
    window.comboBoxStore.setCurrentText("262")
    window.comboBoxPromo.setCurrentIndex(1)
    window.pushButtonPredictSingle.click()
    assert window.last_result is not None and len(window.last_result) == 1
    assert window.stackedWidgetPages.currentIndex() == window.PAGE_RESULTS
    assert window.tableWidgetResults.rowCount() == 1


def test_whatif_flow_and_disclaimer(window):
    window.comboBoxWIStore.setCurrentText("1")
    window.comboBoxWIScenPromo.setCurrentIndex(1)
    window.pushButtonRunWhatIf.click()
    assert window.tableWidgetWhatIf.rowCount() == 2
    assert "nhân quả" in window.labelWIDisclaimer.text()
    assert window.labelWIPctValue.text() not in ("", "—")


def test_csv_prediction_runs_in_background_thread(window):
    from PyQt6.QtCore import QEventLoop, QTimer
    from src import config
    window.forecast.load_csv(config.DATA_SAMPLE / "sample_input_with_actual.csv")
    assert window.forecast.csv_df is not None
    window.last_result = None
    window.forecast.predict_csv()
    assert window._workers, "Dự báo CSV phải chạy trong QThread"
    for _ in range(80):
        loop = QEventLoop()
        QTimer.singleShot(100, loop.quit)
        loop.exec()
        if window.last_result is not None:
            break
    assert window.last_result is not None and len(window.last_result) == 20


def test_invalid_csv_shows_message_not_crash(window, tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("Store,Date\n1,2015-08-01\n", encoding="utf-8")
    window.forecast.load_csv(bad)
    assert window.forecast.csv_df is None
    assert "chưa hợp lệ" in window.labelValidationStatus.text()


def test_mainwindowex_runs_as_script():
    """Chạy trực tiếp ui/MainWindowEx.py (như bấm Run trong PyCharm) phải mở được cửa sổ, không lỗi import."""
    code = ("import runpy, sys; from PyQt6.QtCore import QTimer; from PyQt6.QtWidgets import QApplication;"
            "sys.argv=['MainWindowEx.py'];"
            "orig=QApplication.exec;"
            "QApplication.exec=lambda *a: (QTimer.singleShot(1500, QApplication.quit), orig())[1];"
            "runpy.run_path(r'%s', run_name='__main__')" % (ROOT / "ui" / "MainWindowEx.py"))
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT / "ui", capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-1500:]


def _wait_until(condition, seconds=20):
    from PyQt6.QtCore import QEventLoop, QTimer
    for _ in range(seconds * 10):
        if condition():
            return True
        loop = QEventLoop()
        QTimer.singleShot(100, loop.quit)
        loop.exec()
    return condition()


def test_planning_flow(window):
    window.lineEditPlanStores.setText("1, 262")
    window.comboBoxPlanGranularity.setCurrentText("Tháng")
    window.planning.result = None
    window.pushButtonRunPlan.click()
    assert _wait_until(lambda: window.planning.result is not None)
    assert window.tableWidgetPlanRanking.rowCount() == 2
    assert window.tableWidgetPlan.rowCount() == 4          # 2 tháng (08, 09) × 2 cửa hàng
    assert "ca" in window.labelPlanShiftsValue.text()


def test_planning_invalid_store_shows_message_not_crash(window, monkeypatch):
    shown = []
    monkeypatch.setattr("ui.pages.planning_controller.show_error", lambda *a: shown.append(a))
    window.lineEditPlanStores.setText("817")        # không có trong test.csv
    window.pushButtonRunPlan.click()
    assert _wait_until(lambda: bool(shown))
    assert "817" in str(shown[0])


def test_results_granularity_switch(window):
    window.comboBoxStore.setCurrentText("1")
    window.pushButtonPredictSingle.click()
    for text in ["Ngày", "Tuần", "Tháng"]:
        window.comboBoxResGranularity.setCurrentText(text)     # không lỗi khi đổi kỳ


def test_light_theme_even_in_dark_mode(window):
    """Bảng màu ứng dụng luôn sáng: nền ô văn bản trắng, chữ tối (không phụ thuộc Dark mode của Windows)."""
    from PyQt6.QtGui import QPalette
    pal = QtWidgets.QApplication.instance().palette()
    assert pal.color(QPalette.ColorRole.Base).name() == "#ffffff"
    assert pal.color(QPalette.ColorRole.Text).name() == "#1f2d3d"
    assert "QTextBrowser" in window.MainWindow.styleSheet()
    assert "QComboBox QAbstractItemView" in window.MainWindow.styleSheet()
