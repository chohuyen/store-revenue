"""MainWindowEx – class kế thừa giao diện sinh từ Qt Designer (giống SalesMainWindowEx đã học).

    MainWindow.ui  --pyuic6-->  MainWindow.py (class Ui_MainWindow, KHÔNG sửa tay)
    MainWindowEx(Ui_MainWindow): setupUi() + setupSignalAndSlot() + showWindow()

Chạy chương trình: mở file này (hoặc main.py) trong PyCharm rồi bấm Run,
hoặc gõ:  python ui/MainWindowEx.py

Logic từng màn hình nằm ở ui/pages/*_controller.py; logic nghiệp vụ/ML nằm ở src/.
"""
import sys
from pathlib import Path

# Cho phép chạy trực tiếp file này: thêm thư mục gốc dự án vào đường dẫn import
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication, QMainWindow  # noqa: E402

from src.services.model_service import ModelService  # noqa: E402
from src.services.prediction_service import PredictionService  # noqa: E402
from src.services.whatif_service import WhatIfService  # noqa: E402
from ui.MainWindow import Ui_MainWindow  # noqa: E402
from ui.pages.dashboard_controller import DashboardController  # noqa: E402
from ui.pages.forecast_controller import ForecastController  # noqa: E402
from ui.pages.helpers import Worker, apply_light_theme  # noqa: E402
from ui.pages.performance_controller import PerformanceController  # noqa: E402
from ui.pages.planning_controller import PlanningController  # noqa: E402
from ui.pages.whatif_controller import WhatIfController  # noqa: E402

ABOUT_HTML = """
<h2>XÂY DỰNG VÀ ỨNG DỤNG HỌC MÁY DỰ BÁO DOANH THU CỬA HÀNG BÁN LẺ KẾT HỢP MÔ PHỎNG KỊCH BẢN KINH DOANH
VÀ LẬP KẾ HOẠCH NGUỒN LỰC</h2>
<p><b>Ứng dụng:</b> Store Revenue Forecast &amp; Planning Studio – dự báo doanh thu theo ngày/tuần/tháng, so sánh
cửa hàng, mô phỏng kịch bản khuyến mãi (What-if), quy đổi doanh thu dự báo ra số ca nhân viên và gợi ý hành động.<br>
<b>Môn học:</b> Học máy trong phân tích kinh doanh · <b>GVHD:</b> TS. Trần Duy Thanh · Khoa Hệ thống Thông tin.</p>
<h3>Dữ liệu</h3>
<p>Rossmann Store Sales (Kaggle): 1.115 cửa hàng, dữ liệu ngày 01/01/2013 – 31/07/2015 (train.csv, 1.017.209 dòng),
store.csv (thuộc tính cửa hàng), test.csv (01/08 – 17/09/2015, không có Sales).
Làm sạch: giữ ngày Open ≠ 0 và Sales &gt; 0 (844.338 dòng).<br>
Chia theo thời gian: Train ≤ 07/05/2015 · Validation 08/05 – 18/06/2015 · Test 19/06 – 31/07/2015.</p>
<h3>Mô hình</h3>
<p>Bài toán hồi quy, target log1p(Sales). So sánh Baseline A (TB Store–Thứ–Promo), Baseline B (cùng ngày 52 tuần trước),
Linear Regression, Decision Tree, Random Forest, XGBoost, LightGBM (một mô hình chung cho mọi cửa hàng) và Prophet
(mô hình chuỗi thời gian, mỗi cửa hàng một mô hình – chạy riêng để so sánh). Độ đo: RMSPE, MAE, MSE, RMSE, R².
Mô hình dùng trong ứng dụng được chọn theo RMSPE trên Validation; Test chỉ dùng để báo cáo.
Không dùng Customers (không biết trước khi dự báo).</p>
<h3>Giới hạn</h3>
<ul><li>What-if là mô phỏng theo mô hình, không chứng minh quan hệ nhân quả.</li>
<li>Dữ liệu ngày lễ có doanh thu rất ít; không có Promo vào cuối tuần → các kịch bản này là ngoại suy.</li>
<li>Khoảng dự báo là khoảng thực nghiệm 80% (phân vị tỷ lệ thực tế/dự báo trên Validation).</li>
<li>Kế hoạch nguồn lực dùng giả định "doanh thu 1 ca" do người dùng nhập (dữ liệu không có số liệu nhân sự);
gợi ý là hỗ trợ quyết định theo quy tắc, không phải lời giải tối ưu.</li></ul>
<h3>Nhóm thực hiện</h3>
<ul><li>Nguyễn Gia Bảo – K244030343</li><li>Bùi Thị Khánh Hà – K244060714</li>
<li>Lê Trần Ngọc Huyên – K244060717</li><li>K Thị Lụa – K244060724</li>
<li>Nguyễn Phan Bảo Hân – K244161782</li><li>Lê Vy – K244161824</li></ul>
"""



class MainWindowEx(Ui_MainWindow):
    PAGE_DASHBOARD, PAGE_INPUT, PAGE_RESULTS, PAGE_PERFORMANCE, PAGE_WHATIF, PAGE_PLANNING, PAGE_ABOUT = range(7)

    def __init__(self):
        # Service layer – model chỉ nạp 1 lần, không train lại khi dùng giao diện
        self.model_service = ModelService()
        self.prediction_service = PredictionService(self.model_service)
        self.whatif_service = WhatIfService(self.prediction_service)
        self.dashboard_service = None          # nạp ở luồng nền (đọc train.csv mất vài giây)
        self.last_result = None                # kết quả dự báo gần nhất (DataFrame)
        self.last_warnings = []
        self._workers = []                     # giữ tham chiếu QThread đang chạy

    def setupUi(self, MainWindow):
        apply_light_theme()                    # giữ giao diện sáng kể cả khi Windows bật Dark mode
        super().setupUi(MainWindow)
        self.MainWindow = MainWindow

        # Controller cho từng màn hình
        self.dashboard = DashboardController(self)
        self.forecast = ForecastController(self)
        self.performance = PerformanceController(self)
        self.whatif = WhatIfController(self)
        self.planning = PlanningController(self)
        self.textBrowserAbout.setHtml(ABOUT_HTML)

        self.setupSignalAndSlot()
        self.load_model()
        self.load_dashboard_data()
        self.go_to(self.PAGE_DASHBOARD)

    def showWindow(self):
        self.MainWindow.show()

    # ------------------------------------------------------------------ menu (Signal/Slot)
    def setupSignalAndSlot(self):
        menu = [self.pushButtonMenuDashboard, self.pushButtonMenuInput, self.pushButtonMenuResults,
                self.pushButtonMenuPerformance, self.pushButtonMenuWhatIf, self.pushButtonMenuPlanning,
                self.pushButtonMenuAbout]
        for index, btn in enumerate(menu):
            btn.setCheckable(True)
            btn.clicked.connect(lambda _checked, i=index: self.go_to(i))
        self.menu_buttons = menu

    def go_to(self, index):
        self.stackedWidgetPages.setCurrentIndex(index)
        for i, btn in enumerate(self.menu_buttons):
            btn.setChecked(i == index)

    # ------------------------------------------------------------------ chạy nền
    def run_in_background(self, fn, on_success, on_error, *args):
        worker = Worker(fn, *args)
        worker.succeeded.connect(on_success)
        worker.failed.connect(on_error)
        worker.finished.connect(lambda: self._workers.remove(worker))
        self._workers.append(worker)
        worker.start()
        return worker

    def status(self, text):
        self.statusbar.showMessage(text, 8000)

    # ------------------------------------------------------------------ model
    def load_model(self, force=False):
        try:
            art = self.model_service.load(force=force)
            m = art.metadata
            self.labelModelStatus.setText(
                f"Model: {m['model_name']} v{m['version']}\ntrain {m['trained_at'][:10]}")
            self.forecast.on_model_loaded()
            self.whatif.on_model_loaded()
            self.planning.on_model_loaded()
            self.performance.refresh()
            self.status("Đã nạp model.")
            return True
        except Exception as exc:  # noqa: BLE001
            self.labelModelStatus.setText("Model: CHƯA CÓ\nChạy: python -m src.models.train")
            self.forecast.set_enabled(False)
            self.whatif.set_enabled(False)
            self.planning.set_enabled(False)
            self.status(f"Không nạp được model: {exc}")
            return False

    def load_dashboard_data(self):
        from src.services.dashboard_service import DashboardService
        self.status("Đang đọc dữ liệu lịch sử cho Dashboard…")
        self.run_in_background(DashboardService, self._dashboard_ready, self._dashboard_failed)

    def _dashboard_ready(self, service):
        self.dashboard_service = service
        self.planning.set_history(service.data)
        self.dashboard.refresh()
        self.status("Đã nạp dữ liệu lịch sử.")

    def _dashboard_failed(self, message):
        self.dashboard.show_unavailable(message)
        self.status("Không đọc được train.csv – Dashboard không khả dụng.")

    # ------------------------------------------------------------------ chia sẻ kết quả
    def set_forecast_result(self, result, warnings):
        # Sắp xếp theo ngày rồi cửa hàng để bảng/biểu đồ dễ đọc (không đổi giá trị dự báo)
        self.last_result = result.sort_values(["Date", "Store"], kind="stable").reset_index(drop=True)
        self.last_warnings = warnings
        self.status(f"Đã dự báo {len(result):,} dòng.")
        self.forecast.show_results()
        self.dashboard.show_alerts()
        self.go_to(self.PAGE_RESULTS)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myui = MainWindowEx()
    myui.setupUi(QMainWindow())
    myui.showWindow()
    sys.exit(app.exec())
