"""Controller màn hình Kế hoạch nguồn lực: dự báo kỳ tới -> số ca nhân viên -> xếp hạng -> gợi ý.
Mọi tính toán nằm trong src/services/planning_service.py; file này chỉ đọc input, gọi service và hiển thị."""
import re

import numpy as np
from PyQt6.QtCore import QDate
from PyQt6.QtWidgets import QFileDialog

from src import config
from src.services.planning_service import PlanningService
from ui.pages.helpers import GREY, ChartWidget, fmt_money, show_error, showDataIntoTableWidget


class PlanningController:
    def __init__(self, window):
        self.w = window
        self.ui = ui = window
        self.service = None
        self.result = None
        self.chart = ChartWidget(ui.verticalLayoutPlanChart, window.MainWindow)
        self.chart.message("Nhập cửa hàng, kỳ kế hoạch rồi bấm 'Lập kế hoạch'")
        ui.pushButtonRunPlan.clicked.connect(self.run_plan)
        ui.pushButtonExportPlan.clicked.connect(self.export_plan)
        ui.comboBoxPlanGranularity.currentIndexChanged.connect(self.refresh_granularity)

    def on_model_loaded(self):
        self.service = PlanningService(self.w.prediction_service)
        lo, hi = self.service.date_bounds()
        for edit in (self.ui.dateEditPlanFrom, self.ui.dateEditPlanTo):
            edit.setMinimumDate(QDate(lo.year, lo.month, lo.day))
            edit.setMaximumDate(QDate(hi.year, hi.month, hi.day))
        self.ui.pushButtonRunPlan.setEnabled(True)

    def set_enabled(self, enabled):
        self.ui.pushButtonRunPlan.setEnabled(enabled)
        self.ui.pushButtonExportPlan.setEnabled(enabled)

    def set_history(self, daily):
        """Gọi khi Dashboard đã đọc xong dữ liệu lịch sử -> dùng để so sánh cùng kỳ năm trước."""
        if self.service is not None:
            self.service.history = daily

    # ------------------------------------------------------------------ chạy kế hoạch
    def run_plan(self):
        if self.service is None:
            show_error(self.w.MainWindow, "Chưa nạp được model.")
            return
        stores = [int(s) for s in re.findall(r"\d+", self.ui.lineEditPlanStores.text())]
        basis = "PredHigh" if self.ui.comboBoxPlanBasis.currentIndex() == 1 else "PredictedSales"
        args = (stores, self.ui.dateEditPlanFrom.date().toString("yyyy-MM-dd"),
                self.ui.dateEditPlanTo.date().toString("yyyy-MM-dd"), self.ui.spinBoxRevenuePerShift.value(),
                self.ui.spinBoxMinShifts.value(), basis, self.ui.spinBoxShiftBudget.value(),
                self.ui.comboBoxPlanGranularity.currentText())
        self.ui.pushButtonRunPlan.setEnabled(False)
        self.w.status("Đang lập kế hoạch…")
        self.w.run_in_background(self.service.run, self._done, self._failed, *args)

    def _failed(self, message):
        self.ui.pushButtonRunPlan.setEnabled(True)
        show_error(self.w.MainWindow, message.split(": ", 1)[-1], "Không lập được kế hoạch")

    def _done(self, result):
        self.ui.pushButtonRunPlan.setEnabled(True)
        self.result = result
        plan = result["plan"]
        open_rows = plan[plan["Open"] == 1]
        peaks = (result["suggestions"]["Gợi ý"] == "Tăng ca ngày cao điểm").sum()
        self.ui.labelPlanTotalValue.setText(fmt_money(plan["PredictedSales"].sum()))
        self.ui.labelPlanTotalNote.setText(f"Khoảng 80%: {fmt_money(plan['PredLow'].sum())} – {fmt_money(plan['PredHigh'].sum())}"
                                           f" · {plan['Store'].nunique()} cửa hàng · {plan['Date'].nunique()} ngày")
        self.ui.labelPlanShiftsValue.setText(f"{int(plan['ShiftsNeeded'].sum()):,} ca")
        note = f"TB {open_rows['ShiftsNeeded'].mean():.1f} ca / cửa hàng / ngày mở cửa"
        if "ShiftsAllocated" in plan:
            note += f" · phân bổ được {int(plan['ShiftsAllocated'].sum()):,} ca"
        self.ui.labelPlanShiftsNote.setText(note)
        peak_days = open_rows.groupby("Store")["PredictedSales"].transform(lambda s: s > 1.2 * s.mean()).sum()
        self.ui.labelPlanPeakValue.setText(f"{int(peak_days)} ngày")
        self.ui.labelPlanPeakNote.setText(f"Dự báo > 120% TB kỳ của cửa hàng · {peaks} cửa hàng cần tăng ca")
        self.ui.labelPlanSuggestValue.setText(f"{len(result['suggestions'])} gợi ý")
        self.ui.labelPlanSuggestNote.setText("Xem bảng gợi ý bên dưới (có ghi căn cứ)")
        showDataIntoTableWidget(self.ui.tableWidgetPlanRanking, result["ranking"])
        showDataIntoTableWidget(self.ui.tableWidgetSuggestions, result["suggestions"])
        self.refresh_granularity()
        self.w.status(f"Đã lập kế hoạch cho {plan['Store'].nunique()} cửa hàng.")
        if result["warnings"]:
            self.w.status("Lưu ý: " + result["warnings"][0])

    def refresh_granularity(self):
        if self.result is None:
            return
        granularity = self.ui.comboBoxPlanGranularity.currentText()
        summary = self.service.aggregate(self.result["plan"], granularity)
        shown = summary.assign(Period=summary["Period"].dt.strftime("%Y-%m-%d")).rename(columns={"Period": f"{granularity} bắt đầu"})
        showDataIntoTableWidget(self.ui.tableWidgetPlan, shown)
        self._plot(summary, granularity)

    def _plot(self, summary, granularity):
        ax = self.chart.new_axes()
        totals = summary.groupby("Period")["ShiftsNeeded"].sum()
        labels = [f"{d:%d/%m}" for d in totals.index]
        x = np.arange(len(totals))
        ax2 = ax.twinx()
        ax2.bar(x, totals.values, color=GREY, alpha=0.35, label="Tổng số ca cần")
        ax2.set_ylabel("Số ca")
        for store, g in summary.groupby("Store"):
            g = g.set_index("Period").reindex(totals.index)
            ax.plot(x, g["PredictedSales"] / 1e3, marker="o", label=f"Store {store}")
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45 if len(x) > 12 else 0, fontsize=8)
        ax.set_ylabel("Doanh thu dự báo (nghìn €)")
        ax.set_xlabel(f"{granularity} bắt đầu")
        lines, names = ax.get_legend_handles_labels()
        bars, bar_names = ax2.get_legend_handles_labels()
        ax.legend(lines + bars, names + bar_names, fontsize=8, loc="upper left")
        self.chart.draw()

    def export_plan(self):
        if self.result is None:
            show_error(self.w.MainWindow, "Chưa có kế hoạch để xuất.")
            return
        default = config.OUT_PRED / f"ke_hoach_nguon_luc_{QDate.currentDate().toString('yyyyMMdd')}.csv"
        path, _ = QFileDialog.getSaveFileName(self.w.MainWindow, "Xuất kế hoạch", str(default), "CSV (*.csv)")
        if path:
            plan = self.result["plan"].assign(Date=lambda d: d["Date"].dt.strftime("%Y-%m-%d"))
            saved = self.w.prediction_service.export_csv(plan, path)
            self.w.prediction_service.export_csv(self.result["suggestions"], str(saved).replace(".csv", "_goi_y.csv"))
            self.w.status(f"Đã xuất {saved} và file gợi ý đi kèm")
