"""Controller màn hình What-if Scenario. Chỉ gọi WhatIfService (predict), không train."""
import numpy as np

from ui.pages.forecast_controller import combo_code
from ui.pages.helpers import AMBER, BLUE, TEAL, ChartWidget, fmt_money, fmt_pct, show_error, showDataIntoTableWidget

HOLIDAY_NAMES = {"0": "Không lễ", "a": "Lễ công", "b": "Phục sinh", "c": "Giáng sinh"}


class WhatIfController:
    def __init__(self, window):
        self.w = window
        self.ui = ui = window
        self.chart = ChartWidget(ui.verticalLayoutWIChart, window.MainWindow)
        self.chart.message("Chọn kịch bản rồi bấm 'So sánh 2 kịch bản'")
        ui.pushButtonRunWhatIf.clicked.connect(self.run_compare)
        ui.pushButtonRunGrid.clicked.connect(self.run_grid)

    def set_enabled(self, enabled):
        self.ui.pushButtonRunWhatIf.setEnabled(enabled)
        self.ui.pushButtonRunGrid.setEnabled(enabled)

    def on_model_loaded(self):
        self.set_enabled(True)
        self.ui.comboBoxWIStore.clear()
        self.ui.comboBoxWIStore.addItems([str(s) for s in self.w.model_service.store_ids()])

    def _scenario(self, prefix):
        ui = self.ui
        return {"Promo": int(combo_code(getattr(ui, f"comboBoxWI{prefix}Promo"))),
                "StateHoliday": combo_code(getattr(ui, f"comboBoxWI{prefix}Holiday")),
                "SchoolHoliday": int(combo_code(getattr(ui, f"comboBoxWI{prefix}School")))}

    def _common(self):
        return self.ui.comboBoxWIStore.currentText(), self.ui.dateEditWIDate.date().toString("yyyy-MM-dd")

    def _warnings(self, warnings):
        notes = list(warnings)
        if "StateHoliday" not in self.w.model_service.metadata()["features"]:
            notes.append("Mô hình chỉ biết có/không ngày lễ (cờ IsStateHoliday): đổi a ↔ b ↔ c cho cùng kết quả.")
        self.ui.labelWIWarnings.setText("\n".join("⚠ " + n for n in notes))

    def run_compare(self):
        store, date = self._common()
        base, scen = self._scenario("Base"), self._scenario("Scen")
        if base == scen:
            show_error(self.w.MainWindow, "Kịch bản What-if đang giống hệt kịch bản gốc. Hãy thay đổi ít nhất một yếu tố.",
                       "Chưa có thay đổi")
            return
        try:
            r = self.w.whatif_service.compare(store, date, base, scen)
        except Exception as exc:  # noqa: BLE001
            show_error(self.w.MainWindow, exc, "Không chạy được What-if")
            return
        ui = self.ui
        ui.labelWIBaseValue.setText(fmt_money(r["baseline_sales"]))
        ui.labelWIBaseNote.setText(self._describe(base))
        ui.labelWIScenValue.setText(fmt_money(r["scenario_sales"]))
        ui.labelWIScenNote.setText(self._describe(scen))
        ui.labelWIDiffValue.setText(f"{r['abs_diff']:+,.0f} €")
        ui.labelWIDiffNote.setText(f"Store {store} · {date}")
        ui.labelWIPctValue.setText(fmt_pct(r["pct_diff"]))
        ui.labelWIPctNote.setText("N/A khi dự báo gốc = 0" if r["pct_diff"] is None else "So với kịch bản gốc")
        ui.labelWIDisclaimer.setText("Lưu ý: " + r["disclaimer"])
        table = r["table"]
        showDataIntoTableWidget(ui.tableWidgetWhatIf, table)
        self._warnings(r["warnings"])

        ax = self.chart.new_axes()
        sales = table["PredictedSales"].to_numpy()
        err = np.vstack([sales - table["PredLow"].to_numpy(), table["PredHigh"].to_numpy() - sales])
        ax.bar(["Kịch bản gốc", "Kịch bản What-if"], sales, color=[BLUE, TEAL], yerr=err, capsize=6)
        for i, v in enumerate(sales):
            ax.text(i, v / 2, fmt_money(v), ha="center", color="white", fontweight="bold")
        ax.set_ylabel("Doanh thu dự báo (€)")
        ax.set_title(f"Store {store} · {date} (vạch = khoảng dự báo 80%)")
        self.chart.draw()

    def run_grid(self):
        store, date = self._common()
        school = self._scenario("Base")["SchoolHoliday"]
        try:
            grid, warnings = self.w.whatif_service.scenario_grid(store, date, school)
        except Exception as exc:  # noqa: BLE001
            show_error(self.w.MainWindow, exc, "Không chạy được What-if")
            return
        showDataIntoTableWidget(self.ui.tableWidgetWhatIf, grid)
        self._warnings(warnings)
        ax = self.chart.new_axes()
        holidays = list(HOLIDAY_NAMES)
        x = np.arange(len(holidays))
        for offset, promo, color in [(-0.2, 0, BLUE), (0.2, 1, AMBER)]:
            vals = [grid[(grid.Promo == promo) & (grid.StateHoliday == h)]["PredictedSales"].iloc[0] for h in holidays]
            ax.bar(x + offset, vals, 0.4, color=color, label=f"Promo = {promo}")
        ax.set_xticks(x)
        ax.set_xticklabels([HOLIDAY_NAMES[h] for h in holidays])
        ax.set_ylabel("Doanh thu dự báo (€)")
        ax.set_title(f"Store {store} · {date} · SchoolHoliday = {school}")
        ax.legend(fontsize=8)
        self.chart.draw()

    @staticmethod
    def _describe(s):
        return (f"Promo {s['Promo']} · Lễ {s['StateHoliday']} ({HOLIDAY_NAMES[s['StateHoliday']]}) · "
                f"Nghỉ học {s['SchoolHoliday']}")
