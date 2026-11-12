"""Controller màn hình Data Input + Forecast Results."""
from pathlib import Path

import pandas as pd
from PyQt6.QtCore import QDate
from PyQt6.QtWidgets import QFileDialog

from src import config
from ui.pages.helpers import BLUE, GREY, TEAL, ChartWidget, fmt_money, show_error, showDataIntoTableWidget


def combo_code(combo):
    """Lấy mã ở đầu chuỗi hiển thị, ví dụ 'a - Lễ công cộng' -> 'a'."""
    return combo.currentText().split(" - ")[0].strip()


class ForecastController:
    def __init__(self, window):
        self.w = window
        self.ui = ui = window
        self.csv_df = None
        self.chart_trend = ChartWidget(ui.verticalLayoutResTrend, window.MainWindow)
        self.chart_weekly = ChartWidget(ui.verticalLayoutResWeekly, window.MainWindow)
        self.chart_trend.message("Chưa có kết quả dự báo")
        self.chart_weekly.message("Chưa có kết quả dự báo")

        ui.pushButtonPredictSingle.clicked.connect(self.predict_single)
        ui.comboBoxStore.currentTextChanged.connect(self.show_store_info)
        ui.pushButtonChooseCsv.clicked.connect(self.choose_csv)
        ui.pushButtonUseSample.clicked.connect(
            lambda: self.load_csv(config.DATA_SAMPLE / "sample_input_with_actual.csv"))
        ui.pushButtonUseKaggleTest.clicked.connect(lambda: self.load_csv(config.TEST_CSV))
        ui.pushButtonPredictCsv.clicked.connect(self.predict_csv)
        ui.pushButtonExportCsv.clicked.connect(self.export_csv)
        ui.pushButtonGoDashboard.clicked.connect(lambda: window.go_to(window.PAGE_DASHBOARD))
        ui.comboBoxResGranularity.currentIndexChanged.connect(self.replot_period)

    # ------------------------------------------------------------------ trạng thái
    def set_enabled(self, enabled):
        for b in (self.ui.pushButtonPredictSingle, self.ui.pushButtonPredictCsv, self.ui.pushButtonExportCsv):
            b.setEnabled(enabled)

    def on_model_loaded(self):
        self.set_enabled(True)
        stores = [str(s) for s in self.w.model_service.store_ids()]
        self.ui.comboBoxStore.blockSignals(True)
        self.ui.comboBoxStore.clear()
        self.ui.comboBoxStore.addItems(stores)
        self.ui.comboBoxStore.blockSignals(False)
        self.show_store_info(self.ui.comboBoxStore.currentText())
        m = self.w.model_service.metadata()["metrics"]["test"]
        self.ui.labelResMetricsValue.setText(f"RMSPE {m['RMSPE']:.4f}")
        self.ui.labelResMetricsNote.setText(f"MAE {m['MAE']:,.0f} € · RMSE {m['RMSE']:,.0f} € · {m['n_rows']:,} dòng Test")

    def show_store_info(self, text):
        try:
            info = self.w.model_service.store_info(int(text))
        except (ValueError, TypeError):
            self.ui.labelStoreInfo.setText("Thông tin cửa hàng: mã Store không hợp lệ")
            return
        self.ui.labelStoreInfo.setText(
            f"StoreType {info['StoreType']} · Assortment {info['Assortment']} · "
            f"Đối thủ cách {info['CompetitionDistance']:,.0f} m · Promo2 = {info['Promo2']} "
            f"({info['PromoInterval']}) – các đặc trưng này lấy tự động từ store.csv")

    # ------------------------------------------------------------------ một bản ghi
    def predict_single(self):
        try:
            result, warnings = self.w.prediction_service.predict_frame(pd.DataFrame([{
                "Store": self.ui.comboBoxStore.currentText(),
                "Date": self.ui.dateEditDate.date().toString("yyyy-MM-dd"),
                "Promo": int(combo_code(self.ui.comboBoxPromo)),
                "StateHoliday": combo_code(self.ui.comboBoxStateHoliday),
                "SchoolHoliday": int(combo_code(self.ui.comboBoxSchoolHoliday)),
            }]))
        except Exception as exc:  # noqa: BLE001
            show_error(self.w.MainWindow, exc, "Dữ liệu đầu vào không hợp lệ")
            return
        self.w.set_forecast_result(result, warnings)

    # ------------------------------------------------------------------ CSV
    def choose_csv(self):
        path, _ = QFileDialog.getOpenFileName(self.w.MainWindow, "Chọn file CSV", str(config.DATA_SAMPLE), "CSV (*.csv)")
        if path:
            self.load_csv(Path(path))

    def load_csv(self, path):
        try:
            df = pd.read_csv(path, dtype={"StateHoliday": str})
        except Exception as exc:  # noqa: BLE001
            show_error(self.w.MainWindow, f"Không đọc được file: {exc}")
            return
        self.csv_df = None
        self.ui.labelCsvPath.setText(str(path))
        self.ui.labelPrevRowsValue.setText(f"{len(df):,}")
        self.ui.labelPrevColsValue.setText(str(len(df.columns)))
        self.ui.labelPrevColsNote.setText(", ".join(map(str, df.columns))[:120])
        showDataIntoTableWidget(self.ui.tableWidgetPreview, df, max_rows=200)
        try:
            checked, warnings = self.w.prediction_service.validate_input(df)
        except Exception as exc:  # noqa: BLE001
            self.ui.labelValidationStatus.setText(f"❌ Dữ liệu chưa hợp lệ: {exc}")
            self.ui.labelPrevDatesValue.setText("—")
            self.ui.labelPrevStoresValue.setText("—")
            return
        self.csv_df = df
        self.ui.labelPrevDatesValue.setText(
            f"{checked.Date.min():%Y-%m-%d} → {checked.Date.max():%Y-%m-%d}")
        self.ui.labelPrevStoresValue.setText(f"{checked.Store.nunique():,}")
        extra = " Có cột Sales → sẽ tính sai số." if "Sales" in df.columns else ""
        self.ui.labelValidationStatus.setText("✅ Dữ liệu hợp lệ." + extra +
                                              ("\n⚠ " + "\n⚠ ".join(warnings) if warnings else ""))
        self.ui.comboBoxCsvStoreFilter.clear()
        self.ui.comboBoxCsvStoreFilter.addItems(["Tất cả"] + [str(s) for s in sorted(checked.Store.unique())])

    def predict_csv(self):
        if self.csv_df is None:
            show_error(self.w.MainWindow, "Hãy chọn một file CSV hợp lệ trước khi chạy dự báo.")
            return
        df = self.csv_df
        choice = self.ui.comboBoxCsvStoreFilter.currentText()
        if choice != "Tất cả":
            df = df[pd.to_numeric(df["Store"], errors="coerce") == int(choice)]
        self.ui.pushButtonPredictCsv.setEnabled(False)
        self.w.status(f"Đang dự báo {len(df):,} dòng…")
        self.w.run_in_background(self.w.prediction_service.predict_frame, self._csv_done, self._csv_failed, df)

    def _csv_done(self, output):
        self.ui.pushButtonPredictCsv.setEnabled(True)
        result, warnings = output
        self.w.set_forecast_result(result, warnings)

    def _csv_failed(self, message):
        self.ui.pushButtonPredictCsv.setEnabled(True)
        show_error(self.w.MainWindow, message, "Dự báo thất bại")

    # ------------------------------------------------------------------ Forecast Results
    def show_results(self):
        r = self.w.last_result
        ui = self.ui
        dates = pd.to_datetime(r["Date"])
        stores = r["Store"].unique()
        ui.labelResSubtitle.setText(
            f"{len(r):,} dòng · {len(stores):,} cửa hàng · {dates.min():%Y-%m-%d} → {dates.max():%Y-%m-%d} · "
            f"model {self.w.model_service.metadata()['model_name']}")
        ui.labelResTotalValue.setText(fmt_money(r["PredictedSales"].sum()))
        ui.labelResTotalNote.setText(f"Khoảng 80%: {fmt_money(r['PredLow'].sum())} – {fmt_money(r['PredHigh'].sum())}")
        ui.labelResAvgValue.setText(fmt_money(r["PredictedSales"].mean()))
        if "ActualSales" in r.columns:
            pct = r["PctError"].dropna()
            ui.labelResAvgNote.setText(f"Thực tế TB {fmt_money(r['ActualSales'].mean())} · "
                                       f"MAE {r['AbsError'].mean():,.0f} € · MAPE {pct.mean() * 100:.1f}% ({len(pct)} dòng)")
        else:
            ui.labelResAvgNote.setText("Không có Sales thật trong dữ liệu đầu vào")
        ui.labelResAlertValue.setText(f"{int(r['LowForecastAlert'].sum()):,} dòng")
        ui.labelResAlertNote.setText("Kỳ vọng lịch sử > cận trên khoảng dự báo")
        ui.labelResWarnings.setText("\n".join("⚠ " + w for w in self.w.last_warnings))
        showDataIntoTableWidget(ui.tableWidgetResults, r, max_rows=2000)
        if len(r) > 2000:
            self.w.status("Bảng chỉ hiển thị 2.000 dòng đầu; xuất CSV để xem đủ.")
        self._plot_trend(r, dates, stores)
        self._plot_weekly(r, dates)

    def _plot_trend(self, r, dates, stores):
        ax = self.chart_trend.new_axes()
        data = r.assign(Date=dates)
        if len(stores) == 1 and self.w.dashboard_service is not None:
            hist = self.w.dashboard_service.store_history(stores[0], dates.min())   # None nếu không có train.csv
            if hist is not None and not hist.empty:
                ax.plot(hist.Date, hist.Sales, color=BLUE, marker="o", ms=2, label="Lịch sử (6 tuần trước)")
            title = f"Store {stores[0]}: lịch sử và dự báo"
        else:
            title = "Tổng doanh thu dự báo theo ngày (mọi cửa hàng đã chọn)"
        daily = data.groupby("Date")[[c for c in ["PredictedSales", "PredLow", "PredHigh", "ActualSales"]
                                      if c in data.columns]].sum()
        ax.plot(daily.index, daily.PredictedSales, color=TEAL, marker="o", ms=3, label="Dự báo")
        ax.fill_between(daily.index, daily.PredLow, daily.PredHigh, color=TEAL, alpha=0.15, label="Khoảng 80%")
        if "ActualSales" in daily.columns:
            ax.plot(daily.index, daily.ActualSales, color=GREY, ls="--", marker=".", label="Thực tế")
        ax.axvline(dates.min(), color=GREY, ls=":", lw=1)
        ax.set_title(title)
        ax.set_ylabel("€")
        ax.legend(fontsize=8)
        self.chart_trend.figure.autofmt_xdate()
        self.chart_trend.draw()

    def replot_period(self):
        r = self.w.last_result
        if r is not None:
            self._plot_weekly(r, pd.to_datetime(r["Date"]))

    def _plot_weekly(self, r, dates):
        """Tổng doanh thu dự báo theo Ngày / Tuần / Tháng (chọn ở comboBoxResGranularity)."""
        choice = self.ui.comboBoxResGranularity.currentText()
        freq = {"Ngày": "D", "Tuần": "W", "Tháng": "M"}[choice]
        period = dates.dt.to_period(freq).dt.start_time
        totals = r.assign(Period=period.to_numpy()).groupby("Period")["PredictedSales"].sum()
        ax = self.chart_weekly.new_axes()
        fmt = "%m/%Y" if freq == "M" else "%d/%m"
        ax.bar([f"{d:{fmt}}" for d in totals.index], totals.values / 1e3, color=TEAL)
        if len(totals) > 12:
            ax.tick_params(axis="x", labelrotation=60, labelsize=7)
        ax.set_ylabel("Nghìn €")
        ax.set_xlabel({"D": "Ngày", "W": "Tuần bắt đầu (Thứ Hai)", "M": "Tháng"}[freq])
        self.chart_weekly.draw()

    def export_csv(self):
        if self.w.last_result is None:
            show_error(self.w.MainWindow, "Chưa có kết quả để xuất.")
            return
        default = config.OUT_PRED / f"forecast_{QDate.currentDate().toString('yyyyMMdd')}.csv"
        path, _ = QFileDialog.getSaveFileName(self.w.MainWindow, "Xuất CSV", str(default), "CSV (*.csv)")
        if path:
            saved = self.w.prediction_service.export_csv(self.w.last_result, path)
            self.w.status(f"Đã xuất {saved}")
