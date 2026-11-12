"""Controller màn hình Business Dashboard: KPI, xu hướng, top cửa hàng, so sánh, cảnh báo."""
import re

from ui.pages.helpers import BLUE, TEAL, ChartWidget, fmt_money, show_error, showDataIntoTableWidget


class DashboardController:
    def __init__(self, window):
        self.w = window
        ui = window
        self.ui = ui
        self.chart_trend = ChartWidget(ui.verticalLayoutDashTrend, window.MainWindow)
        self.chart_top = ChartWidget(ui.verticalLayoutDashTop, window.MainWindow)
        self.chart_compare = ChartWidget(ui.verticalLayoutDashCompare, window.MainWindow)
        for c in (self.chart_trend, self.chart_top, self.chart_compare):
            c.message("Đang nạp dữ liệu…")
        ui.pushButtonDashApply.clicked.connect(self.refresh)
        ui.spinBoxTopN.valueChanged.connect(self.refresh_top)
        ui.pushButtonCompareStores.clicked.connect(self.refresh_compare)
        ui.lineEditCompareStores.returnPressed.connect(self.refresh_compare)

    # ------------------------------------------------------------------ bộ lọc
    def _filters(self):
        start = self.ui.dateEditDashFrom.date().toString("yyyy-MM-dd")
        end = self.ui.dateEditDashTo.date().toString("yyyy-MM-dd")
        if start > end:
            raise ValueError("'Từ ngày' phải nhỏ hơn hoặc bằng 'Đến ngày'")
        return start, end, self.ui.comboBoxDashStoreType.currentText()

    def show_unavailable(self, message):
        for c in (self.chart_trend, self.chart_top, self.chart_compare):
            c.message("Không đọc được dữ liệu lịch sử (datasets/train.zip)")
        self.ui.labelKpiTotalNote.setText(message)

    # ------------------------------------------------------------------ cập nhật
    def refresh(self):
        service = self.w.dashboard_service
        if service is None:
            return
        try:
            start, end, store_type = self._filters()
            k = service.kpis(start, end, store_type)
        except ValueError as exc:
            show_error(self.w.MainWindow, exc)
            return
        self.ui.labelKpiTotalValue.setText(f"{k['total_sales'] / 1e6:,.2f} triệu €")
        self.ui.labelKpiTotalNote.setText(f"{k['start']} → {k['end']} · {k['rows']:,} dòng cửa hàng-ngày")
        self.ui.labelKpiAvgValue.setText(fmt_money(k["avg_sales_per_store_day"]))
        self.ui.labelKpiAvgNote.setText("Trung bình trên các ngày mở cửa có doanh thu")
        self.ui.labelKpiStoresValue.setText(f"{k['stores']:,} · {k['months']:,} tháng")
        self.ui.labelKpiStoresNote.setText(f"StoreType: {store_type} · bộ lọc ngày tính theo tháng")
        self.ui.labelKpiPromoValue.setText(f"{fmt_money(k['avg_sales_promo'])} / {fmt_money(k['avg_sales_no_promo'])}")
        self.ui.labelKpiPromoNote.setText("Chênh lệch quan sát, không phải tác động nhân quả")

        trend = service.trend(start, end, store_type)
        ax = self.chart_trend.new_axes()
        ax.plot(trend["Date"], trend["Sales"] / 1e6, color=BLUE, marker="o", ms=3)
        ax.set_ylabel("Triệu €")
        ax.set_title("Tổng doanh thu theo tháng (lịch sử)")
        self.chart_trend.figure.autofmt_xdate()
        self.chart_trend.draw()
        self.refresh_top()
        self.refresh_compare()
        self.show_alerts()

    def refresh_top(self):
        service = self.w.dashboard_service
        if service is None:
            return
        try:
            start, end, store_type = self._filters()
            top = service.top_stores(self.ui.spinBoxTopN.value(), start, end, store_type)
        except ValueError as exc:
            show_error(self.w.MainWindow, exc)
            return
        showDataIntoTableWidget(self.ui.tableWidgetTopStores, top)
        ax = self.chart_top.new_axes()
        ordered = top.iloc[::-1]
        ax.barh(ordered["Store"].astype(str), ordered["TotalSales"] / 1e6, color=TEAL)
        ax.set_xlabel("Tổng doanh thu (triệu €)")
        ax.set_ylabel("Store")
        self.chart_top.draw()

    def refresh_compare(self):
        service = self.w.dashboard_service
        if service is None:
            return
        text = self.ui.lineEditCompareStores.text()
        stores = [int(s) for s in re.findall(r"\d+", text)]
        if not stores or len(stores) > 8:
            show_error(self.w.MainWindow, "Nhập từ 1 đến 8 mã cửa hàng, cách nhau bằng dấu phẩy (ví dụ: 1, 262, 817)")
            return
        try:
            start, end, _ = self._filters()
            table = service.compare_stores(stores, start, end)
        except ValueError as exc:
            show_error(self.w.MainWindow, exc)
            return
        ax = self.chart_compare.new_axes()
        for store in table.columns:
            ax.plot(table.index, table[store] / 1e3, marker="o", ms=2, label=f"Store {store}")
        ax.set_ylabel("Nghìn € / tháng")
        ax.legend(fontsize=8)
        self.chart_compare.figure.autofmt_xdate()
        self.chart_compare.draw()
        missing = sorted(set(stores) - set(table.columns))
        if missing:
            self.w.status(f"Không có dữ liệu cho cửa hàng: {missing}")

    def show_alerts(self):
        result = self.w.last_result
        if result is None:
            return
        alerts = result[result["LowForecastAlert"] == 1]
        self.ui.labelAlertsSummary.setText(
            f"{len(alerts):,}/{len(result):,} dòng dự báo có mức kỳ vọng lịch sử (TB cùng Store–Thứ–Promo) "
            f"cao hơn cận trên khoảng dự báo 80%. Đây là tín hiệu cần xem lại, không phải kết luận.")
        cols = ["Store", "Date", "Promo", "StateHoliday", "PredictedSales", "PredHigh", "ExpectedSales"]
        showDataIntoTableWidget(self.ui.tableWidgetAlerts, alerts[cols], max_rows=500)
