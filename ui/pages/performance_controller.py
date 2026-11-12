"""Controller màn hình Model Performance: bảng so sánh, biểu đồ RMSPE, nạp lại / train lại model."""
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import QMessageBox

from src import config
from ui.pages.helpers import BLUE, TEAL, ChartWidget, show_error, showDataIntoTableWidget


class PerformanceController:
    def __init__(self, window):
        self.w = window
        self.ui = ui = window
        self.chart = ChartWidget(ui.verticalLayoutPerfChart, window.MainWindow)
        self.chart.message("Chưa có bảng so sánh mô hình")
        ui.progressBarTrain.setVisible(False)
        ui.pushButtonReloadModel.clicked.connect(lambda: window.load_model(force=True))
        ui.pushButtonRetrain.clicked.connect(self.retrain)

    def refresh(self):
        ui = self.ui
        try:
            table = self.w.model_service.comparison_table()
            meta = self.w.model_service.metadata()
        except Exception as exc:  # noqa: BLE001
            ui.textBrowserPerfNotes.setText(str(exc))
            return
        shown = table.copy()
        shown["Selected"] = np.where(shown["Selected"], "✔ được chọn", "")
        showDataIntoTableWidget(ui.tableWidgetModelComparison, shown.drop(columns=["Train seconds"], errors="ignore"))

        v, t = meta["metrics"]["val"], meta["metrics"]["test"]
        ba = meta["baseline_a_metrics"]
        ui.labelPerfModelValue.setText(f"{meta['model_name']} · v{meta['version']}")
        ui.labelPerfModelNote.setText(meta["selection_rule"])
        ui.labelPerfValValue.setText(f"RMSPE {v['RMSPE']:.4f}")
        ui.labelPerfValNote.setText(f"MAE {v['MAE']:,.0f} · RMSE {v['RMSE']:,.0f} · R² {v['R2']:.3f} · "
                                    f"Baseline A RMSPE {ba['val']['RMSPE']:.4f}")
        ui.labelPerfTestValue.setText(f"RMSPE {t['RMSPE']:.4f}")
        ui.labelPerfTestNote.setText(f"MAE {t['MAE']:,.0f} · RMSE {t['RMSE']:,.0f} · R² {t['R2']:.3f} · "
                                     f"Baseline A RMSPE {ba['test']['RMSPE']:.4f}")

        ax = self.chart.new_axes()
        labels = table["Model"].str.split(" (", regex=False).str[0].tolist()
        x = np.arange(len(table))
        ax.bar(x - 0.2, table["Val RMSPE"], 0.4, color=BLUE, label="Validation")
        ax.bar(x + 0.2, table["Test RMSPE"], 0.4, color=TEAL, label="Test")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=25, ha="right")
        ax.set_ylabel("RMSPE (thấp hơn = tốt hơn)")
        ax.legend(fontsize=8)
        self.chart.draw()
        ui.textBrowserPerfNotes.setHtml(self._notes(table, meta))

    @staticmethod
    def _notes(table, meta):
        base_a = table[table.Model.str.startswith("Baseline A")].iloc[0]
        t = meta["metrics"]["test"]
        lines = [f"<p><b>Chọn mô hình theo Validation.</b> {meta['model_name']} có Val RMSPE "
                 f"{meta['metrics']['val']['RMSPE']:.4f}, thấp hơn Baseline A ({base_a['Val RMSPE']:.4f}).</p>"]
        if t["RMSPE"] < base_a["Test RMSPE"]:
            cmp = "thấp hơn"
        else:
            cmp = "không thấp hơn"
        lines.append(f"<p><b>Test RMSPE</b> {t['RMSPE']:.4f} {cmp} Baseline A ({base_a['Test RMSPE']:.4f}).")
        seed_file = config.OUT_EVAL / "experiment_seed_stability.csv"
        if seed_file.exists():
            s = pd.read_csv(seed_file)
            s = s[s.Model == meta["model_name"]]
            if len(s):
                lines.append(f" Qua {len(s)} seed, Test RMSPE dao động {s['Test RMSPE'].min():.4f}–"
                             f"{s['Test RMSPE'].max():.4f} (TB {s['Test RMSPE'].mean():.4f}) → chênh lệch với "
                             f"Baseline A nằm trong biên độ dao động, <b>không kết luận được</b> mô hình tốt hơn "
                             f"trên Test RMSPE.")
        lines.append("</p>")
        lines.append(f"<p><b>MAE/RMSE Test:</b> {t['MAE']:,.0f} / {t['RMSE']:,.0f} so với Baseline A "
                     f"{base_a['Test MAE']:,.0f} / {base_a['Test RMSE']:,.0f}. RMSE thấp hơn nghĩa là ít lỗi lớn hơn.</p>")
        lr = table[table.Model == "Linear Regression"]
        if len(lr):
            lines.append(f"<p><b>Linear Regression</b> (one-hot Store, thứ, tháng, năm) đạt R² Test "
                         f"{lr.iloc[0]['Test R2']:.3f}, so với {t['R2']:.3f} của {meta['model_name']}: quan hệ giữa doanh thu "
                         f"và các yếu tố không tuyến tính nên mô hình dạng cây phù hợp hơn.</p>")
        pr = table[table.Model.str.startswith("Prophet")]
        if len(pr):
            pr = pr.iloc[0]
            lines.append(f"<p><b>Prophet</b> (mỗi cửa hàng 1 mô hình chuỗi thời gian) đạt Val RMSPE {pr['Val RMSPE']:.4f} "
                         f"và Test RMSPE {pr['Test RMSPE']:.4f}. Prophet chạy riêng để so sánh, không dùng trong ứng dụng vì "
                         f"cần 1.115 mô hình riêng và thư viện Prophet/Stan; {meta['model_name']} là một mô hình chung, "
                         f"có Test RMSPE thấp hơn và dùng được cho What-if theo thuộc tính cửa hàng.</p>")
        lines.append("<p><b>Vì sao dùng mô hình ML trong ứng dụng:</b> Baseline A chỉ là bảng trung bình theo "
                     "Store–Thứ–Promo, không phản ứng với ngày lễ, nghỉ học hay mùa vụ nên không phục vụ được What-if.</p>")
        iv = meta["prediction_interval"]
        lines.append(f"<p><b>Khoảng dự báo:</b> [{iv['ratio_low']:.3f} × dự báo, {iv['ratio_high']:.3f} × dự báo], "
                     f"bao phủ {iv['test_coverage'] * 100:.1f}% giá trị thật trên Test.</p>")
        if "StateHoliday" not in meta["features"]:
            lines.append("<p><b>Lưu ý:</b> mô hình chỉ dùng cờ có/không ngày lễ, nên "
                         "không phân biệt StateHoliday a/b/c.</p>")
        return "".join(lines)

    def retrain(self):
        answer = QMessageBox.question(
            self.w.MainWindow, "Train lại model",
            "Train lại toàn bộ (2 baseline, Random Forest, XGBoost, LightGBM) mất khoảng 1–10 phút "
            "tuỳ máy và sẽ ghi đè trained/sales_model.joblib. Tiếp tục?")
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.ui.pushButtonRetrain.setEnabled(False)
        self.ui.progressBarTrain.setVisible(True)
        self.ui.labelTrainStatus.setText("Đang train… (giao diện vẫn dùng được)")
        self.w.run_in_background(self.w.model_service.retrain, self._done, self._failed)

    def _done(self, _artifact):
        self.ui.pushButtonRetrain.setEnabled(True)
        self.ui.progressBarTrain.setVisible(False)
        self.ui.labelTrainStatus.setText("Đã train xong và nạp model mới.")
        self.w.load_model(force=True)

    def _failed(self, message):
        self.ui.pushButtonRetrain.setEnabled(True)
        self.ui.progressBarTrain.setVisible(False)
        self.ui.labelTrainStatus.setText("Train thất bại.")
        show_error(self.w.MainWindow, message)
