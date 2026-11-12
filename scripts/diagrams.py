"""Vẽ sơ đồ quy trình, kiến trúc và lưu đồ (flowchart) từng màn hình cho báo cáo.

Ký hiệu lưu đồ: hình elip = bắt đầu/kết thúc · hình chữ nhật = xử lý · hình thoi = điều kiện ·
hình bình hành = nhập/xuất dữ liệu.
Kết quả: results/figures/diagrams/*.png
Cần cài Graphviz (https://graphviz.org/download/) để chạy lại:  python scripts/diagrams.py
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

OUT = config.OUT_FIG / "diagrams"
FONT = "DejaVu Sans"
HEAD = f'''graph [fontname="{FONT}", fontsize=13, labelloc=t, nodesep=0.35, ranksep=0.45, pad=0.3, dpi=160];
node [fontname="{FONT}", fontsize=11, shape=box, style="rounded,filled", fillcolor="#eef3fa", color="#1f6feb"];
edge [fontname="{FONT}", fontsize=10, color="#555555"];
'''


def start(name, text="Bắt đầu"):
    return f'{name} [label="{text}", shape=ellipse, fillcolor="#d7f2ee", color="#12b5a6"];\n'


def end(name, text="Kết thúc"):
    return f'{name} [label="{text}", shape=ellipse, fillcolor="#d7f2ee", color="#12b5a6"];\n'


def proc(name, text):
    return f'{name} [label="{text}"];\n'


def cond(name, text):
    return f'{name} [label="{text}", shape=diamond, style=filled, fillcolor="#fff4dc", color="#f59e0b"];\n'


def io(name, text):
    return f'{name} [label="{text}", shape=parallelogram, style=filled, fillcolor="#f2f2f2", color="#888888"];\n'


def flow(title, body, rankdir="TB"):
    return f'digraph G {{\n{HEAD}rankdir={rankdir};\nlabel="{title}";\n{body}}}\n'


DIAGRAMS = {}

# ------------------------------------------------------------------ 1. quy trình 3 phiên
DIAGRAMS["01_quy_trinh_3_phien"] = flow("Quy trình xây dựng và tích hợp mô hình học máy (3 phiên)", '''
subgraph cluster_p { label="PREPARING DATA"; style="rounded,dashed"; color="#1f6feb";
  d1 [label="Dữ liệu Rossmann\\ntrain.csv · store.csv · test.csv", shape=cylinder, fillcolor="#f2f2f2", color="#888888"];
  d2 [label="Kiểm tra schema, giá trị thiếu,\\ntrùng lặp, kiểu dữ liệu\\n(src/data/loader.py)"];
  d3 [label="Làm sạch: Open ≠ 0 và Sales > 0\\n→ 844.338 dòng"];
  d4 [label="Ghép store.csv + tạo 24 đặc trưng\\n(src/features/pipeline.py)"];
  d5 [label="Chia theo thời gian\\nTrain / Validation / Test"];
  d1 -> d2 -> d3 -> d4 -> d5; }
subgraph cluster_b { label="BUILD MODEL"; style="rounded,dashed"; color="#f59e0b";
  b1 [label="Chọn giải thuật: 2 baseline, Linear Regression,\\nDecision Tree, Random Forest, XGBoost,\\nLightGBM (+ Prophet so sánh)"];
  b2 [label="Train với target log1p(Sales)\\n(src/models/train.py)"];
  b3 [label="Đánh giá trên Validation\\nRMSPE · MAE · MSE · RMSE · R²"];
  b4 [label="Mô hình đạt yêu cầu?", shape=diamond, style=filled, fillcolor="#fff4dc", color="#f59e0b"];
  b5 [label="Đánh giá Test 1 lần, SHAP,\\nkhoảng dự báo 80%\\n(src/models/evaluate.py)"];
  b6 [label="Lưu model + metadata\\n(trained/sales_model.joblib)", shape=cylinder, fillcolor="#f2f2f2", color="#888888"];
  b1 -> b2 -> b3 -> b4; b4 -> b5 [label="Có"]; b5 -> b6; }
subgraph cluster_u { label="USE MODEL"; style="rounded,dashed"; color="#12b5a6";
  u1 [label="Nạp model 1 lần khi mở ứng dụng\\n(src/models/model_loader.py)"];
  u2 [label="Gọi predict qua các service:\\nDự báo · What-if · Kế hoạch nguồn lực"];
  u3 [label="Hiển thị trên giao diện PyQt6\\n→ nhà quản lý ra quyết định"];
  u1 -> u2 -> u3; }
compound=true; newrank=true;
{ rank=same; d1; b1; u1; }
{ rank=same; d3; b3; u3; }
d3 -> b3 [ltail=cluster_p, lhead=cluster_b, constraint=false, penwidth=2];
b5 -> u3 [ltail=cluster_b, lhead=cluster_u, constraint=false, penwidth=2];
b4 -> d4 [label="Chưa: quay lại\\ndữ liệu / đặc trưng", style=dashed, constraint=false];
u4 [label="Theo thời gian: dữ liệu mới\\n→ đánh giá lại, train lại\\n(quay về PREPARING DATA)", shape=note, style=dashed, fillcolor="#ffffff"];
u3 -> u4 [style=dashed];
''', rankdir="TB")

# ------------------------------------------------------------------ 2. kiến trúc
DIAGRAMS["02_kien_truc_he_thong"] = flow("Kiến trúc phần mềm Store Revenue Forecast & Planning Studio", '''
node [width=3.2];
ui [label="UI LAYER\\nui/MainWindow.ui → ui/MainWindow.py (pyuic6)\\nclass Ui_MainWindow: 7 màn hình (QStackedWidget)", fillcolor="#e8f0fb"];
ctl [label="CONTROLLER LAYER\\nui/MainWindowEx.py: class MainWindowEx(Ui_MainWindow)\\nui/pages/*_controller.py: Signal/Slot, bảng, biểu đồ, QThread", fillcolor="#e8f0fb"];
svc [label="SERVICE LAYER (src/services)\\nPredictionService · WhatIfService · PlanningService\\nDashboardService · ModelService", fillcolor="#fff4dc", color="#f59e0b"];
ml [label="ML LAYER (src/models)\\ntrain · evaluate · predict · model_loader", fillcolor="#d7f2ee", color="#12b5a6"];
data [label="DATA LAYER\\nsrc/data/loader.py · src/features/pipeline.py", fillcolor="#d7f2ee", color="#12b5a6"];
files [label="datasets/ (CSV) · trained/ (model) · results/ (kết quả)", shape=cylinder, fillcolor="#f2f2f2", color="#888888"];
ui -> ctl [label="setupUi"]; ctl -> svc [label="gọi hàm"]; svc -> ml [label="predict"]; svc -> data; ml -> data;
ml -> files; data -> files;
''')

# ------------------------------------------------------------------ 3. quy trình Qt Designer
DIAGRAMS["03_quy_trinh_giao_dien"] = flow("Quy trình xây dựng giao diện PyQt6 bằng Qt Designer", '''
a [label="Thiết kế giao diện\\nQt Designer", shape=component];
b [label="MainWindow.ui\\n(file XML)", shape=note, fillcolor="#f2f2f2", color="#888888"];
c [label="pyuic6 -x ui/MainWindow.ui\\n-o ui/MainWindow.py"];
d [label="MainWindow.py\\nclass Ui_MainWindow\\n(không sửa tay)", shape=note, fillcolor="#f2f2f2", color="#888888"];
e [label="MainWindowEx.py\\nclass MainWindowEx(Ui_MainWindow)\\nsetupUi · setupSignalAndSlot · showWindow"];
f [label="ui/pages/*_controller.py\\nbutton.clicked.connect(hàm xử lý)"];
g [label="main.py\\nQApplication → MainWindowEx()\\n→ setupUi(QMainWindow()) → showWindow()"];
a -> b -> c -> d -> e -> f; e -> g [style=dashed, label="chạy"];
''', rankdir="LR")

# ------------------------------------------------------------------ 4. train
DIAGRAMS["04_luu_do_train_model"] = flow("Lưu đồ huấn luyện và lưu mô hình (python -m src.models.train)",
    start("s") + io("i1", "Đọc train.csv (tự giải nén train.zip), store.csv") + proc("p1", "Làm sạch: Open ≠ 0 và Sales > 0") +
    proc("p2", "Chia theo thời gian:\\nTrain ≤ 07/05/2015 · Val 08/05–18/06 · Test 19/06–31/07") +
    proc("p3", "RossmannPipeline.fit(Train)\\n(encoder, median chỉ học từ Train)") +
    proc("p4", "Tính Baseline A, Baseline B") +
    proc("p5", "Lần lượt train 5 mô hình hồi quy\\n(target = log1p(Sales))") + cond("c1", "Còn mô hình\\nchưa train?") +
    proc("p6", "Đánh giá Validation:\\nRMSPE, MAE, MSE, RMSE, R²") +
    proc("p7", "Chọn mô hình có RMSPE Validation thấp nhất") +
    proc("p8", "Đánh giá Test (chỉ để báo cáo)\\n+ thêm kết quả Prophet nếu có") +
    proc("p9", "Tính khoảng dự báo 80% từ Validation") +
    io("o1", "Lưu trained/sales_model.joblib\\n+ model_metadata.json + model_comparison.csv") +
    proc("p10", "evaluate: sai số theo nhóm, SHAP, vẽ hình") + end("e") + '''
s -> i1 -> p1 -> p2 -> p3 -> p4 -> p5 -> c1;
c1 -> p5 [label="Có"]; c1 -> p6 [label="Không"];
p6 -> p7 -> p8 -> p9 -> o1 -> p10 -> e;
''')

# ------------------------------------------------------------------ 5. khởi động
DIAGRAMS["05_luu_do_khoi_dong"] = flow("Lưu đồ khởi động ứng dụng (main.py / MainWindowEx.py)",
    start("s") + proc("p1", "Tạo QApplication, MainWindowEx()\\nsetupUi(QMainWindow())") +
    proc("p2", "Tạo controller 7 màn hình\\nnối Signal/Slot menu") + io("i1", "Nạp trained/sales_model.joblib") +
    cond("c1", "Nạp model\\nthành công?") + proc("p3", "Bật chức năng dự báo, What-if, kế hoạch\\nhiện tên model ở menu") +
    proc("p4", "Tắt nút dự báo\\nhiện 'Model: CHƯA CÓ'") +
    proc("p5", "QThread: đọc train.csv cho Dashboard\\n(giao diện không bị treo)") + cond("c2", "Đọc dữ liệu\\nthành công?") +
    proc("p6", "Vẽ Dashboard") + proc("p7", "Báo không có dữ liệu lịch sử") +
    proc("p8", "showWindow(): hiện Business Dashboard") + end("e", "Chờ người dùng thao tác") + '''
s -> p1 -> p2 -> i1 -> c1; c1 -> p3 [label="Có"]; c1 -> p4 [label="Không"];
p3 -> p5; p4 -> p5; p5 -> c2; c2 -> p6 [label="Có"]; c2 -> p7 [label="Không"]; p6 -> p8; p7 -> p8; p8 -> e;
''')

# ------------------------------------------------------------------ 6. dashboard
DIAGRAMS["06_luu_do_dashboard"] = flow("Lưu đồ màn hình Business Dashboard",
    start("s", "Chọn menu Business Dashboard") + io("i1", "Chọn Từ ngày, Đến ngày, StoreType") +
    proc("p0", "Bấm 'Áp dụng'") + cond("c1", "Từ ngày ≤ Đến ngày\\nvà có dữ liệu?") +
    proc("p1", "Thông báo lỗi (QMessageBox)") +
    proc("p2", "Tính 4 KPI: tổng doanh thu, TB/cửa hàng/ngày,\\nsố cửa hàng · tháng, TB có/không KM") +
    proc("p3", "Vẽ xu hướng doanh thu theo tháng") + io("i2", "Chọn số cửa hàng top N") +
    proc("p4", "Biểu đồ + bảng top cửa hàng") + io("i3", "Nhập mã cửa hàng (1–8 mã), bấm 'So sánh'") +
    proc("p5", "Biểu đồ so sánh doanh thu theo tháng") +
    cond("c2", "Đã có lần\\ndự báo nào?") + proc("p6", "Bảng cảnh báo dự báo thấp\\nhơn kỳ vọng") + end("e") + '''
s -> i1 -> p0 -> c1; c1 -> p1 [label="Không"]; p1 -> i1 [style=dashed];
c1 -> p2 [label="Có"]; p2 -> p3 -> i2 -> p4 -> i3 -> p5 -> c2;
c2 -> p6 [label="Có"]; c2 -> e [label="Không"]; p6 -> e;
''')

# ------------------------------------------------------------------ 7. nhập liệu + dự báo
DIAGRAMS["07_luu_do_du_bao"] = flow("Lưu đồ màn hình Data Input → Forecast Results",
    start("s", "Chọn menu Data Input") + cond("c0", "Dự báo 1 bản ghi\\nhay file CSV?") +
    io("i1", "Chọn Store, Date, Promo,\\nStateHoliday, SchoolHoliday") +
    io("i2", "Chọn file CSV / file mẫu / test.csv") +
    proc("p1", "Đọc file, hiện bảng xem trước\\n(số dòng, số cột, khoảng ngày, số cửa hàng)") +
    cond("c1", "Đủ cột, đúng giá trị?\\n(Store tồn tại, Promo 0/1,\\nStateHoliday 0/a/b/c, Date hợp lệ)") +
    proc("p2", "Hiện lỗi cụ thể\\n(không làm sập ứng dụng)") + io("i3", "Lọc cửa hàng (tuỳ chọn),\\nbấm 'Chạy dự báo CSV'") +
    proc("p3", "PredictionService.predict_frame\\n(QThread; model đã lưu, không train lại)") +
    proc("p4", "Ghép store.csv → 24 đặc trưng → XGBoost\\n→ expm1, Open = 0 ⇒ 0 €") +
    proc("p5", "Khoảng 80%, mức kỳ vọng,\\ncờ cảnh báo, sai số nếu có Sales") +
    io("o1", "Forecast Results: KPI, biểu đồ xu hướng,\\ntổng theo Ngày/Tuần/Tháng, bảng chi tiết") +
    cond("c2", "Bấm 'Xuất CSV'?") + io("o2", "Lưu results/predictions/*.csv") + end("e") + '''
s -> c0; c0 -> i1 [label="1 bản ghi"]; c0 -> i2 [label="CSV"];
i2 -> p1 -> c1; c1 -> p2 [label="Không"]; p2 -> i2 [style=dashed]; c1 -> i3 [label="Có"]; i3 -> p3;
i1 -> p3; p3 -> p4 -> p5 -> o1 -> c2; c2 -> o2 [label="Có"]; c2 -> e [label="Không"]; o2 -> e;
''')

# ------------------------------------------------------------------ 8. model performance
DIAGRAMS["08_luu_do_model_performance"] = flow("Lưu đồ màn hình Model Performance",
    start("s", "Chọn menu Model Performance") + io("i1", "Đọc model_comparison.csv,\\nmodel_metadata.json") +
    proc("p1", "Hiện bảng 8 phương pháp, KPI Validation/Test,\\nbiểu đồ RMSPE, nhận xét") +
    cond("c1", "Người dùng bấm?") + proc("p2", "Nạp lại model từ file") +
    cond("c2", "Xác nhận train lại?\\n(mất 1–2 phút)") +
    proc("p3", "QThread: chạy train.main()\\nthanh tiến trình, giao diện vẫn dùng được") +
    cond("c3", "Train thành công?") + proc("p4", "Nạp model mới, cập nhật mọi màn hình") +
    proc("p5", "Thông báo lỗi") + end("e") + '''
s -> i1 -> p1 -> c1; c1 -> p2 [label="Nạp lại model"]; c1 -> c2 [label="Train lại model"]; c1 -> e [label="Không"];
p2 -> p1; c2 -> p3 [label="Có"]; c2 -> e [label="Không"]; p3 -> c3; c3 -> p4 [label="Có"]; c3 -> p5 [label="Không"];
p4 -> p1; p5 -> e;
''')

# ------------------------------------------------------------------ 9. what-if
DIAGRAMS["09_luu_do_whatif"] = flow("Lưu đồ màn hình What-if Scenario",
    start("s", "Chọn menu What-if Scenario") + io("i1", "Chọn Store, Ngày") +
    cond("c0", "So sánh 2 kịch bản\\nhay xem mọi tổ hợp?") +
    io("i2", "Kịch bản gốc và kịch bản What-if:\\nPromo, StateHoliday, SchoolHoliday") +
    cond("c1", "Hai kịch bản\\ngiống hệt nhau?") + proc("p1", "Nhắc thay đổi ít nhất 1 yếu tố") +
    proc("p2", "Tạo 2 dòng: giữ nguyên Store, Date,\\nthuộc tính cửa hàng; chỉ đổi yếu tố đã chọn") +
    proc("p3", "predict bằng model đã lưu\\n(KHÔNG gọi fit/train)") +
    cond("c2", "Doanh thu gốc = 0?") + proc("p4", "% chênh lệch = N/A") + proc("p5", "Tính chênh lệch € và %") +
    proc("p6", "Tạo 8 tổ hợp Promo {0,1} × StateHoliday {0,a,b,c}\\n→ predict") +
    io("o1", "KPI, biểu đồ có khoảng 80%, bảng kịch bản,\\ncảnh báo ngoại suy + 'không phải nhân quả'") + end("e") + '''
s -> i1 -> c0; c0 -> i2 [label="2 kịch bản"]; c0 -> p6 [label="Mọi tổ hợp"];
i2 -> c1; c1 -> p1 [label="Có"]; p1 -> i2 [style=dashed]; c1 -> p2 [label="Không"]; p2 -> p3 -> c2;
c2 -> p4 [label="Có"]; c2 -> p5 [label="Không"]; p4 -> o1; p5 -> o1; p6 -> o1; o1 -> e;
''')

# ------------------------------------------------------------------ 10. kế hoạch nguồn lực
DIAGRAMS["10_luu_do_ke_hoach_nguon_luc"] = flow("Lưu đồ màn hình Kế hoạch nguồn lực",
    start("s", "Chọn menu Kế hoạch nguồn lực") +
    io("i1", "Nhập: cửa hàng, Từ–Đến ngày, xem theo Ngày/Tuần/Tháng,\\nmức doanh thu (trung tâm / cận trên 80%),\\ndoanh thu 1 ca, số ca tối thiểu, tổng ca có sẵn") +
    proc("p0", "Bấm 'Lập kế hoạch' (QThread)") +
    cond("c1", "Có cửa hàng, ≤ 50 cửa hàng,\\nngày hợp lệ, cửa hàng có\\nlịch trong test.csv?") +
    proc("p1", "Hiện lỗi cụ thể") +
    proc("p2", "Lấy lịch kỳ tới từ test.csv\\n(Open, Promo, ngày lễ thật)") +
    proc("p3", "Dự báo từng ngày + khoảng 80%\\n(model đã lưu)") +
    proc("p4", "Số ca = ⌈doanh thu ÷ doanh thu 1 ca⌉\\n≥ số ca tối thiểu; đóng cửa = 0") +
    cond("c2", "Tổng ca có sẵn > 0?") + proc("p5", "Phân bổ ca theo tỷ trọng doanh thu\\n(không vượt nhu cầu)") +
    proc("p6", "Tổng hợp theo Ngày/Tuần/Tháng;\\nxếp hạng cửa hàng, so với cùng kỳ năm trước") +
    proc("p7", "Gợi ý theo quy tắc: tăng ca ngày cao điểm, xem lại vận hành,\\ncân nhắc khuyến mãi (What-if), thiếu ca, ngày đóng cửa") +
    io("o1", "KPI, biểu đồ doanh thu + số ca,\\nbảng kế hoạch, xếp hạng, gợi ý (có căn cứ)") +
    cond("c3", "Bấm 'Xuất CSV'?") + io("o2", "Lưu kế hoạch + file gợi ý") + end("e") + '''
s -> i1 -> p0 -> c1; c1 -> p1 [label="Không"]; p1 -> i1 [style=dashed]; c1 -> p2 [label="Có"];
p2 -> p3 -> p4 -> c2; c2 -> p5 [label="Có"]; c2 -> p6 [label="Không"]; p5 -> p6 -> p7 -> o1 -> c3;
c3 -> o2 [label="Có"]; c3 -> e [label="Không"]; o2 -> e;
''')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, dot in DIAGRAMS.items():
        subprocess.run(["dot", "-Tpng", "-o", str(OUT / f"{name}.png")], input=dot.encode("utf-8"), check=True)
        print("Đã vẽ", OUT / f"{name}.png")


if __name__ == "__main__":
    main()
