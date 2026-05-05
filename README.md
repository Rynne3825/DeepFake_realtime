# DeepFake Realtime

Ứng dụng hoán đổi khuôn mặt theo thời gian thực qua webcam, viết bằng Python với `insightface`, `onnxruntime` và `customtkinter`.

## Yêu cầu

- Windows
- Python `3.11`
- Webcam
- GPU NVIDIA là tùy chọn, không bắt buộc

## Cài đặt

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Chuẩn bị model

Thư mục `models/` không được đưa lên GitHub. Sau khi clone repo, cần tự tạo lại thư mục này và tải model từ Google Drive sau:

- [Google Drive models](https://drive.google.com/drive/folders/1VlyLts-t0g9VRqpowmpp8hE5h6j-rgkD?usp=sharing)

Sau đó chép các model sau vào thư mục `models/`:

- `models\inswapper_128.onnx` hoặc `models\inswapper_128_fp16.onnx`
- `models\gfpgan-1024.onnx`
- `models\GPEN-BFR-256.onnx`
- `models\GPEN-BFR-512.onnx`

Ứng dụng sẽ ưu tiên `inswapper_128_fp16.onnx`. Nếu không có, app sẽ tự fallback sang `inswapper_128.onnx`.

Nếu thiếu các file trên, ứng dụng sẽ báo `FileNotFoundError`.

Lưu ý thêm: `insightface` còn dùng bộ model `buffalo_l` cho phát hiện/nhận diện khuôn mặt. Vì code hiện tại gọi `insightface.app.FaceAnalysis(name="buffalo_l")`, máy chạy lần đầu có thể cần Internet để InsightFace tự tải bộ model này về cache local.

## Chạy ứng dụng

```powershell
.venv\Scripts\python.exe main.py
```

## GPU có chạy hay không?

Ứng dụng sẽ tự dò `Execution Provider` của `onnxruntime`.

- Nếu máy có NVIDIA GPU, driver phù hợp, và `onnxruntime-gpu` tải được `CUDAExecutionProvider`, app sẽ chạy bằng GPU.
- Nếu không có CUDA phù hợp, app sẽ tự rơi về `CPUExecutionProvider`.
- Code hiện cũng có nhánh `DmlExecutionProvider`, nhưng `requirements.txt` hiện chưa cài `onnxruntime-directml`, nên cấu hình chia sẻ hiện tại nên hiểu là tối ưu cho NVIDIA GPU hoặc CPU.

Bạn có thể tự kiểm tra sau khi cài:

```powershell
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

Nếu thấy `CUDAExecutionProvider` trong danh sách thì máy đó chạy GPU NVIDIA được.

## Gửi cho người khác cần những gì?

Người nhận repo cần:

1. Clone mã nguồn.
2. Cài Python `3.11`.
3. Tạo môi trường ảo và cài `requirements.txt`.
4. Tự chép các file model vào thư mục `models/`.
5. Có Internet ở lần chạy đầu nếu `buffalo_l` chưa có trong cache của InsightFace.
6. Chạy `python main.py`.

## Ghi chú về dependency

`requirements.txt` hiện đã bao phủ các dependency runtime được import trực tiếp trong mã nguồn:

- `opencv-python`
- `numpy`
- `Pillow`
- `onnxruntime-gpu`
- `insightface`
- `customtkinter`

Tệp này chưa bao gồm model `.onnx`, vì model là dữ liệu ngoài repo chứ không phải package Python.
