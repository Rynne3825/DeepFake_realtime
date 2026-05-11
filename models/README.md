# Models

Thư mục này dùng để chứa các file model `.onnx` và cache model phụ trợ.

Repo GitHub không bắt buộc phải chứa toàn bộ model nặng. Nếu checkout sạch chưa có model, tải từ:

- [Google Drive models](https://drive.google.com/drive/folders/1VlyLts-t0g9VRqpowmpp8hE5h6j-rgkD?usp=sharing)

Các file app có thể dùng:

- `inswapper_128.onnx`
- `inswapper_128_fp16.onnx` (tùy chọn, dùng khi bật FP16)
- `gfpgan-1024.onnx`
- `GPEN-BFR-256.onnx`
- `GPEN-BFR-512.onnx`

Ngoài các file trên, `insightface.app.FaceAnalysis(name="buffalo_l")` còn có thể tải thêm cache model `buffalo_l` vào máy local ở lần chạy đầu.

Nếu thiếu model swap/enhance cần thiết, ứng dụng sẽ báo `FileNotFoundError`.
