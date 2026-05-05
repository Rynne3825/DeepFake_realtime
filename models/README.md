# Models

Thư mục này không chứa các file model nặng trong GitHub.

Sau khi clone repo, hãy tải model từ:

- [Google Drive models](https://drive.google.com/drive/folders/1VlyLts-t0g9VRqpowmpp8hE5h6j-rgkD?usp=drive_link)

Sau đó đặt các file sau vào thư mục này:

- `inswapper_128.onnx` hoặc `inswapper_128_fp16.onnx`
- `gfpgan-1024.onnx`
- `GPEN-BFR-256.onnx`
- `GPEN-BFR-512.onnx`

Ứng dụng sẽ ưu tiên `inswapper_128.onnx`. Nếu không có, app sẽ tự dùng `inswapper_128_fp16.onnx`.

Nếu thiếu model, ứng dụng sẽ báo `FileNotFoundError`.
