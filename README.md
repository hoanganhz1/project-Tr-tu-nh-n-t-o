# project-Tr-tu-nh-n-t-o
FaceSecure là hệ thống nhận diện khuôn mặt thời gian thực sử dụng MTCNN và FaceNet (InceptionResnetV1). Chương trình hỗ trợ thu thập dataset, trích xuất embedding 512 chiều, lưu trữ dữ liệu khuôn mặt, xác minh 1:1 và nhận dạng 1:N bằng Cosine Distance với giao diện PyQt5.
## Tính năng

- Phát hiện khuôn mặt theo thời gian thực sử dụng MTCNN
- Căn chỉnh khuôn mặt và trích xuất đặc trưng bằng FaceNet
- Tạo vector đặc trưng khuôn mặt (embedding) 512 chiều
- So khớp dựa trên khoảng cách Cosine
- Xác thực khuôn mặt 1:1
- Nhận diện khuôn mặt 1:N
- Quản lý tập dữ liệu khuôn mặt
- Ngưỡng nhận diện có thể tùy chỉnh
- Giao diện ứng dụng desktop hiện đại sử dụng PyQt5
