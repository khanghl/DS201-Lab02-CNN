## Lab 02: Mạng Noron Tích Chập (Convolutional Neural Networks - CNN)

### Mục Tiêu
- Nắm vững nguyên lý hoạt động và cách tính output shape, số tham số của các tầng trong CNN: Convolution, Pooling (AvgPool2d, MaxPool2d, AdaptiveAvgPool2d), BatchNorm, Dropout, Concatenate và Residual Connection

- Sử dụng thống nhất duy nhất bộ dữ liệu VinaFood21 (21 lớp món ăn Việt Nam, cùng SEED = 42 cho phép chia train/val/test) xuyên suốt cả 4 bài thực hành:
  - **Bài 1:** LeNet-5 điều chỉnh (64×64×3, 339.421 tham số, so sánh AvgPool2d và MaxPool2d)
  
  - **Bài 2:** GoogLeNet rút gọn (64×64×3, 4 InceptionBlock + AdaptiveAvgPool2d(1), 232.905 tham số)
  
  - **Bài 3\*:** ResNet-18 huấn luyện từ đầu (96×96×3, [2, 2, 2, 2] BasicBlock, 11.187.285 tham số, so sánh Không và Có Data Augmentation)
  
  - **Bài 4\*:** ResNet-50 Pretrained (microsoft/resnet-50, 224×224×3, 23.551.061 tham số, so sánh Feature Extraction lr=1e-3 và Fine-tuning lr=1e-4)

---

### Cấu Trúc Folder 
- data.py: Định nghĩa MEAN, STD chuẩn ImageNet, hàm build_transforms (tách riêng train & val/test) và DataLoader đồng bộ theo seed=42

- models.py: Định nghĩa LeNet5, InceptionBlock, GoogLeNetMini, BasicBlock, ResNet18, PretrainedResnet và các hàm đếm tham số n_params, n_trainable_params

- engine.py: Hàm train_one_epoch, evaluate (model.eval() + @torch.no_grad()) và fit (chọn checkpoint best.pt theo val macro-F1 cao nhất, lưu history.csv)

- utils.py: Các hàm vẽ biểu đồ phân bố lớp, đường cong Loss/Accuracy/macro-F1, Confusion Matrix 21×21, đánh giá tập test evaluate_test và bảng tổng hợp so sánh print_comparison_table

- assignments_.ipynb: Báo cáo thực hành hoàn chỉnh cho cả 4 bài

---

### Lưu ý Bộ Dữ Liệu
- Tập dữ liệu **VinaFood21** (~2.7GB): [Google Drive Link](https://drive.google.com/file/d/1UpZOf0XlwvB4rKpyZ35iwTA8oWHqDBbR/view?usp=share_link)
  
- Giải nén vào /VinaFood21/ (hoặc để ensure_vinafood21() trong data.py tự động tải và giải nén khi chạy trên Google Colab)
