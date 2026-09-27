# So sánh 4 Mô hình Classification cho MNIST

## 1. Logistic Regression (LR)

### Ưu điểm
✓ Rất nhanh - training < 1 giây  
✓ Dễ hiểu - chỉ là weighted sum  
✓ Ít overfitting  

### Nhược điểm
✗ Chỉ tốt với dữ liệu linearly separable  
✗ Độ chính xác thấp - ~92% trên MNIST  

---

## 2. Support Vector Machine (SVM)

### Ưu điểm
✓ Mạnh với high-dimensional data  
✓ Kernel trick xử lý dữ liệu phi tuyến tính  
✓ Độ chính xác cao - ~95% trên MNIST  

### Nhược điểm
✗ **Chậm** - Training: O(n²) hoặc O(n³)  
✗ Cần **tuning kernel**  
✗ Cần **scaling dữ liệu**  

---

## 3. Neural Network (NN)

### Ưu điểm
✓ **MẠNH NHẤT** cho image classification  
✓ Độ chính xác cao nhất - **98-99%**  
✓ Học features tự động  

### Nhược điểm
✗ Cần **nhiều data**  
✗ Khó **tuning hyperparameters**  
✗ Dễ **overfitting**  

---

## 4. Decision Tree (DT)

### Ưu điểm
✓ Dễ hiểu & interpretable  
✓ Không cần scaling  

### Nhược điểm
✗ Dễ overfitting  
✗ Độ chính xác thấp - ~87%  

---

## 📊 BẢNG SO SÁNH

| Tiêu chí | LR | SVM | NN | DT |
|----------|----|----|----|----|
| Tốc độ | ⚡⚡⚡⚡⚡ | ⚡⚡ | ⚡⚡⚡ | ⚡⚡⚡⚡ |
| Độ chính xác | 92% | 95% | **98-99%** | 87% |

---

## 🎯 KẾT LUẬN

Chọn TOP 2: Neural Network + SVM