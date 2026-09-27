
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("1. DOC DU LIEU")
print("=" * 60)

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(f"Train shape: {train_df.shape}")   # (42000, 785) -> cot dau la 'label', 784 cot con lai la pixel
print(f"Test shape : {test_df.shape}")    # (28000, 784) -> khong co label (dung de submit Kaggle)
print()
print("5 dong dau cua train:")
print(train_df.head())

print("\n" + "=" * 60)
print("2. TACH LABEL VA PIXEL")
print("=" * 60)

y_train = train_df["label"]
X_train = train_df.drop(columns=["label"])

print(f"So luong mau: {X_train.shape[0]}")
print(f"So dac trung (pixel) moi mau: {X_train.shape[1]}  (= 28 x 28 = 784)")
print(f"Gia tri pixel nho nhat: {X_train.values.min()}, lon nhat: {X_train.values.max()}")


n_missing = train_df.isnull().sum().sum()
print(f"So gia tri thieu (missing values) trong tap train: {n_missing}")

print("\n" + "=" * 60)
print("3. PHAN BO CAC LOP (0-9)")
print("=" * 60)

class_counts = y_train.value_counts().sort_index()
print(class_counts)

plt.figure(figsize=(8, 5))
class_counts.plot(kind="bar", color="#4C72B0")
plt.title("Phan bo so luong mau theo tung chu so (0-9)")
plt.xlabel("Chu so")
plt.ylabel("So luong mau")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
print("-> Da luu bieu do phan bo class: class_distribution.png")
plt.close()

print("\n" + "=" * 60)
print("4. HIEN THI ANH MAU CHO TUNG CHU SO")
print("=" * 60)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for digit in range(10):
    # lay 1 anh dau tien co label = digit
    idx = y_train[y_train == digit].index[0]
    img = X_train.loc[idx].values.reshape(28, 28)

    ax = axes[digit // 5, digit % 5]
    ax.imshow(img, cmap="gray")
    ax.set_title(f"Digit: {digit}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("sample_digits.png", dpi=150)
print("-> Da luu anh mau: sample_digits.png")
plt.close()


print("\n" + "=" * 60)
print("5. THONG KE DO SANG (PIXEL INTENSITY)")
print("=" * 60)

mean_intensity = X_train.values.mean()
std_intensity = X_train.values.std()
print(f"Do sang trung binh cua pixel: {mean_intensity:.2f}")
print(f"Do lech chuan: {std_intensity:.2f}")

plt.figure(figsize=(8, 5))
plt.hist(X_train.values.flatten(), bins=50, color="#55A868")
plt.title("Phan bo gia tri pixel (toan bo tap train)")
plt.xlabel("Gia tri pixel (0-255)")
plt.ylabel("Tan suat")
plt.tight_layout()
plt.savefig("pixel_distribution.png", dpi=150)
print("-> Da luu bieu do phan bo pixel: pixel_distribution.png")
plt.close()


print("\n" + "=" * 60)
print("TOM TAT KET QUA EDA")
print("=" * 60)
print(f"- Tong so mau train: {X_train.shape[0]}, test: {test_df.shape[0]}")
print(f"- Moi anh: 28x28 = 784 pixel, gia tri 0-255 (grayscale)")
print(f"- 10 lop (chu so 0-9), phan bo: xem class_distribution.png")
print(f"- Khong co missing values" if n_missing == 0 else f"- CANH BAO: co {n_missing} gia tri thieu")
print("- Cac file bieu do da duoc luu trong thu muc hien tai")
print("=" * 60)