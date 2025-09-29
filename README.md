# HW1 書面報告 — 簡單線性回歸與 CRISP-DM

---

## 1. 作業目的

在本作業中，我們透過簡單線性回歸 (Simple Linear Regression, SLR) 來探索 **自變數 X 與因變數 Y 之間的線性關係**。

目的包括：

1. 練習從資料生成到模型擬合、評估的完整流程。
2. 熟悉 **CRISP-DM (Cross Industry Standard Process for Data Mining)** 方法論：
    - 系統性理解問題、資料、建模與評估流程。
3. 使用互動式 Python 工具 (Streamlit) 觀察模型對不同參數和資料分布的反應。

---

## 2. 資料理解（Data Understanding）

我們使用**人工生成的線性資料**，公式如下：


 <img width="248" height="50" alt="image" src="https://github.com/user-attachments/assets/10454ede-1833-4bd5-a69b-ec05595d8aa4" />


其中：
- a：斜率 (Slope)
- b：截距 (Intercept)
- ϵ∼N(0,σ2)：高斯雜訊

資料生成特性：

- 可調整點數 n，範圍 [10, 1000]。
- 斜率 a 與截距 b 可自由設定。
- 雜訊標準差 σ 可控制資料波動。
- 隨機種子確保結果可重現。

> Python 實作示例：
> 

```python
np.random.seed(seed)
X = np.linspace(-10, 10, n_points)
noise = np.random.normal(0, noise_std, n_points)
Y = a * X + b + noise
```

- 透過散佈圖觀察資料分布，確認線性趨勢。

---

## 3. 模型建立（Modeling）

### 3.1 模型選擇

使用 **普通最小二乘 (OLS) 線性回歸**：

<img width="178" height="54" alt="image" src="https://github.com/user-attachments/assets/1b9d90e2-e59e-4e72-9cc3-5a646d05ea33" />


### 3.2 Python 實作

```python
from sklearn.linear_model import LinearRegression
X_reshaped = X.reshape(-1, 1)
model = LinearRegression()
model.fit(X_reshaped, Y)
Y_pred = model.predict(X_reshaped)
```

---

### 4. 模型評估（Evaluation）

### 4.1 性能指標

- **決定係數** R²：衡量擬合解釋變異比例

<img width="273" height="75" alt="image" src="https://github.com/user-attachments/assets/137f2642-a39c-4a7f-8883-f708275e2f48" />

### 4.22 觀察與討論

- **低噪聲**時，擬合直線接近真實線。
- **高噪聲**或資料點少時，擬合線波動大，R²降低。
- 交互式修改參數可以直觀理解模型對資料的敏感性。

---

## 5. 部署與互動（Deployment）

- 使用 **Streamlit Web App**：
    - 左側 Sidebar 控制參數：
        - 斜率 a、截距 b
        - 資料點數量 n
        - 雜訊標準差 σ
    - 右側 Main Panel 顯示：
        - 散佈圖與擬合直線
        - 離群點
        - 模型係數與評估指標
- 可即時調整並觀察結果，提升對回歸模型的理解。
- 支援 CSV 資料下載，方便後續分析或報告插圖。

---

## 8. 結論（Conclusion）

1. 簡單線性回歸可有效描述自變數與因變數之間的線性關係。
2. CRISP-DM 提供系統化流程，從資料生成到模型評估皆有章可循。
3. 互動式設計幫助理解：
    - 雜訊對模型的影響
    - 資料量對估計穩定性的影響
    - 離群值偵測的重要性
4. 實作過程提供 Python 實務操作經驗，並熟悉可視化分析與 Web 部署。

---

## 9. 圖表示

1. **資料與擬合線圖**
    - 原始資料點（藍色）、擬合線（紅色）、離群點（橘色）

link :  https://simple-regression-graph.streamlit.app/

<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/47e3f59b-03fd-4b93-b8bf-19046a705e3d" />

