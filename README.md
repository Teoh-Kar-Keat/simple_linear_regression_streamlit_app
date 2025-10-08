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

APP 鏈接 :  https://simple-regression-graph.streamlit.app/

<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/47e3f59b-03fd-4b93-b8bf-19046a705e3d" />

## 10. 指令

**Prompt：**

請幫我生成一份完整的《HW1 書面報告 — 簡單線性回歸與 CRISP-DM》，

報告需以 **Markdown 格式** 撰寫，內容包含文字說明、數學公式與 Python 程式碼。

---

報告必須分為以下章節：

1. **作業目的 (Objective)**
    
    說明作業重點：以簡單線性回歸探索 X、Y 的線性關係，並結合 CRISP-DM 流程。
    
2. **資料理解 (Data Understanding)**
    
    - 介紹資料生成公式 Y=aX+b+εY = aX + b + \varepsilonY=aX+b+ε，其中 ε∼N(0,σ2)\varepsilon \sim N(0, \sigma^2)ε∼N(0,σ2)。
    
    - 解釋各參數：a、b、σ、n、seed。
    
    - 加入 Python 生成程式碼，能調整這些參數並繪出散佈圖。
    
3. **建模 (Modeling)**
    
    - 使用 `sklearn.linear_model.LinearRegression` 完成普通最小二乘 (OLS) 模型。
    
    - 請顯示模型擬合公式與估計結果。
    
    ### 報告要求：
    
    - 程式碼需輸出斜率、截距、R²。
    
4. **評估 (Evaluation)**
    
    - 使用 R² 作為主要指標。
    
    - 生成多組不同噪聲標準差 σ 的比較圖，說明雜訊對模型的影響。
    
    - 圖表說明：資料散佈圖、真實線、擬合線、R² 值。
    
5. **部署與互動 (Deployment)**
    
    - 使用 Streamlit 實作互動式 Web App：
    
    - Sidebar 控制項（a、b、n、σ）
    
    - 主畫面顯示：
    
    - 散佈圖與擬合線
    
    - 模型係數與 R²
    
    - 提供 CSV 下載功能（X, Y）。
    
6. **結論 (Conclusion)**
    
    - 整合 CRISP-DM 六階段在本實作的應用。
    
    - 簡述線性回歸與噪聲關係、資料量對模型穩定度的觀察。
    
7. **附錄 (Appendix)**
    
    - 附上 Streamlit 完整可執行程式碼。

