# HW1 書面報告 — 簡單線性回歸與 CRISP-DM

## 1. 封面
**題目**：HW1 — Simple Linear Regression with CRISP-DM  
**學生姓名**：XXX  
**日期**：2025/09/29  
**課程名稱**：XXX  

---

## 2. 研究目的（Business Understanding）
在本作業中，我們透過簡單線性回歸 (Simple Linear Regression, SLR) 來探索 **自變數 X 與因變數 Y 之間的線性關係**。  

目的包括：

1. 練習從資料生成到模型擬合、評估的完整流程。
2. 熟悉 **CRISP-DM (Cross Industry Standard Process for Data Mining)** 方法論：
   - 系統性理解問題、資料、建模與評估流程。
3. 使用互動式 Python 工具 (Streamlit) 觀察模型對不同參數和資料分布的反應。

---

## 3. 資料理解（Data Understanding）
我們使用**人工生成的線性資料**，公式如下：

\[
Y = a \cdot X + b + \epsilon
\]

其中：

- \(a\)：斜率 (Slope)  
- \(b\)：截距 (Intercept)  
- \(\epsilon \sim \mathcal{N}(0, \sigma^2)\)：高斯噪聲

資料生成特性：

- 可調整點數 \(n\)，範圍 [10, 1000]  
- 斜率 \(a\) 與截距 \(b\) 可自由設定  
- 噪聲標準差 \(\sigma\) 可控制資料波動  
- 隨機種子確保結果可重現  

> **Python 實作示例**：
```python
np.random.seed(seed)
X = np.linspace(-10, 10, n_points)
noise = np.random.normal(0, noise_std, n_points)
Y = a * X + b + noise
透過散佈圖觀察資料分布，確認線性趨勢。

4. 資料準備（Data Preparation）
將生成資料整理為 DataFrame (x, y)。

資料檢查：

無缺失值

離群點檢測：

殘差 
𝑟
𝑖
=
𝑦
𝑖
−
𝑦
^
𝑖
r 
i
​
 =y 
i
​
 − 
y
^
​
  
i
​
 

判斷 
∣
𝑟
𝑖
∣
>
2
𝜎
𝑟
∣r 
i
​
 ∣>2σ 
r
​
  為潛在離群值

可視化：

散佈圖：X vs Y

殘差圖：預測值 vs 殘差

為互動設計，使用者可修改：

斜率、截距、點數、噪聲、seed

5. 模型建立（Modeling）
5.1 模型選擇
使用 普通最小二乘 (OLS) 線性回歸：

𝑦
^
=
𝑎
^
⋅
𝑋
+
𝑏
^
y
^
​
 = 
a
^
 ⋅X+ 
b
^
 
5.2 Python 實作
python
複製程式碼
from sklearn.linear_model import LinearRegression
X_reshaped = X.reshape(-1, 1)
model = LinearRegression()
model.fit(X_reshaped, Y)
Y_pred = model.predict(X_reshaped)
5.3 模型參數
斜率估計 
𝑎
^
=
𝑚
𝑜
𝑑
𝑒
𝑙
.
𝑐
𝑜
𝑒
𝑓
[
0
]
a
^
 =model.coef 
[
​
 0]

截距估計 \hat{b} = model.intercept_

可計算 
𝑅
2
R 
2
 、平均殘差、殘差標準差

6. 模型評估（Evaluation）
6.1 性能指標
決定係數 
𝑅
2
R 
2
 ：

𝑅
2
=
1
−
∑
(
𝑦
𝑖
−
𝑦
^
𝑖
)
2
∑
(
𝑦
𝑖
−
𝑦
ˉ
)
2
R 
2
 =1− 
∑(y 
i
​
 − 
y
ˉ
​
 ) 
2
 
∑(y 
i
​
 − 
y
^
​
  
i
​
 ) 
2
 
​
 
平均殘差：
1
𝑛
∑
(
𝑦
𝑖
−
𝑦
^
𝑖
)
n
1
​
 ∑(y 
i
​
 − 
y
^
​
  
i
​
 )

殘差標準差：
𝜎
𝑟
=
1
𝑛
∑
(
𝑦
𝑖
−
𝑦
^
𝑖
)
2
σ 
r
​
 = 
n
1
​
 ∑(y 
i
​
 − 
y
^
​
  
i
​
 ) 
2
 
​
 

6.2 殘差分析
繪製 殘差圖，檢查是否存在系統性偏差

離群值標記，視覺化辨識異常點

6.3 觀察與討論
低噪聲時，擬合直線接近真實線

高噪聲或資料點少時，擬合線波動大，R²降低

交互式修改參數可以直觀理解模型對資料的敏感性

7. 部署與互動（Deployment）
使用 Streamlit Web App：

左側 Sidebar 控制參數：

斜率 
𝑎
a、截距 
𝑏
b

資料點數量 
𝑛
n

噪聲標準差 
𝜎
σ

隨機種子

右側 Main Panel 顯示：

散佈圖與擬合直線

殘差圖與離群點

模型係數與評估指標

可即時調整並觀察結果，提升對回歸模型的理解

支援 CSV 資料下載，方便後續分析或報告插圖

8. 結論（Conclusion）
簡單線性回歸可有效描述自變數與因變數之間的線性關係

CRISP-DM 提供系統化流程，從資料生成到模型評估皆有章可循

互動式設計幫助理解：

噪聲對模型的影響

資料量對估計穩定性的影響

殘差分析與離群值偵測的重要性

實作過程提供 Python 實務操作經驗，並熟悉可視化分析與 Web 部署

9. 圖表示意
資料與擬合線圖

原始資料點（藍色）、擬合線（紅色）、離群點（橘色）

殘差圖

殘差 vs 預測值，0 線參考

圖表可使用 st.pyplot(fig) 於 Streamlit app 生成，並截圖插入報告

10. 參考文獻
Han, J., Kamber, M., & Pei, J. (2012). Data Mining: Concepts and Techniques. Morgan Kaufmann.

Scikit-learn Documentation: https://scikit-learn.org/stable/

Streamlit Documentation: https://docs.streamlit.io/

yaml
複製程式碼

---

如果你願意，我可以幫你**加入範例圖表截圖的 Markdown 語法**，讓報告看起來像真正提交的作業文件。  

你希望我幫你加嗎？
