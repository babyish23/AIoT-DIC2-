# DIC2 多元線性迴歸：50 Startups

用 scikit-learn 做 50 Startups 的多元線性迴歸，流程依 CRISP-DM。Boston Housing 是第二題，直接呼叫同一支 `mlr_workflow.py`，只換資料集和目標欄位。

## 執行

```bash
pip install -r requirements.txt
python main.py
python solve_50_startups.py
python solve_boston_housing.py
```

`python main.py` 會把 31 種特徵組合全部跑完，印出測試 MSE、Top 20，以及最好的 5 組，並寫進網頁。

也可以換成自己的 CSV：

```bash
python solve_50_startups.py --url https://example.com/data.csv --target Profit
```

50 Startups 預設資料：

https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv

分割是 `test_size=0.2`、`random_state=0`。State 以 California 當對照組。

## 結果

Lasso、Forward SFS、RFE、ElasticNet 的測試 RMSE 都在 k=2 最低：

- 特徵：R&D Spend、Marketing Spend
- Test RMSE：8198.80
- Test R²：0.9474

5 個特徵全放進去時，Test RMSE 升到 9137.99。

Boston Housing 用同一套五種特徵選擇。測試 RMSE 最低是 Exhaustive Search、k=8、RMSE 5.6389。

圖和表格在 `output/startups/`、`output/boston/`。

## 網頁

`index.html` 讀取 `data/presets.json`。頁面可以：

1. 用下拉選單切換 50 Startups 和 Boston Housing
2. 貼上任意 CSV 網址，或選本機 CSV

課程內建的兩個網址會顯示 sklearn 的結果。其他 CSV 在瀏覽器裡即時做迴歸。
