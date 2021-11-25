# AOI_defect-detection

※ AOI瑕疵分類
1. 競賽來源：[人工智慧共創平台](https://aidea-web.tw/topic/285ef3be-44eb-43dd-85cc-f0388bf85ea4)

2. 議題提供單位：工業技術研究院

3. 比賽內容：自動光學檢查（Automated Optical Inspection，簡稱 AOI）。希望藉由 AOI 影像訓練深度學習模型，判讀軟性電子顯示器的表面瑕疵。

4. 資料集：，6個類別（正常類別 + 5 種瑕疵類別）的影像資料。
    * train_images.zip：訓練所需的影像資料（PNG格式），共計2,528張。
    * test_images.zip：測試所需的影像資料（PNG格式），共計10,142張。
    * train.csv：包含 2 個欄位，ID(影像的檔名) 和 Label(瑕疵分類類別)。
    * test.csv：包含 2 個欄位，ID 和 Label。
    * Label：（0 表示 normal，1 表示 void，2 表示 horizontal defect，3 表示 vertical defect，4 表示 edge defect，5 表示 particle）。

---
※ 電腦軟硬體配置
1. 作業系統：Windows10
2. CPU：Intel-10700K
3. GPU：RTX2060S(顯存8G)
4. RAM：48G
5. Frame：Tensorflow

---
※程式說明：
1. 01_detect_classify.py：移動圖檔到對應類別的資料夾。
2. 02_transfer_three_channels.py：將單通道圖檔轉換成三通道圖檔。
3. 03_train_model_MobileV3.py：以遷移學習的方式訓練影像分類模型。
4. 04_predit.py：辨識圖檔屬於哪個分類。

---
※實作過程：
1. 讀取train.csv後，依照類別移動圖檔到對應類別的資料夾。

2. 移動圖檔到對應資料夾後，統計各類別資料數量。

3. 承上，發現資料集有不均衡問題，嘗試解決方法如下：
  3.1 資料數量不足的類別，以複製圖檔的方式，使每個類別的圖檔數量一致。
  3.2 模型訓練時，設定class_weight：透過Keras class_weight傳遞權重，可調節損失函數對不同類別的敏感度，更加關注代表性不足的類別。

4. 遷移學習+資料前處理：
  * 因訓練集數量較少(即便是複製後，遠低於5000張)，故採用keras application預訓練進行遷移學習，模型效果與訓練效率較佳。
  * 官方提供的訓練集為單通道灰階圖片，而預訓練模型之訓練集(imagenet之子集)為三通道圖片，故訓練前須先進行資料前處理，將待訓練樣本轉換為三通道圖片。

5. 選用MobileNetV3預訓練模型：因樣本數量遠低於5000張，較適合使用輕量化、淺層模型。

6. 學習曲線與訓練過程

7. 預測test資料集
  * 複製圖檔：99.35881%。
    ![image](https://github.com/midnightla0710/AOI_defect-detection/blob/main/pictures/copy/rank.jpg)
  * 設定class_weight：99.13686%。
    ![image](https://github.com/midnightla0710/AOI_defect-detection/blob/main/pictures/class_weight/rank.jpg)

8. 未來改進：使用GAN進行資料擴增，並選用準確率更高的深層。

---
※ 最終分數與排名：MobileNetV3模型的測試準確99.35881%，位居排名第19。
