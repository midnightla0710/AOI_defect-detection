# AOI_defect-detection

競賽來源：[人工智慧共創平台-AOI瑕疵分類](https://aidea-web.tw/topic/285ef3be-44eb-43dd-85cc-f0388bf85ea4)

議題提供單位：工業技術研究院

比賽內容：自動光學檢查（Automated Optical Inspection，簡稱 AOI）[1]，為高速高精度光學影像檢測系統，運用機器視覺做為檢測標準技術，可改良傳統上以人力使用光學儀器進行檢測的缺點，應用層面包括從高科技產業之研發、製造品管，以至國防、民生、醫療、環保、電力…等領域。工研院電光所投入軟性電子顯示器之研發多年，在試量產過程中，希望藉由 AOI 技術提升生產品質。本次邀請各界資料科學家共襄盛舉，針對所提供的 AOI 影像資料，來判讀瑕疵的分類，藉以提升透過數據科學來加強 AOI 判讀之效能。

本議題所提供之影像資料，包含 6 個類別（正常類別 + 5 種瑕疵類別）。
train_images.zip：訓練所需的影像資料（PNG格式），共計 2,528 張。
train.csv：包含 2 個欄位，ID(影像的檔名) 和 Label(瑕疵分類類別)。
test_images.zip：測試所需的影像資料（PNG格式），共計 10,142 張。
test.csv：包含 2 個欄位，ID 和 Label。
Label：（0 表示 normal，1 表示 void，2 表示 horizontal defect，3 表示 vertical defect，4 表示 edge defect，5 表示 particle）。
Label：瑕疵分類類別（其值只能是下列其中之一：0、1、2、3、4、5）。

專案介紹
本專案目的為藉由AOI影像訓練深度學習模型辨識晶片表面瑕疵，使用框架為tensorflow。預訓練mobilenetv3模型的測試準確已達到99.35881%。

程式說明：
01_detect_classify.py：
02_transfer_three_channels.py
03_train_model_MobileV3.py
04_predit.py

實作過程：
1. ：讀取CSV，依照類別移動到對應資料夾
2. 分配後，各類別資料量如下，有資料不均衡問題。(類別與數量表格化)
3. 嘗試解決方法如下：
  3.1 采用复制“重复使用”的方式，使得数据“看起来”均衡。
  3.2 class_weights：透過Keras class_weight傳遞權重，可調節損失函數對不同類別的敏感度，更加關注代表性不足的類別。99.13686%
4. 遷移學習+資料前處理：樣本數量少，採用keras application預訓練進行遷移學習，模型效果與訓練效率較佳。官方提供的訓練集為單通道灰階圖片，而預訓練模型之訓練集(imagenet子集)為三通到圖片，故訓練前須先進行資料前處理，將待訓練樣本轉換為三通道圖片。
5. 選用mobilenetv3(因為樣本擴增後，樣本數量僅XXX，數量稀少，較適合使用淺層模型)
6. 學習曲線與訓練過程
7. 預測test資料集，分數與排名
8. 未來改進：資料擴增-GAN或opencv合成
