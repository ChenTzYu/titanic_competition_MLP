# titanic_competition_MLP

在kaggle上的MLP練習

結果請直接點開 titanic_competition.md
前半部分使用pandas做數據處理
神經網路的部分使用keras來完成
model的建立可以在檔案裡直接搜尋model，因為模型很簡單，所以模型的部分其實代碼並不多

因為titanic的資料庫出來很長一段時間了
所以如果用整個資料庫來做訓練導致overfitting就能夠得到很高分
但是此舉並沒有意義，因此就沒有再對模型的超參數做後續的修改

最後結果，validation accuracy約為0.85，實際在test sets上約為0.77
