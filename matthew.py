from sklearn import metrics

y_true = [0, 1, 2, 3, 4, 5, 3]
y_pred = [0, 1, 2, 3, 4, 5, 6]

res = metrics.matthews_corrcoef(y_true, y_pred)
print(res)
