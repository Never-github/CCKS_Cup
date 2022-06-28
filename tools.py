import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def getThreshold(rrank):
    distanceFlagList = rrank
    distanceFlagList = sorted(distanceFlagList, key=lambda sp: sp[0], reverse=False)

    threshold = distanceFlagList[0][0] - 0.01
    minValue = 0
    currentValue = 0
    for i in range(1, len(distanceFlagList)):
        if distanceFlagList[i - 1][1] == 1:
            currentValue += 1
        else:
            currentValue -= 1

        if currentValue < minValue:
            threshold = (distanceFlagList[i][0] + distanceFlagList[i - 1][0]) / 2.0
            minValue = currentValue
    # print('threshold... ', threshold)
    return threshold

def transform_prediction_to_json(predictions, urls, output_path='result.txt'):
    with open(output_path, 'w') as f:
        for u, p in zip(urls, predictions):
            dic = {'url': u, 'label': int(p)}
            json_dic = json.dumps(dic)
            f.writelines(json_dic+'\n')

def evaluation_func(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"acc {acc} p {p} r {r} f1 {f1}")