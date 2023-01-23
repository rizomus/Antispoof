from sklearn.metrics import confusion_matrix

def confusion_matrix_scorer(cm):
    return {'tn': cm[0, 0], 'fp': cm[0, 1],
            'fn': cm[1, 0], 'tp': cm[1, 1]}

def rates(y_true, y_pred, ret=False):
    cm = confusion_matrix(y_true, y_pred)
    sc = confusion_matrix_scorer(cm)
    FAR = sc['fp'] / len(y_true)        # False Acceptance Rate
    FRR = sc['fn'] / len(y_true)        # False Rejection Rate
    HTER = (FAR + FRR) / 2              # Half-Total Error Rate

    if ret:
        return FAR, FRR, HTER
    print(f'FAR = {FAR * 100:.1f} %, FRR = {FRR * 100:.1f} %, HTER = {HTER * 100:.1f} %')
