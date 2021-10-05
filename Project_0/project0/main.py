def get_sum_metrics(predictions, metrics=[]):
    for i in range(3):
        n=i
        metrics.append(lambda x: x + n)

    sum_metrics = 0
    for metric in metrics:
        sum_metrics += metric(predictions)



    return sum_metrics

print(get_sum_metrics(2))