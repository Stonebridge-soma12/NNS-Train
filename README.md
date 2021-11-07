# GPU Server

모델 학습 서버

각 인스턴스마다 실행되어 메세지 큐로 부터 들어오는 학습 요청을 기다린다.

## Request body

- 메세지큐에 전달되는 Body는 다음과 같은 형태다.

```json
{
  "train_id": 1,
  "user_id": 1,
  "config": {
    "optimizer_name": "Adam",
    "optimizer_config": {
      "learning_rate": 0.001,
      "beta_1": 0.9,
      "beta_2": 0.999,
      "epsilon": 1e-07,
      "amsgrad": false
    },
    "loss": "binary_crossentropy",
    "metrics": [
      "accuracy"
    ],
    "batch_size": 32,
    "epochs": 4,
    "early_stop": {
      "usage": true,
      "monitor": "loss",
      "patience": 2
    },
    "learning_rate_reduction": {
      "usage": true,
      "monitor": "val_accuracy",
      "patience": 2,
      "factor": 0.25,
      "min_lr": 0.0000003
    }
  },
  "data_set": {
    "train_uri": "https://datasetURL.com",
    "valid_uri": "",
    "shuffle": false,
    "label": "labelName",
    "normalization": {
      "usage": true,
      "method": "MinMax"
    },
    "kind": "TEXT"
  }
}

```