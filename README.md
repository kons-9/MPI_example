# MPI
スパコンプログラミングの課題で作成したMPIプログラムです。

## 実行
transpose.cを実行する場合は以下のコマンドを実行してください。
```
make run ARG=transpose
```

## 説明
1. transpose.c
行列転置をscatterとgatherを用いて実装したプログラムです。
2. powm.c
べき乗法を用いて行列の固有値を求めるプログラムです。
3. mat_vec.c
行列とベクトルの積を計算するプログラムです。
4. cannon.c
キャノンのアルゴリズムを用いて行列の積を計算するプログラムです。
