# Pytorch Implementation of NeMo's Conformer-CTC
❗ 実装はまだ途中で、以下のところは実装予定
- [ ] より近いデータの前処理
- [ ] NoamAnneallingスケジューたの実装・導入
- [ ] パラメータ数には差が出ていますが、その理由を検証

## 目的
NeMoのモデルをPytorchで実装することによって、必要な前処理とモデルの理解を深めることを目的とする。

## 実行方法
```
poetry install
```
```
eval $(poetry env activate)
```
```
python -m src.scripts.run_training
```
