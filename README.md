# Description
Implementation of "Real-time Motion Planning for Robotic Teleoperation Using Dynamic-goal Deep Reinforcement Learning"

https://ieeexplore.ieee.org/document/9108691

# 説明
## 論文について
このリポジトリは、Kaveh Kamaliらの"Real-time Motion Planning for Robotic Teleoperation Using Dynamic-goal Deep Reinforcement Learning"という論文の実装です。
この論文では、ロボットアームの制御に強化学習を用いることで、障害物を回避する軌跡を短時間で生成することができます。
短時間での経路計画ができることで、リアルタイムの操作が可能です。

## 実装について
昨年の夏、学部4年時に作成したコードです。

強化学習の環境としてPyBulletという物理シミュレータを用いています。
強化学習アルゴリズムとして、PPOを用いています。

実際にプログラムを実行する場合には、ロボットアームのモデルを準備する必要があるため、デモ動画を掲載しています。

## Demo
白い箱が障害物です。この障害物を回避した経路計画が可能です。
始点と終点はランダムに生成しています。

https://github.com/Shiccho/DGDRL_implementation/assets/94341374/6e4c3b72-603b-4909-960b-453e3f408385

## How to use
1. makePList.pyを実行し、始点と終点のデータセットを作成
2. main.pyを実行
