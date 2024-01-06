# Description
Implementation of "Real-time Motion Planning for Robotic Teleoperation Using Dynamic-goal Deep Reinforcement Learning"

https://ieeexplore.ieee.org/document/9108691

# 説明
## 論文について
このリポジトリは、Kaveh Kamaliらの"Real-time Motion Planning for Robotic Teleoperation Using Dynamic-goal Deep Reinforcement Learning"という論文についての個人による非公式の実装です。
この論文では、ロボットアームの制御に強化学習を用いることで、障害物を回避する軌跡を短時間で生成する手法が提案されています。
短時間での経路計画ができるため、リアルタイムでロボットアームの操作が可能です。

強化学習アルゴリズムとして、PPOを用いています。
## 実装について

強化学習の環境としてPyBulletという物理シミュレータを用いています。

実際にプログラムを実行する場合には、ロボットアームのモデルを準備する必要があるため、デモ動画を掲載しています。

## Demo
ランダムに生成した始点と終点に対して、障害物を回避した軌跡を生成しています。
白い箱が静的な障害物です。

https://github.com/Shiccho/DGDRL_implementation/assets/94341374/6e4c3b72-603b-4909-960b-453e3f408385

## How to use
1. makePList.pyを実行し、始点と終点のデータセットを作成
2. main.pyを実行
