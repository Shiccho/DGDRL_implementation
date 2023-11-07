# Description
Implementation of "Real-time Motion Planning for Robotic Teleoperation Using Dynamic-goal Deep Reinforcement Learning"

https://ieeexplore.ieee.org/document/9108691

# 説明
## 論文について
このリポジトリは、"Real-time Motion Planning for Robotic Teleoperation Using Dynamic-goal Deep Reinforcement Learning"という論文に対する追実装です。
この論文では、ロボットアームの制御に強化学習を用いることで、障害物を回避する軌跡を短時間で生成することができます。
短時間での経路計画ができることで、リアルタイムの操作が可能です。

強化学習の環境としてPyBulletという物理シミュレータを用いています。
強化学習アルゴリズムとしては、PPOというものを用いています。

## Demo
白い箱が障害物です。この障害物を回避した経路計画が可能です。
始点と終点はランダムに生成しています。

https://github.com/Shiccho/DGDRL_implementation/assets/94341374/6e4c3b72-603b-4909-960b-453e3f408385

