# LiDARデータを用いた自動人（マネキン）検出システム

本リポジトリは、**マネキン（人の代替物）検出**を目的とした  
**LiDAR点群処理パイプライン**を実装したものです。

本システムでは以下の手法を用いています：

- **DBSCANクラスタリング**（物体領域の分離）
- **TinyPointNet**（軽量な PointNet ベース分類モデル）
- **Open3D による可視化**（クラスタ表示および 3D バウンディングボックス）

---

## 必要な環境・ライブラリ（事前にインストール）

以下のライブラリをインストールしてください。

```bash
pip install numpy
pip install open3d
pip install scikit-learn
pip install matplotlib
pip install torch torchvision torchaudio

**プロジェクト構成**
Automated-human-detection-system-using-LiDAR-data/
│
├── .venv/                      # Python 仮想環境（git では無視）
│
├── data/                       # 生の点群データ（git では無視）
│   ├── mannequin/              # マネキンの PLY 点群サンプル
│   ├── background/             # 背景物体の PLY 点群サンプル
│   └── test data/              # テスト用のフルシーン点群
│                               # ※ フォルダ名にスペースあり
│
├── data_npy/                   # NPY形式に変換された学習用データ（git では無視）
│   ├── mannequin/              # マネキン点群（Nx3 の numpy 配列）
│   └── background/             # 背景点群（Nx3 の numpy 配列）
│
├── clusters_data/              # フルシーンから抽出されたクラスタ（git では無視）
│   └── cluster_*.ply
│
├── scratch code/               # 過去の試行・破棄したスクリプト（参考用）
│
├── .gitignore                  # Git の無視設定（データ・モデル・IDE設定など）
│
├── tinypointnet2_model.py      # TinyPointNet ニューラルネットワーク定義
├── prepare_dataset.py          # PLY → NPY 変換および Dataset クラス
├── train_pointnet.py           # TinyPointNet の学習スクリプト
├── check_model.py              # 学習済みモデルの簡易動作確認
├── create_cluster_drone.py     # フルシーン点群に対する DBSCAN クラスタリング
├── test_better_pointnet.py     # 検出・可視化を行うメイン実行スクリプト
│
└── pointnet_mannequin_classifier.pth
                                # 学習済みモデル重み（git では無視）

