# Visor — ROS2 ビューワー (RViz代替) プロジェクト

## 背景と目的

- RViz2の代替となるROS2ビューワーを自作する
- 動機: RVizはMac上でフリーズ等の問題があり、プラグイン開発のハードルも高い。
  自作して拡張していく方が速いと判断した
- 実行環境: macOS(Apple Silicon)・Linux・Windows(x86_64、`.msi` で配布)の 3 OS 対応。
  ビューワー自体はROS2環境に依存しない(rclcpp/rclpyを使わない)
- 将来的な用途イメージ: AMR/AGVフリート監視。倉庫管制画面のような
  「暗背景 + シアン/パープルの発光アクセント」のビジュアルを目指す

## アーキテクチャ(決定済み・変更しないこと)

- ROS2側は rmw_zenoh(RMW_IMPLEMENTATION=rmw_zenoh_cpp)で稼働している。
  実機・シミュレータともに zenoh ルーターが起動済みの前提
- 本アプリは zenoh クライアントとしてルーターに直接接続する
  (zenoh-bridge-ros2dds は使わない。ブリッジは存在しない)
- ネイティブDDS、rclcpp、rclpy は絶対に使わない(依存地獄を避けるための決定)
- Linux同一マシンでシミュレータと併用する場合も接続構成は同じ
  (localhostのルーターに接続するだけ)。接続構成は常に1本
- 接続エンドポイントは設定で指定(デフォルト: tcp/localhost:7447)
- ROS_DOMAIN_ID も設定値として持つ(デフォルト: 0)
- 3D描画は wgpu(Mac: Metal / Linux: Vulkan / Windows: DX12)。プラットフォーム分岐コードは
  原則書かない。必要になった場合は設計判断ログ(decisions.md)に理由を記録する
- 型のフィールド構造はネットワークから取得できないため、common_interfaces の
  .msg 定義ファイル一式を assets/ に同梱し、動的CDRデコーダで解釈する

## 対象バージョン

<!-- TODO: 実環境に合わせて確定させること -->
- ROS 2 ディストリビューション: (例: Jazzy)
- rmw_zenoh バージョン: (例: 0.x.y / ブランチ名)
- zenoh crate: rmw_zenoh が依存する zenoh と同系列(1.x)に固定。勝手に更新しない

## rmw_zenoh固有の仕様(comm/実装時の要点)

- キー表現の形式:
  `<domain_id>/<トピック名>/<マングリングされた型名>/<型ハッシュ>`
  例: `0/scan/sensor_msgs::msg::dds_::LaserScan_/RIHS01_xxxx...`
- トピック名に含まれる `/` は keyexpr 上では別の区切りにエンコードされる。
  実装時に rmw_zenoh 本体(github.com/ros2/rmw_zenoh)のソースで
  対象バージョンの最新仕様を必ず確認すること
- トピック発見: liveliness token(`@ros2_lv/<domain_id>/**`)の購読と、
  既存トークンの取得(liveliness get)で行う
- 型名の相互変換関数が必要:
  `sensor_msgs::msg::dds_::LaserScan_` ←→ `sensor_msgs/msg/LaserScan`
- ペイロードは CDR シリアライズされた ROS2 メッセージ
  (4バイトのエンカプセレーションヘッダ付き)。decode/ 以降は共通処理

## 技術スタック(固定・勝手に変更/追加しないこと)

- 言語: Rust(開発者はRustほぼ未経験。説明は丁寧に)
- GUI: eframe + egui + egui_dock
- 3D描画: wgpu(egui-wgpu の Callback で統合)
- 通信: zenoh crate 1.x系(バージョンは「対象バージョン」参照)
- 数学: nalgebra
- スレッド間通信: crossbeam-channel
  (zenoh受信は tokio タスク、UIはメインスレッド)

## フォルダ構成

```
visor/                         # 作業ディレクトリ実体は ros2_viewer/（据え置き）
├── CLAUDE.md
├── Cargo.toml
├── README.md                  # 起動手順のみ簡潔に
├── assets/
│   └── msgs/                  # .msg定義の同梱
│       ├── std_msgs/msg/*.msg
│       ├── geometry_msgs/msg/*.msg
│       ├── sensor_msgs/msg/*.msg
│       ├── nav_msgs/msg/*.msg
│       └── tf2_msgs/msg/*.msg
├── src/
│   ├── main.rs                # eframe起動のみ。ロジックを書かない
│   ├── app.rs                 # eguiアプリ本体(egui_dockレイアウト保持)
│   ├── theme.rs               # 暗背景+シアン系のカラー定義を一箇所に集約
│   ├── comm/
│   │   ├── mod.rs
│   │   ├── session.rs         # zenoh接続、購読管理、channel供給
│   │   ├── discovery.rs       # liveliness購読によるノード/トピック/型の列挙
│   │   └── keyexpr.rs         # rmw_zenohキー表現の組立/分解、型名マングリング変換
│   ├── decode/
│   │   ├── mod.rs
│   │   ├── msg_parser.rs      # .msgテキスト→型定義
│   │   ├── cdr.rs             # 動的CDRデコード
│   │   └── value.rs           # デコード結果の中間表現(JSONライクな値型)
│   ├── tf/
│   │   ├── mod.rs
│   │   └── buffer.rs          # リングバッファ、lookup_transform
│   ├── render/
│   │   ├── mod.rs             # Renderer trait + レジストリ
│   │   ├── viewport.rs        # wgpuセットアップ、カメラ、グリッド、軸
│   │   └── renderers/
│   │       ├── mod.rs
│   │       ├── laser_scan.rs  # 以降、1表示型=1ファイル
│   │       ├── point_cloud2.rs
│   │       ├── marker.rs
│   │       ├── occupancy_grid.rs
│   │       ├── path.rs
│   │       └── urdf.rs
│   └── ui/
│       ├── mod.rs
│       ├── topic_list.rs      # トピック一覧パネル
│       ├── raw_view.rs        # Rawメッセージパネル
│       ├── frame_tree.rs      # TFツリーパネル
│       └── display_list.rs    # 3D表示中アイテムの管理パネル
├── src/bin/
│   └── probe.rs               # 疎通確認CLI(cargo run --bin probe)
└── tests/
    └── fixtures/              # CDRデコードの既知バイト列フィクスチャ
```

## 設計方針

- プラグイン拡張性が最重要。新しい表示型は Renderer trait の実装ファイルを
  renderers/ に1つ追加し、レジストリに1行登録するだけで増やせる構造を維持する
- decode/ と tf/ と comm/keyexpr は必ずユニットテストを書く
  (既知バイト列→期待値のフィクスチャ。String/配列/ネスト型/
   エンディアン/アラインメントを網羅)
- 重い処理(点群デコード、頂点バッファ構築)で描画スレッドをブロックしない
- 色はすべて theme.rs に定義する(色をコード中に散らばらせない)
- 3Dシーンの配色は暗背景+シアン系で固定。テーマで塗り替えない
  (theme.rs トップレベルの const。頂点色に焼かれるものもここ)
- パネル・ウィジェットのUI色だけが Dark / Light の2モードを持つ
  (theme::ui::palette()。Dark 側の値は当初の暗背景+シアン系そのまま)

## 開発ルール

- コード内コメント・doc コメント・UI 文字列は英語で書く(Rust に限らず yml / wxs / toml など
  リポジトリ内のコード系ファイル全部)。日本語はリポジトリ外の開発文書だけ
- 各タスク完了時に `cargo clippy` と `cargo test` を通すこと
- wgpu / egui / zenoh はAPIの変化が速い。ビルドエラー時は最新のcrate
  ドキュメントを確認してから修正すること
- rmw_zenoh のキー表現・liveliness仕様は、実装前に対象バージョンの
  ソースコードで確認すること
- 開発文書(タスク一覧 tasks.md・設計判断ログ decisions.md・プラグイン作成ガイド plugins.md など)は
  公開しない方針のためリポジトリ管理外。置き場所は各開発者のローカル設定を参照する。
  リポジトリに docs/ を作らない(.gitignore 済み)
- 1セッションで扱うのは原則1タスク。タスク一覧は tasks.md を参照
- 設計判断が発生したら decisions.md に日付付きで追記する

## 検証環境

- ROS2環境(実機またはシミュレータ)の起動コードは既存のものを使用
  (このリポジトリのスコープ外)
- 実機・シミュレータともに rmw_zenoh + zenoh ルーターが稼働している
- 接続先エンドポイントは CLI引数または環境変数 `ZENOH_ENDPOINT` で指定
  (デフォルト: tcp/localhost:7447 — Linuxローカルのシミュレータ環境では
   引数なしで動く)
