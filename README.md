# shared-deep-learning-from-scratch
ゼロから学ぶディープラーニング輪読会の共有レポジトリ

# 利用方法
- 基本的には個人別にmasterからブランチを切って、その中で開発を行う。
  - ブランチ名はpersonal/[githubアカウント名]を基本にする。
- ./README.md等の全体で共有したいファイルはmasterに直接commitする。
- 利用方法等の共有したい情報は./README.mdに記載する。

## 開発環境

### DevContainer を使用しない場合

本を参考に作成

### DevContainer を使用する場合

#### 事前準備

- Docker インストール
- VSCode インストール
- VSCode の拡張機能「Remote - Containers」インストール
  - https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers
- 本リポジトリの clone

#### 手順

1. VSCode 起動
2. 左下の緑色のアイコンクリック
3. 「Remote-Containers - Open Folder in Container...」クリック
4. しばらく待つ
   - 初回の場合コンテナ image の取得や作成が行われる
5. 起動したら開発可能

## 参考

- [公式リポジトリ](https://github.com/oreilly-japan/deep-learning-from-scratch)
