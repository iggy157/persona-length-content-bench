# persona-length-content-bench

対話エージェントに **性格ペルソナ・語数上限・発話内容** の 3 つの指示を同時に与え、語数上限を絞り発話内容の項目を増やしながら、
各モデルが 3 つをどこまで守り続けられるか（多制約下の耐性）と、守れなくなったとき何を残し何を捨てて何を優先するか（優先傾向）を測るベンチマークです。
15 のオープンモデルで測った結果をもとに、耐性で絞り、残したい指示で選び、速度とサイズで決める、という順でモデルを選べます。

論文：原田悠矢・狩野芳伸「対話エージェントにおけるペルソナ対文字数・発話内容の優先度比較ベンチマークによる LLM 間比較と最適化」人工知能学会 計算創律科学研究会（SIG-CSS）、2026 年 12 月。

| | |
| --- | --- |
| モデル選定ページ | https://133.167.32.100/aiwolf/persona-length-content-bench/ |
| 生データ（生成した返答・判定モデルの評定・採点表） | https://huggingface.co/datasets/iggy157/persona-length-content-bench |
| 付録（本文で省いた表と限界、設計の意図） | [docs/appendix.md](docs/appendix.md) |
| 論文の表と図 | [results/tables](results/tables)、[results/figs](results/figs) |

## 構成

```
config/            実験の設定（モデル、主実験 exp_b4 / exp_b5、基準実験 exp_c、判定モデル、300 の場面）
src/pcb/           実装（試行の構築 → 生成 → 性格の評定 → 採点）
data/raw/ipip/     IPIP-NEO の 30 側面の項目（公開ドメイン）
data/built/        構築した試行（プロンプト、ペルソナ、発話内容の項目）。exp_b4: 3 項目、exp_b5: 0〜2 項目、exp_c: 基準実験
outputs/exp_*/     採点表 scores.csv.gz と評定 ratings.csv.gz（生成した返答そのものと判定モデルの生出力は Hugging Face）
scripts/make_results.py   採点表と評定から論文のすべての表と図を作る
results/tables/    表 3〜8 と全条件の集計（CSV と Markdown）
results/figs/      図 1〜5
web/               モデル選定ページ（HTML 1 枚、scripts/build_select_page.py で results/tables から生成）
```

## 実験の流れ

主実験は、性格ペルソナ（IPIP-NEO の 30 側面 × 強・弱・なし）をシステムプロンプトに、語数上限（3, 5, 8, 10, 15, 20, 30 語、無制限）と
発話内容（開始時刻・部屋・持ち物・曜日から 0〜3 項目）をターンごとのプロンプトに置き、友人からの 300 通のメッセージに返信させます。
各条件を temperature 0.7 で 3 回生成し、モデルあたり 86,400 試行です。コード上は 3 項目の `exp_b4` と 0〜2 項目の `exp_b5` に分かれています。

基準実験 `exp_c` は、同じ種類の指示をシステムプロンプトとターンごとのプロンプトの両方に矛盾する値で置き（性格 vs 性格、語数 vs 語数、内容 vs 内容）、
どちらに従うかを両順序で測ります。優先傾向からは、この値を引いて指示のタイミングの効果を取り除きます。

判定：語数は空白区切りで数え、発話内容は項目ごとの正規表現で検出します。性格は、返答だけを見た判定モデル（gpt-oss-120b と Llama-3.3-70B）に
その側面の IPIP-NEO 10 項目を 1〜5 で評定させ、性格なしの返答との差を表出量とします。

## 再現

採点表と評定からの再現（表と図）：

```
uv sync
uv run python scripts/make_results.py
```

生成からの再現には vLLM で各モデルを `config/models.yaml` の `base_url` に立て、次を実行します（モデルあたり 86,400 試行 × 3 周）。

```
uv run pcb build-b4 && uv run pcb build-b5 && uv run pcb build-c
uv run pcb generate --exp b4 --models <name>        # 生成（exp b5, c も同様。周は generate 内の seed で指定）
uv run pcb rate --exp b4 --judges gpt-oss-120b,llama-3.3-70b   # 判定モデルによる性格の評定
uv run pcb evaluate --exp b4                        # 採点表 scores.csv
```

生データの各ファイルの形式は Hugging Face のデータセットカードに書いてあります。

## 評価の定義

- **耐性**：3 つの指示を同時に与えた条件で、性格の残存率（表出量 ÷ 性格のみのときの表出量）、語数上限を守った割合、発話内容をすべて伝えた割合の調和平均をとり、
  語数上限 8 段階 × 項目数 1〜3 の全組合せで平均した値。3 つすべてが高いときにしか高くなりません。
- **崩壊点**：性格ペルソナなしで発話内容 3 項目と語数上限を与え、語数と内容の両方を 80% 以上守れた最も厳しい語数上限。
- **優先傾向**：崩壊点で 3 つを同時に与えたときに各指示を守った割合から、基準実験でその指示の側が従われた割合を引いた差。
  正ならその指示を余計に残し、負なら捨てています。

## ライセンス

コードは MIT、データは CC BY 4.0。IPIP-NEO の項目は公開ドメインです。
