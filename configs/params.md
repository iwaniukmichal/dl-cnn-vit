# Rekomendowane hiperparametry dla configow

## Zrodla

1. [`plan.tex`](/Users/michaliwaniuk/Desktop/DL/.agents/skills/dl-implementation-plan-planner/references/plan.tex)
   - Glowny punkt odniesienia dla faz eksperymentu. Plan zaklada ResNet-18 i DeiT-Tiny, ale tutaj interpretacja dla CNN zostala przelozona na EfficientNet-B3.
2. [`/Users/michaliwaniuk/Desktop/DL/src/archdyn/data/transforms.py`](/Users/michaliwaniuk/Desktop/DL/src/archdyn/data/transforms.py)
   - Pokazuje co realnie znacza `baseline`, `standard`, `advanced`, `combined` w tym repo. `standard` to crop/flip/color jitter, a CutMix nie jest tu aplikowany.
3. [`/Users/michaliwaniuk/Desktop/DL/src/archdyn/training/supervised.py`](/Users/michaliwaniuk/Desktop/DL/src/archdyn/training/supervised.py) i [`/Users/michaliwaniuk/Desktop/DL/src/archdyn/training/fewshot.py`](/Users/michaliwaniuk/Desktop/DL/src/archdyn/training/fewshot.py)
   - Pokazuja ograniczenia implementacji: supervised obsluguje tylko `none` i `cosine`; few-shot nie uzywa schedulera, nie uzywa `training.batch_size`, a CutMix jest stosowany tylko w supervised.
4. [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://proceedings.mlr.press/v97/tan19a.html)
   - Zrodlo dla charakterystyki EfficientNet, natywnej skali modelu, transferability oraz typowej regularizacji CNN z rodziny EfficientNet. W oryginalnym treningu autorzy uzywaja RMSProp, niskiego weight decay i drop connect.
5. [Torchvision docs: `efficientnet_b3`](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b3.html)
   - Oficjalna dokumentacja uzywanego wariantu. Potwierdza, ze wagi ImageNet dla B3 sa zwiazane z preprocesingiem `resize 320 -> crop 300`, wiec `input_size: 224` jest kompromisem pod fairness/compute, a nie natywna recepta modelu.
6. [Training data-efficient image transformers & distillation through attention (DeiT)](https://proceedings.mlr.press/v139/touvron21a.html)
   - Najwazniejsze zrodlo dla DeiT: AdamW, cosine, weight decay 0.05, stochastic depth 0.1 i silna augmentacja sa zgodne z literatura. To punkt odniesienia dla search/phase3.
7. [Oficjalne repo DeiT (`README_deit.md`)](https://raw.githubusercontent.com/facebookresearch/deit/main/README_deit.md)
   - Pokazuje praktyczne komendy treningowe i fine-tuningowe. Przydaje sie do oceny, jak konserwatywnie ustawic LR w repo bez warmupu.
8. [Prototypical Networks for Few-shot Learning](https://papers.neurips.cc/paper/6996-prototypical-networks-for-few-shot-learning)
   - Punkt odniesienia dla `n_way`, `k_shot`, `q_query`, liczby epizodow i ogolnej logiki few-shot. W papierze query=15 jest standardem, a wiekszy `way` czesto pomaga, ale zwykle na zbiorach z wieksza liczba klas niz 10.
9. [CINIC-10 Is Not ImageNet or CIFAR-10](https://www.research.ed.ac.uk/en/datasets/cinic-10-is-not-imagenet-or-cifar-10/)
   - Potwierdza rozmiar i split zbioru: 270k obrazow, 90k/90k/90k, 10 klas, 32x32. To uzasadnia ostrozniejsze regularyzowanie niz przy ImageNet i sens lekkiego few-shot na 10% danych.
10. [CutMix: Regularization Strategy to Train Strong Classifiers With Localizable Features](https://openaccess.thecvf.com/content_ICCV_2019/html/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.html)
   - Zrodlo dla `cutmix_alpha ~= 1.0` i sensu mieszajacych augmentacji w klasyfikacji supervised. Nie jest to jednak domyslna recepta dla prototypical networks.

## Jak czytac rekomendacje

- Ponizej sa zakresy docelowe do poprawy YAML-i, nie "jedyna poprawna wartosc".
- Dla EfficientNet-B3 czesc rekomendacji jest inferencja z literatury i oficjalnych docs pod transfer learning na CINIC-10, bo plan byl napisany pod ResNet-18.
- Dla DeiT zakresy sa celowo konserwatywne, bo obecna implementacja nie ma warmupu, repeated augmentation, mixup ani rand erasing.
- Dla EfficientNet-B3 `input_size: 224` jest OK, jesli priorytetem jest uczciwe porownanie z DeiT i koszty. Jesli priorytetem jest maksymalizacja wyniku CNN, lepszy bylby rozmiar blizszy natywnemu B3, czyli 300.

## Status implementacyjny po poprawkach

1. W `fewshot` scheduler moze byc teraz ustawiany z YAML, tak samo jak `lr`, `weight_decay` i `drop_path`.
2. `training.batch_size` zostal usuniety z YAML-i few-shot, zeby nie sugerowal nieistniejacego batching-u epizodow.
3. W `fewshot` wariant `advanced` moze teraz uzywac `cutmix_alpha`; CutMix jest aplikowany do query images podczas treningu.
4. W `fewshot` wariant `combined` oznacza teraz `standard` transformacje plus CutMix na query images podczas treningu.
5. W `supervised` `advanced` = CutMix-only, a `combined` = standard + CutMix. To nadal jest slabsze niz pelna recepta DeiT z Mixup/RandAugment/RandErasing, ale jest spojne w ramach tego repo.
6. Dla `custom_cnn` `dataset.input_size` zostal ustawiony jawnie na `32`, a EfficientNet-B3 na `300`.
7. Few-shot przeszedl z `n_way: 10` na `n_way: 5`, a frakcje danych zostaly zmniejszone, zeby lepiej odpowiadaly scenariuszowi low-data.

## Audyt aktualnych configow

### Phase 1

| Config | Ocena | Komentarz |
| --- | --- | --- |
| `configs/phase1/custom_cnn_baseline.yaml` | OK | `lr=1e-3`, `wd=1e-4`, `epochs=50`, `bs=128` sa sensowne dla malego CNN trenowanego od zera. `input_size` jest juz jawnie ustawione na `32`, wiec config nie jest mylacy. |
| `configs/phase1/efficientnet_b3_baseline.yaml` | OK | `lr=3e-4`, `wd=1e-4`, brak schedulera i `drop_path=0.0` sa sensowne jako baseline transfer learning. `input_size` zostal podniesiony do `300`, wiec jest blizej natywnej recepty B3. |
| `configs/phase1/deit_tiny_baseline.yaml` | OK | `lr=5e-5`, `wd=0.05`, brak schedulera i `drop_path=0.0` sa konserwatywne, ale dobrze pasuja do baseline bez mocnej augmentacji. |

### Phase 2

| Config | Ocena | Komentarz |
| --- | --- | --- |
| `configs/phase2/efficientnet_b3_search.yaml` | OK warunkowo | Space ma dobra os regularizacji (`drop_path`, `wd`, `scheduler`), a frakcja spadla do `0.1`, co lepiej pasuje do szybkiego searchu. Nadal warto rozwazyc nizsza dolna granice LR po pierwszych wynikach. |
| `configs/phase2/deit_tiny_search.yaml` | OK | `lr=[5e-5,2e-4]`, `wd=[0.05,0.1]`, `drop_path=[0,0.1]`, `scheduler=[none,cosine]` dobrze odwzorowuja literature DeiT, a frakcja `0.1` lepiej odpowiada "reduced search". |

### Phase 3

| Config | Ocena | Komentarz |
| --- | --- | --- |
| `configs/phase3/efficientnet_b3_baseline.yaml` | OK warunkowo | `cosine`, `drop_path=0.1`, `wd=5e-4` sa sensowne, ale `lr=1e-3` jest dla full fine-tune dosc agresywny przy `bs=32`. Warto go traktowac jako gorna granice, nie default. |
| `configs/phase3/efficientnet_b3_standard.yaml` | OK warunkowo | Jak wyzej. Sama augmentacja `standard` jest sensowna dla CINIC-10 i pretrained CNN. |
| `configs/phase3/efficientnet_b3_advanced.yaml` | OK warunkowo | CutMix-only ma sens jako osobna ablacja. Nadal glowna watpliwosc dotyczy `lr=1e-3`. |
| `configs/phase3/efficientnet_b3_combined.yaml` | OK warunkowo | Najbardziej sensowny wariant aug dla supervised CNN, ale nadal z ostroznym podejsciem do `lr=1e-3`. |
| `configs/phase3/deit_tiny_baseline.yaml` | OK | `lr=2e-4`, `wd=0.1`, `cosine`, `drop_path=0.1` sa zgodne z kierunkiem z DeiT. |
| `configs/phase3/deit_tiny_standard.yaml` | OK | Rozsadne ustawienie pod repo bez warmupu i bez RandAugment/Mixup. |
| `configs/phase3/deit_tiny_advanced.yaml` | OK warunkowo | CutMix-only jest zgodne z planem jako osobna ablacja, ale slabsze od pelnej literaturowej recepty DeiT. |
| `configs/phase3/deit_tiny_combined.yaml` | OK | Najblizsze temu, co w tym repo da sie zrobic pod "mocniejsza" augmentacje dla DeiT. |

### Phase 4: few-shot

| Config | Ocena | Komentarz |
| --- | --- | --- |
| `configs/phase4/protonet_efficientnet_b3_baseline.yaml` | OK warunkowo | `n_way=5` i `fraction=0.05` nadaja epizodom wieksza roznorodnosc. Dodatkowo config ma juz `scheduler`, wiec latwiej recznie przepisac najlepsze parametry z phase2. |
| `configs/phase4/protonet_efficientnet_b3_standard.yaml` | OK | Jak wyzej; `standard` dalej jest rozsadnym referencyjnym wariantem few-shot. |
| `configs/phase4/protonet_efficientnet_b3_advanced.yaml` | OK warunkowo | `advanced` moze juz realnie uzywac CutMix w treningu few-shot, ale nadal warto potwierdzic empirycznie, czy query-only CutMix pomaga na tym setupie. |
| `configs/phase4/protonet_efficientnet_b3_combined.yaml` | OK | `combined` oznacza teraz sensownie `standard + CutMix` takze w few-shot. |
| `configs/phase4/protonet_deit_tiny_baseline.yaml` | OK warunkowo | `n_way=5`, `fraction=0.05` i aktywny `scheduler` usuwaja glowny problem metodologiczny poprzedniej wersji. |
| `configs/phase4/protonet_deit_tiny_standard.yaml` | OK | Standardowa augmentacja ma sens i epizody sa teraz bardziej "meta" niz przy poprzednim 10-way. |
| `configs/phase4/protonet_deit_tiny_advanced.yaml` | OK warunkowo | `advanced` jest juz faktycznie innym wariantem niz baseline, bo moze uzywac CutMix na query. |
| `configs/phase4/protonet_deit_tiny_combined.yaml` | OK | `combined` ma teraz sensowna implementacje w few-shot i nie sprowadza sie juz do samego `standard`. |

### Phase 4: reduced supervised

| Config | Ocena | Komentarz |
| --- | --- | --- |
| `configs/phase4/reduced_supervised_efficientnet_b3.yaml` | Do korekty | Na 10% danych `lr=1e-3` jest raczej zbyt wysokie. Lepiej zejsc do `1e-4 .. 3e-4` i zostawic cosine + combined aug. |
| `configs/phase4/reduced_supervised_deit_tiny.yaml` | OK warunkowo | `lr=2e-4` to gorna czesc sensownego zakresu; jesli trening bedzie niestabilny, pierwsza zmiana to zejscie do `1e-4` lub `5e-5`. |

### Analysis i ensemble

| Config | Ocena | Komentarz |
| --- | --- | --- |
| `configs/analysis/embeddings_efficientnet_b3.yaml` | OK | `drop_path=0.1`, `bs=64` i `input_size=300` sa spojne z nowym setupem EfficientNet-B3. |
| `configs/analysis/embeddings_deit_tiny.yaml` | OK | `drop_path=0.1` i `bs=64` sa sensowne dla analizy embeddingow. |
| `configs/ensembles/supervised_best_models.yaml` | OK | Tu istotne hiperparametry treningowe sa juz "dziedziczone" z checkpointow. `bs=64` do inference/stackingu jest sensowne. |

## Docelowe hiperparametry do poprawy YAML-i

### Phase 1: baseline architecture comparison

| Model | Drop path | LR | Weight decay | Scheduler | Epochs | Batch size | Few-shot | Augmentation | Komentarz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `custom_cnn` | `0.0` dla malego modelu; `0.0 .. 0.1` tylko jesli architektura zrobi sie wyraznie glebsza i ma skipy | `3e-4 .. 1e-3` | `1e-4 .. 5e-4` | `none` dla czystego baseline; `cosine` dopiero jesli model zrobi sie wiekszy lub trening dluzszy | `50 .. 100` | `128 .. 256` | n/d | `baseline` | Dla malego CNN z 32x32 i AdamW obecne ustawienie jest sensowne. Jesli model urosnie, zwykle najpierw zwieksz `wd`, potem rozwaz `cosine`; `drop_path` ma sens glownie dla glebszych residualowych wariantow. |
| `efficientnet_b3` | `0.0 .. 0.1` | `1e-4 .. 3e-4` jako bezpieczny default; `5e-4` jako agresywniejszy wariant | `1e-4 .. 5e-4` | `none` albo `cosine`; dla samego baseline brak schedulera jest OK | `20 .. 40` | `32 .. 64` | n/d | `baseline` | W oryginale EfficientNet uzywa mocniejszej regularizacji i natywnej rozdzielczosci; dla transfer learning na CINIC-10 przy `224` lepszy jest ostrozny LR niz recepta ImageNet. |
| `deit_tiny` | `0.0 .. 0.1` | `5e-5 .. 1e-4` jako baseline; `2e-4` dopiero po searchu | `0.05` typowo, ewentualnie `0.1` | `none` albo `cosine`; bez warmupu nie przesadzac z LR | `20 .. 30` | `32 .. 64` | n/d | `baseline` | Literatury DeiT sugeruja mocna regularizacje, ale obecne repo nie ma warmupu ani pelnej recepty augmentacyjnej, wiec baseline powinien pozostac konserwatywny. |

### Phase 2: hyperparameter search na 10-20% treningu

| Model | Drop path | LR | Weight decay | Scheduler | Epochs | Batch size | Few-shot | Augmentation | Komentarz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `efficientnet_b3` | przetestowac `0.0` vs `0.1`; ewentualnie `0.05` jako srodek | minimum `1e-4` lub `2e-4` vs wyzsza opcja `3e-4` lub `5e-4`; `1e-3` tylko jako skrajny test | `1e-4` vs `5e-4` | `none` vs `cosine` | `15 .. 20` | `32 .. 64` | n/d | `baseline` | Najwieksza rzecz do poprawy wzgledem obecnego configu to dodanie nizszego LR. Search ma sprawdzic stabilnosc fine-tune, nie tylko "jak wysoko da sie podniesc krok". |
| `deit_tiny` | `0.0` vs `0.1` | `5e-5` vs `2e-4` jest OK; opcjonalnie `1e-4` zamiast jednego ze skrajow | `0.05` vs `0.1` | `none` vs `cosine` | `15 .. 20` | `32 .. 64` | n/d | `baseline` | Obecny space jest dobrze ustawiony. Jesli kiedys dodacie warmup, wtedy sensownie byloby rozszerzyc LR w gore. |

### Phase 3: full-data supervised + augmentation study

| Model | Drop path | LR | Weight decay | Scheduler | Epochs | Batch size | Few-shot | Augmentation | Komentarz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `efficientnet_b3` | `0.05 .. 0.1` | `1e-4 .. 5e-4` jako typowy zakres; `1e-3` tylko jesli search na pewno to uzasadni | `1e-4 .. 5e-4` | `cosine` preferowany | `30 .. 50` | `32 .. 64` | n/d | `baseline`, `standard`, `advanced`, `combined`; dla `advanced/combined` `cutmix_alpha=1.0` jest OK | Dla pretrained CNN na CINIC-10 najlepsze beda zwykle `standard` albo `combined`. `advanced` jako CutMix-only ma sens eksperymentalnie, ale nie jako oczywisty default. |
| `deit_tiny` | `0.1` typowo | `5e-5 .. 2e-4` | `0.05 .. 0.1` | `cosine` | `30 .. 50` | `32 .. 64` | n/d | `standard`/`combined` preferowane; `cutmix_alpha=1.0` jest zgodne z DeiT, ale repo nie ma Mixup/RandAugment/RandErasing | Dla DeiT mocniejsza augmentacja zwykle pomaga bardziej niz dla CNN. Poniewaz repo ma okrojona implementacje, `combined` jest najblizsze literaturze, ale nadal slabsze od pelnej recepty DeiT. |

### Phase 4A: prototypical networks / few-shot

| Backbone | Drop path | LR | Weight decay | Scheduler | Epochs | Batch size | Few-shot | Augmentation | Komentarz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `efficientnet_b3` | `0.0 .. 0.1`; przy malo danych raczej nie przekraczac `0.1` | `1e-4 .. 3e-4` | `1e-4 .. 5e-4` | obecnie nieuzywany w kodzie; jesli kiedys dodacie, `cosine` albo `plateau` | `20 .. 40` albo do plateau walidacji | nieuzywane w obecnej implementacji | Zalecane: `n_way=5`, `k_shot in {1,5}`, `q_query=10 .. 15`, `train_episodes=100 .. 300` na epoke, `val_episodes=100`, `test_episodes=300 .. 600` | tylko `baseline` albo lekki `standard` | Na CINIC-10 10-way jest metodologicznie slabsze, bo wszystkie klasy sa stale obecne. Do ProtoNet lepiej nie wrzucac CutMix bez jawnego wsparcia dla mieszanych etykiet w epizodach. |
| `deit_tiny` | `0.0 .. 0.1` | `5e-5 .. 1e-4` | `0.05` typowo; `0.1` tylko jesli stabilne | obecnie nieuzywany w kodzie; jesli kiedys dodacie, `cosine` | `20 .. 40` albo do plateau walidacji | nieuzywane w obecnej implementacji | Zalecane: `n_way=5`, `k_shot in {1,5}`, `q_query=10 .. 15`, `train_episodes=100 .. 300`, `val_episodes=100`, `test_episodes=300 .. 600` | tylko `baseline` albo lekki `standard` | Dla ViT few-shot najlepiej zostac przy niskim LR i lekkiej augmentacji. Obecny `q_query=15` jest sensowny i zgodny z ProtoNet literature. |

### Phase 4B: reduced supervised on 10% danych

| Model | Drop path | LR | Weight decay | Scheduler | Epochs | Batch size | Few-shot | Augmentation | Komentarz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `efficientnet_b3` | `0.05 .. 0.1` | `1e-4 .. 3e-4` | `1e-4 .. 5e-4` | `cosine` | `20 .. 40` | `32 .. 64` | n/d | `combined` preferowane, `cutmix_alpha=1.0` OK | Na 10% danych CNN zwykle korzysta z combined aug, ale potrzebuje mniejszego LR niz na full-data phase3. |
| `deit_tiny` | `0.1` | `5e-5 .. 1e-4`, maksymalnie `2e-4` jesli stabilne | `0.05 .. 0.1` | `cosine` | `20 .. 40` | `32 .. 64` | n/d | `combined` preferowane | Dla DeiT na malo danych najpierw obnizylbym LR, dopiero potem ruszal `drop_path` lub `wd`. |

## Najkrotsza lista poprawek, ktore najbardziej warto zrobic pozniej w YAML-ach

1. Dla EfficientNet-B3 dodac nizszy LR w `phase2` i zejsc z `lr=1e-3` w `phase4/reduced_supervised`.
2. W `fewshot` odejsc od `n_way=10` na rzecz `5-way` (lub jasno uzasadnic, ze celem jest "10-way on all classes", a nie klasyczny episodic few-shot).
3. Nie interpretowac `advanced/combined` w `fewshot` jako CutMix, dopoki kod few-shot tego realnie nie wspiera.
4. Dla `custom_cnn` zostawic obecne ustawienia jako baseline, ale jesli model zostanie powiekszony, podniesc `wd`, wydluzyc trening i dopiero wtedy rozwazac `cosine`/`drop_path`.
5. Zdecydowac jawnie, czy EfficientNet-B3 ma byc trenowany "fair" w `224`, czy "natywnie" blizej `300`; to zmienia bardziej preprocesing niz same hiperparametry optymalizacji.
