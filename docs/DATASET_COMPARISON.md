# Открытые датасеты для детектора синтезированной речи

Актуальность проверки: **21 ноября 2026 года**. Задача проекта: бинарная
классификация `REAL / FAKE` по аудио, основной будущий тракт —
`audio → Mel-spectrogram → CNN`. Этот документ не задаёт архитектуру CNN и
параметры Mel-спектрограммы.

## Краткий вывод

Для ВКР нельзя выбрать один набор, который одновременно даёт современность,
русский язык, строгие официальные splits, небольшую загрузку и реалистичный
внешний тест.

- Лучший **официально разделённый базовый корпус** — ASVspoof 5. Он современнее
  ASVspoof 2019, содержит 32 TTS/VC/adversarial attack ID, 1 922 непересекающихся
  диктора и codec-вариативность в eval. Но он только английский и весит
  142,34 GB.
- Лучший **русскоязычный источник для обучения** — RuASD, если самостоятельно
  построить source/speaker/generator-disjoint splits. Он содержит 37
  русскоязычных TTS/voice-cloning систем, 691,68 ч fake и 234,07 ч real. Его
  слабое место — отсутствие официальных train/dev/test splits.
- MLAAD v11 — сильное дополнение по разнообразию генераторов: 1 153,6 ч fake,
  205 TTS-моделей, 54 языка и 50,2 ч русского fake. Однако bona fide нужно
  получать отдельно из M-AILABS, а доступ к файлам MLAAD на Hugging Face сейчас
  требует принятия лицензии и токена.
- ASVspoof 2021 DF удобен как официальный codec-robust **test**, но не как train:
  это evaluation-only набор.
- In-the-Wild нужно сохранить как независимый внешний тест. Он мал, реалистичен
  и не должен участвовать ни в обучении, ни в выборе гиперпараметров.
- Собственные записи 5 людей и их RVC/MuXVS версии — только финальный внешний
  тест. Их нельзя использовать даже для выбора порога.

## Сводная таблица фактов

`UNKNOWN` означает, что точное значение не найдено в официальном источнике или
оригинальной публикации. Размеры — объём официальной загрузки, а не распакованных
данных.

| Датасет | Год / назначение | Файлы: REAL / FAKE | Дикторы | Языки; русский | Генерация | Формат / sample rate | Splits и labels | Объём / доступ |
|---|---|---:|---:|---|---|---|---|---|
| **ASVspoof 5** | 2024 challenge; SDD и SASV | Track 1: **1 004 081** = 188 819 / 815 262; ещё 2 282 enrollment-аудио | **1 922** speaker-disjoint | English; **русского нет** | 32 ID: TTS, VC, 7 adversarial variants | FLAC, mono, 16 kHz | train/dev/eval; все Track 1 labels опубликованы | **142,34 GB**; ODC-By, bona fide CC BY 4.0; прямой Zenodo |
| **ASVspoof 2019 LA** | 2019 logical access anti-spoofing | **121 461** = 12 483 / 108 978 | 107 исходных VCTK speakers | English; нет | 19 attacks: TTS + VC, neural и классические | FLAC, 16-bit, 16 kHz | train 25 380; dev 24 844; eval 71 237; labels есть | **7,64 GB**; ODC-By; Edinburgh DataShare |
| **ASVspoof 2021 DF** | 2021 deepfake detection после compression | **611 829** = 22 617 / 589 212 | **93** | ASVspoof English + VCC language codes E/F/G/M; русского нет | 110 attack labels / 100+ algorithms; TTS + VC; разные vocoder families | FLAC, 16-bit, 16 kHz; 9 codec-условий | evaluation only: progress/eval/hidden; labels опубликованы после challenge | **34,53 GB**; Zenodo указывает ODbL; прямой download |
| **WaveFake** | 2021; различия neural vocoders/TTS | В архиве **0 / 117 985**; real source не распространяется | 2 женских source speakers | English, Japanese; нет | 6 архитектур, 10 sets; neural vocoders + один полный TTS pipeline; **VC нет** | 16-bit PCM WAV; единый sample rate в datasheet не указан | официального split нет; все опубликованные файлы fake | **28,92 GB**, ≈175 ч fake; CC BY-SA 4.0; Zenodo |
| **MLAAD v11** | 2024–2026; multilingual/open-generator generalization | **0 / 534 000** в MLAAD; real берётся отдельно из M-AILABS | точное число UNKNOWN; reference speaker есть с v8 | 54 языка; **Russian: 50,2 ч fake** | **205 TTS models**, 127 architectures; текущий repo также содержит RVC directories | WAV, 16-bit, 22 050 Hz | общего binary split нет; `meta.csv` на language/model | **1 153,6 ч fake**; CC BY-NC 4.0; HF gated/токен |
| **In-the-Wild** | 2022; реалистичная cross-dataset проверка | **31 779** = 19 963 / 11 816 | **58** публичных персон | English; нет | методы конкретных файлов не размечены | WAV, 16 kHz | один evaluation corpus; binary labels внутри архива | **8,16 GB**, 37,9 ч; CC BY-SA 4.0; прямой HF ZIP |
| **SpoofCeleb** | 2024/2025; in-the-wild SDD + SASV | **2 687 292 total**; разбиение REAL/FAKE по файлам protocol, итоговые class totals в статье не приведены | **1 251** | English; нет | **23 contemporary TTS**, включая neural-codec SpeechLM; **VC нет** | формат и sample rate: **UNKNOWN** в статье | train 2 540 421; validation 55 741; eval 91 130; protocols есть | **134,16 GB**; CC BY 4.0; HF manual gated, рассмотрение 2–3 дня |
| **RuASD** | 2026; русский anti-spoofing + robustness | точное число файлов **UNKNOWN** (HF: 100K–1M); 234,07 ч / 691,68 ч | точное число UNKNOWN; metadata содержит `speakers` | **Russian** | **37** Russian-capable TTS/voice-cloning systems; clean + reproducible noise/reverb/codec variants | WAV; sample rate UNKNOWN | HF показывает один `train`; официальных train/dev/test нет; JSON labels есть | **≈250,45 GB**; CC BY-NC-SA 4.0; HF streaming/WebDataset |
| **SpeechFake** | 2025; large-scale multilingual modern attacks | fake: **3 338 508**; real source counts/protocol rows опубликованы, но сами real corpora берутся отдельно | BD 1 363, MD 179 (fake statistics in paper) | 46 языков; наличие Russian в перечне paper **не подтверждено отдельно** | 40 tools в исследовании; release включает 30 open-source: TTS, VC, neural vocoders | mono WAV, 16 kHz | BD/MD train/dev/test/hidden; labels и metadata заявлены | >3 000 ч; ModelScope; условия лицензий неоднородны |

## Подробный разбор

### 1. ASVspoof 5

Официальные источники: [страница ASVspoof](https://www.asvspoof.org/),
[Zenodo record 14498691](https://zenodo.org/records/14498691),
[описание базы](https://arxiv.org/abs/2502.08857),
[официальные baselines](https://github.com/asvspoof-challenge/asvspoof5).

**Назначение и состав.** База построена из English-раздела Multilingual
LibriSpeech. Исходные и сгенерированные данные разделены по дикторам. В Track 1:

| Split | Файлов | bonafide | spoof | Уникальных `SPEAKER_ID` | Attack ID | Codec |
|---|---:|---:|---:|---:|---|---|
| train | 182 357 | 18 797 | 163 560 | 400 | A01–A08 | `-` |
| dev | 140 950 | 31 334 | 109 616 | 785 | A09–A16 | `-` |
| eval | 680 774 | 138 688 | 542 086 | 737 | A17–A32 | `-`, C01–C11 |

Числа выше получены локальным разбором официального protocol archive, а не
перенесены из обзорной таблицы.

**Точные поля Track 1 protocol:**

```text
SPEAKER_ID FLAC_FILE_NAME SPEAKER_GENDER CODEC CODEC_Q CODEC_SEED
ATTACK_TAG ATTACK_LABEL KEY TMP
```

Это 10 полей. В официальном README есть текстовая опечатка «FIVE columns», но
перечень и сами строки содержат 10 значений.

**Файлы protocol, доступные отдельно:**

```text
ASVspoof5.train.tsv
ASVspoof5.dev.track_1.tsv
ASVspoof5.dev.track_2.enroll.tsv
ASVspoof5.dev.track_2.trial.tsv
ASVspoof5.eval.track_1.tsv
ASVspoof5.eval.track_2.enroll.tsv
ASVspoof5.eval.track_2.trial.tsv
ASVspoof5.codec.config.csv
```

Архив `ASVspoof5_protocols.tar.gz` весит 20 669 392 bytes. Аудио разбито на 5
train tar (37,50 GB), 3 dev tar (20,06 GB) и 10 eval tar (84,76 GB). Поэтому
можно скачать конкретную tar-часть, но **нельзя запросить отдельные FLAC внутри
tar с Zenodo**. Практический порядок:

1. скачать только protocol;
2. выбрать допустимые split/attack/speaker IDs;
3. оценить число нужных файлов;
4. скачать минимальное число tar-частей;
5. после распаковки оставить только выбранные FLAC.

**Атаки.** Train A01–A08 — zero-shot neural TTS (Glow-TTS, Grad-TTS,
FastPitch, VITS variants). Dev A09–A12 и A14 — TTS, A13/A15/A16 — VC. Eval
содержит TTS A17/A19/A21/A22/A28/A29, VC A24/A25/A26 и adversarial варианты
A18/A20/A23/A27/A30/A31/A32 поверх TTS/VC. Eval также включает XTTS, YourTTS,
diffusion VC и adversarial Malafide/Malacopula. Это наиболее современный из
официальных ASVspoof-корпусов в сравнении.

**Пригодность.** Отличный train/dev/test дизайн и хороший источник для
Mel-CNN, но полный корпус слишком велик для обычной Colab-сессии. Русского нет.
Нельзя случайно объединять splits: attack IDs и speakers специально сделаны
непересекающимися.

### 2. ASVspoof 2019 LA

Источники: [официальная страница](https://www.asvspoof.org/index2019.html),
[Edinburgh DataShare](https://datashare.ed.ac.uk/items/31074a11-b6f6-4e92-a4ad-07093f8c0c45),
[evaluation plan](https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf),
[original paper](https://arxiv.org/abs/1911.01601).

| Split | Total | bonafide | spoof | Attacks |
|---|---:|---:|---:|---|
| train | 25 380 | 2 580 | 22 800 | A01–A06 |
| dev | 24 844 | 2 548 | 22 296 | A01–A06 |
| eval | 71 237 | 7 355 | 63 882 | A07–A19, включая 2 known anchors |

В A01–A19 присутствуют четыре train TTS + два train VC, а eval содержит 7 TTS
и 6 VC. Есть WaveNet/WaveRNN, neural source-filter, Tacotron-подобные системы,
WORLD/STRAIGHT и классические VC. По меркам 2026 года атаки уже не современные,
но корпус остаётся удобным небольшим baseline с корректными speaker-disjoint
splits.

Официальный DataShare выдаёт LA как единый ZIP 7 640 952 520 bytes; protocol не
предлагается там отдельным небольшим файлом. Он находится внутри ZIP. Labels
всех трёх частей опубликованы. Автоматическая загрузка без аккаунта возможна.

### 3. ASVspoof 2021 DF

Источники: [официальная страница и keys](https://www.asvspoof.org/index2021.html),
[официальный DF Zenodo](https://zenodo.org/records/4835108),
[официальный evaluation package](https://github.com/asvspoof-challenge/2021/tree/main/eval-package),
[summary paper](https://arxiv.org/abs/2210.02437).

Аудио состоит только из evaluation data. Участникам не давали matched train/dev;
для обучения использовались ASVspoof 2019 data. Поэтому использовать DF как
train можно технически после публикации labels, но это разрушает его основное
назначение как независимого benchmark; для ВКР он закреплён за TEST.

Официальный `DF-keys-full.tar.gz` (25 985 925 bytes) доступен отдельно и содержит
611 829 строк. Его 13 полей:

```text
speaker trial codec source attack label trim subset vocoder
task team gender_pair language
```

Реально полученные распределения:

- labels: 22 617 bona fide; 589 212 spoof;
- subsets: eval 533 928; progress 59 325; hidden 18 576;
- sources: ASVspoof 159 696; VCC2018 186 183; VCC2020 265 950;
- speakers: 93;
- codecs: `nocodec`, low/high MP3, low/high M4A, low/high OGG, MP3→M4A,
  OGG→M4A — по 67 981 файлов;
- vocoders: traditional 275 553; neural autoregressive 152 379; neural
  non-autoregressive 120 915; unknown 29 043; waveform concatenation 11 322;
  bona fide 22 617;
- `attack` содержит 110 spoof labels плюс `-` для bona fide.

Это сильный codec/channel test для Mel-CNN. Риск: источник частично наследует
ASVspoof 2019; если ASVspoof 2019 используется в train, результаты нельзя
интерпретировать как полностью независимую проверку source-domain.

### 4. WaveFake

Источники: [официальный GitHub](https://github.com/RUB-SysSec/WaveFake),
[официальный Zenodo](https://zenodo.org/records/5642694),
[original paper](https://arxiv.org/abs/2111.02813),
[official datasheet](https://zenodo.org/records/5270336/files/datasheet.pdf).

Архив содержит только 117 985 fake WAV (≈175 ч). Real reference data не
перераспространяются: это LJSpeech 13 100 английских файлов одного диктора и
JSUT basic5000 5 000 японских файлов одного диктора. Есть MelGAN, large MelGAN,
Parallel WaveGAN, multi/full-band MelGAN, HiFi-GAN, WaveGlow и полный
Conformer+PWG TTS pipeline. Voice Conversion отсутствует.

Плюсы: простая структура и полезный controlled vocoder benchmark. Минусы:
нет официального split, fake-only release, всего два source speakers, старые
генераторы, высокий риск speaker/source leakage. Для основной ВКР это только
дополнительный out-of-domain test, не основной train.

### 5. MLAAD v11

Источники: [официальная страница](https://deepfake-total.com/mlaad),
[author Hugging Face](https://huggingface.co/datasets/mueller91/MLAAD),
[актуальная original paper](https://arxiv.org/abs/2401.09512).

MLAAD постоянно обновляется. Версию необходимо записывать в каждом эксперименте.
По paper v11 от 01.09.2026:

- 534 000 synthetic utterances, 1 153,6 ч;
- 205 TTS models, 127 architecture labels, 54 языка;
- русский: 50,2 ч fake;
- WAV, 16-bit, 22 050 Hz;
- поля `meta.csv`: `path`, `original_file`, `language`,
  `is_original_language`, `duration`, `training_data`, `model_name`,
  `architecture`, `transcript`, `reference_speaker` (последнее с v8).

MLAAD не содержит real audio. Для бинарного обучения авторы требуют соответствующие
M-AILABS originals. Это важно: запись «0 REAL» в таблице не означает, что dataset
непригоден, а означает необходимость второго официального источника.

Текущий список HF без скачивания аудио подтвердил русский каталог и как минимум
23 Russian model directories, включая `RVC`, Edge-TTS, Chatterbox Multilingual,
MOSS-TTS, Qwen3-TTS, OpenAI TTS-1 HD, MMS-TTS-RUS, Bark и XTTS. Однако repo
gated: чтение `meta.csv` возвращает 401 без принятия CC BY-NC 4.0 и HF token.

MLAAD удобен для generator-held-out validation и русского дополнения к train.
Он не подходит для оценки VC как единственный источник: основная методология и
статистика paper описывают TTS, хотя текущий расширенный repo уже содержит RVC.

### 6. In-the-Wild

Источники: [официальная страница](https://deepfake-demo.aisec.fraunhofer.de/in_the_wild),
[author Hugging Face](https://huggingface.co/datasets/mueller91/In-The-Wild),
[original paper](https://www.isca-archive.org/interspeech_2022/muller22_interspeech.html).

Корпус содержит 31 779 English clips 58 политиков и знаменитостей: 19 963 real
(20,7 ч) и 11 816 fake (17,2 ч). Материал найден в открытых online videos, поэтому
присутствуют реальные шумы, монтаж, каналы и compression. Точные generator IDs
не опубликованы.

Официальный author HF даёт прямой ZIP 8 161 489 023 bytes без ручной авторизации;
отдельный маленький labels-файл в корне HF не опубликован. Поэтому metadata
нельзя получить отдельно от полного ZIP. Использовать только как external test.

### 7. SpoofCeleb

Источники: [project/paper](https://arxiv.org/abs/2409.17285),
[author Hugging Face](https://huggingface.co/datasets/jungjee/spoofceleb).

Real source — обработанный VoxCeleb1/TITW, то есть шумные записи из реального
мира. 23 TTS были обучены на этих же условиях. Включены VITS, MQTTS, VALL-E,
Delay, GradTTS, Matcha-TTS, TransformerTTS, BVAE-TTS, neural-codec language
models и разные vocoders. VC авторы намеренно оставили будущей работе.

Splits speaker-disjoint: 1 171 train speakers, 40 validation, 40 evaluation.
Attack split также контролируемый: A01–A10 train, A06/A07/A11–A14 validation,
A15–A23 evaluation. Это отличный train/test дизайн для unseen TTS, но английский
язык, большой объём и отсутствие VC ограничивают пользу для данной ВКР.

HF download manual-gated: требуется institutional email/affiliation, решение
может занять 2–3 дня. Без принятого доступа нельзя автоматически получить даже
protocol из архива. Размер 27 частей — 134 157 296 519 bytes.

### 8. RuASD — добавлен из-за русского языка

Источник: [официальный/авторский Hugging Face](https://huggingface.co/datasets/lab260/RuASD).

RuASD добавлен сверх обязательного списка, потому что это единственный найденный
актуальный набор, прямо созданный для русского anti-spoofing и содержащий real и
fake. На дату проверки card заявляет:

- 37 Russian-capable TTS/voice-cloning systems;
- 691,68 ч fake и 234,07 ч bona fide;
- real из нескольких русских open corpora;
- clean + noise, reverberation и codec perturbations;
- JSON metadata с `sample_id`, `label`, `group`, `subset`, `augmentation`,
  `source_type`, `duration`, `speakers`, `model`, transcript и `ops`.

HF repo — 250 WebDataset tar shards общим размером ≈250,45 GB. Dataset Viewer
позволил прочитать небольшую metadata-выборку без полного скачивания. Точный
полный file count, sample rate и официальный speaker/generator-disjoint split в
card не приведены. Все записи представлены одним HF split `train`.

Перед использованием нужно самостоятельно создать immutable manifest:

1. оставить `group=raw` для первого controlled эксперимента;
2. разделить реальные **source corpora/speakers**, а fake — **model families**;
3. не допускать, чтобы augmented version одного `source_audio` попала в другой
   split;
4. заморозить random seed и списки ID в Git, но не аудио;
5. только после фиксации split считать метрики.

### 9. SpeechFake — добавлен как современный multilingual benchmark

Источники: [ACL 2025 paper](https://aclanthology.org/2025.acl-long.493/),
[официальный GitHub](https://github.com/YMLLG/SpeechFake),
[официальный ModelScope](https://www.modelscope.cn/datasets/inclusionAI/SPEECHFAKE).

SpeechFake включает Bilingual Dataset (English/Chinese) и Multilingual Dataset
на 46 языков. Fake data: BD 1 953 016, BD unseen commercial test 50 000, MD
1 335 492. Использованы TTS, VC/voice-cloning и neural vocoders; среди современных
методов StyleTTS2, OpenVoice, GPT-SoVITS, Fish Speech, CosyVoice, Seed-VC,
Parler-TTS, FireRedTTS. Все clips нормализованы в mono WAV 16 kHz.

Причины не делать его основным выбором сейчас:

- paper не перечисляет все 46 языков текстом, поэтому наличие Russian не
  подтверждено без metadata;
- 10 commercial API subsets не выпускаются из-за лицензий;
- GitHub badge/`LICENSE` говорит CC BY 4.0, тогда как appendix paper указывает
  CC BY-NC 4.0 для основной части и alternative licenses для отдельных методов;
  условия каждого subset нужно проверить перед использованием;
- загрузка огромна, а ModelScope metadata/API менее удобны в Colab, чем HF.

## Что можно скачать автоматически

| Датасет | Metadata без audio | Audio без ручной авторизации | Streaming / partial | Вывод для Colab |
|---|---|---|---|---|
| ASVspoof 5 | **Да**, 20,7 MB protocol | Да, tar parts | Только по tar-частям, не по FLAC | Хорошо для protocol; full corpus тяжёлый |
| ASVspoof 2019 LA | Нет отдельного protocol в DataShare | Да, единый 7,64 GB ZIP | Нет официального streaming | Допустимо, но нужен весь ZIP |
| ASVspoof 2021 DF | **Да**, 26,0 MB keys | Да, 4 части ≈8,6 GB | По четырём tar parts | Хороший test при достаточном диске |
| WaveFake | Datasheet да; per-file index отдельно не найден | Да, единый 28,9 GB ZIP | Нет | Слишком велик для ограниченной Colab |
| MLAAD v11 | Tree виден; `meta.csv` требует token | После принятия лицензии + HF token | HF API; repo file-level | Хорошо после авторизации; не тянуть весь corpus |
| In-the-Wild | Отдельно не найдено | **Да**, 8,16 GB ZIP | Нет | Удобный внешний test |
| SpoofCeleb | Нет без gated archive | Нет, manual approval | Tar split после доступа | Заранее запросить доступ |
| RuASD | **Да**, Dataset Viewer preview | Да | HF streaming/WebDataset shards | Лучший русский источник; фильтровать on-the-fly |
| SpeechFake | GitHub описание да; полный index требует ModelScope | Да по официальной инструкции | ModelScope subsets/files | Только выборочные subsets |

## Оценка пригодности и рисков

| Dataset | Современность атак | TTS | VC | Speaker diversity | Language diversity | Codec/channel | TRAIN | VALIDATION | TEST | Главный leakage-риск |
|---|---|---|---|---|---|---|---|---|---|---|
| ASVspoof 5 | Очень высокая | Да | Да | Очень высокая | Низкая | Высокая в eval | **Да** | **Да** | **Да** | смешение официальных speakers/attacks между splits |
| ASVspoof 2019 LA | Низкая–средняя | Да | Да | Средняя | Низкая | Низкая | Да, baseline | Да | Да | модель учит artefacts VCTK/старых vocoders |
| ASVspoof 2021 DF | Средняя | Да | Да | Средняя | Низкая | **Очень высокая** | Нет для этой ВКР | Нет | **Да** | source overlap с ASVspoof 2019 |
| WaveFake | Низкая | Да | Нет | Очень низкая | Низкая | Низкая | Не основной | Нет | Доп. test | одинаковые тексты/speakers в real и fake; no split |
| MLAAD v11 | **Очень высокая** | **Да** | Ограниченно/current RVC | UNKNOWN | **Очень высокая** | В base нет | **Да, с M-AILABS** | **Да, generator-held-out** | Да | один original/transcript в разных splits; generator family overlap |
| In-the-Wild | Реальные, но unknown | UNKNOWN | UNKNOWN | Средняя | Низкая | **Реальная высокая** | **Нет** | **Нет** | **Внешний** | tuning threshold по test |
| SpoofCeleb | Высокая | **Да** | Нет | **Очень высокая** | Низкая | Реальная высокая | Да | Да | **Да** | VoxCeleb/TITW identity overlap с внешними данными |
| RuASD | **Очень высокая** | **Да** | voice cloning | UNKNOWN | Русский only | Высокая/augment | **Да** | **Да, свой split** | **Да, held-out models** | augmented source/voice/model leakage |
| SpeechFake | **Очень высокая** | Да | Да | Высокая | **Очень высокая** | Низкая после uniform encoding | Да | Да | Да | повтор исходного content/voice между subsets |

Все перечисленные наборы пригодны технически для CNN по Mel-спектрограммам.
Однако это не гарантирует cross-dataset generalization: CNN легко учит sample
rate, codec, silence, amplitude, corpus или generator shortcuts. До извлечения
Mel признаки нужно унифицировать только внутри pipeline, сохраняя отдельные
оценки на clean и codec/channel conditions.

## Три стратегии эксперимента

### A. Минимальная — ограниченная Colab

**TRAIN**

- ASVspoof 2019 LA train;
- небольшой balanced RuASD `raw` subset (например, до 20–40 тыс. файлов),
  получаемый streaming; real/fake 1:1;
- RuASD generator families, source corpora и speakers заранее закрепить за train.

**VALIDATION**

- ASVspoof 2019 LA dev;
- отдельные RuASD source speakers и 3–5 целиком held-out generators.

**TEST**

- ASVspoof 2019 LA eval;
- небольшой заранее зафиксированный стратифицированный ASVspoof 2021 DF subset
  только для инженерной проверки codec robustness. Не называть его официальным
  полным score.

**EXTERNAL TEST**

- In-the-Wild;
- собственные 5 real speakers + RVC/MuXVS только один раз после фиксации модели.

Плюс: помещается в Colab. Минус: основной English baseline старый.

### B. Оптимальная — рекомендация для ВКР

**TRAIN**

- balanced ASVspoof 5 train subset из одной-двух tar-частей: все bona fide в
  выбранных частях и равное число spoof, стратифицированное по A01–A08, speaker
  и gender;
- balanced RuASD `raw` subset 60–100 тыс. файлов с русскими real и современными
  TTS/voice-cloning attacks;
- при наличии HF token — MLAAD Russian subset только для генераторов, которых
  нет в RuASD train, с соответствующим M-AILABS Russian bona fide.

**VALIDATION**

- ASVspoof 5 dev subset с A09–A16 (unseen относительно train);
- RuASD held-out speakers/source corpora и held-out generator families;
- ни один `source_audio`, transcript-original pair или augmentation family не
  пересекается с train.

**TEST**

- ASVspoof 2021 DF: полный набор, если есть 35–45 GB свободного места; иначе
  заранее зафиксированный stratified subset по label/source/codec/vocoder;
- ASVspoof 5 eval subset по A17–A32 и codec C01–C11, если доступны нужные tar.

**EXTERNAL TEST**

- In-the-Wild;
- собственный real/RVC/MuXVS набор 5 дикторов.

Это лучший баланс: современный официальный English benchmark + русский target
domain + codec test + два реально независимых внешних теста.

### C. Расширенная — больше GPU и диска

**TRAIN**

- ASVspoof 5 train full;
- крупный source/speaker/generator-disjoint RuASD raw subset;
- MLAAD v11 Russian и выбранные multilingual generators + M-AILABS real;
- при согласованной лицензии — SpeechFake train subsets;
- SpoofCeleb train, если manual access получен и нужен in-the-wild TTS domain.

**VALIDATION**

- ASVspoof 5 dev full;
- RuASD/MLAAD generator-held-out manifests;
- SpoofCeleb validation.

**TEST**

- ASVspoof 5 eval full;
- ASVspoof 2021 DF full;
- SpoofCeleb eval;
- SpeechFake unseen-method tests.

**EXTERNAL TEST**

- In-the-Wild;
- собственный real/RVC/MuXVS набор.

Расширенная стратегия требует отдельного leakage audit: одни и те же source
corpora, TTS families и speakers могут встречаться в разных больших наборах.

## Ресурсы для оптимальной стратегии

Ниже не оценка времени обучения, а план диска и количества примеров. Точное
число зависит от выбранного manifest и длительности окна.

### Разумный первый manifest

- ASVspoof 5 train: ориентировочно 5–15 тыс. balanced files из одной-двух
  tar-частей; все 37 594 balanced files потребуют скачать полный train (37,5 GB);
- RuASD train: 30–50 тыс. real + столько же fake = 60–100 тыс.;
- validation: 10–20 тыс. ASVspoof 5 dev + 10–20 тыс. RuASD;
- test: 20–45 тыс. ASVspoof 2021 DF stratified subset для итераций;
- внешний In-the-Wild: 31 779 files полностью.

Итого для рабочей итерации: примерно **140–235 тыс. аудиофайлов**. Это диапазон,
а не характеристика исходных datasets.

### Диск

- одна train tar-часть ASVspoof 5: ≈7,5 GB;
- одна dev part: ≈6,6–6,8 GB;
- RuASD: примерно 1 GB на shard; скачивать только нужные shards либо streaming;
- одна ASVspoof 2021 DF part: ≈8,6 GB; весь DF: 34,5 GB;
- In-the-Wild: 8,16 GB archive.

Практический download budget оптимальной итерации: **≈45–70 GB**. Полная
современная оценка с ASVspoof 2021 DF потребует **≈70–100 GB** с учётом
распаковки/кэша. В стандартной Colab это возможно только при аккуратной очистке
временных архивов или внешнем Google Drive/Cloud Storage.

### Подготовленные окна

Если использовать 4-секундные окна 16 kHz mono float32, одно незажатое окно
занимает около 256 KiB. 300 тыс. окон — около 73 GiB только waveforms; сохранение
всех Mel tensors создаст ещё один крупный кэш. Поэтому:

- декодировать FLAC/WAV и ресэмплировать **on-the-fly**;
- нарезать окна детерминированно по manifest/seed;
- на первой итерации ограничить 1–3 окна на файл;
- кэшировать только небольшой validation/test или compact features после того,
  как параметры Mel будут отдельно утверждены;
- большие HF/WebDataset наборы читать streaming;
- manifest, split IDs, counts и checksums хранить в Git, аудио и features — нет.

Разумный первый диапазон — **200–400 тыс. обучающих окон**, если он получается из
speaker/generator-balanced manifest. Это ограничение ресурсов, а не утверждение
об оптимальном качестве. Точное время обучения без выбранных Mel/CNN/batch/GPU
параметров оценивать нельзя.

## Правила против data leakage

1. Сначала фиксировать роли datasets, потом смотреть test metrics.
2. Делить RuASD/MLAAD не случайно по файлам, а по speakers, source corpora,
   generator family и `source_audio`/`original_file` groups.
3. Все augmentations одного original должны принадлежать одному split.
4. Не использовать ASVspoof 5 dev/eval и ASVspoof 2021 DF для обучения.
5. Не подбирать threshold по In-the-Wild или собственным записям.
6. Для каждого результата сохранять version/revision dataset и manifest hash.
7. Отдельно сообщать in-domain, unseen-attack, cross-language, codec и external
   результаты; не сводить их в одну accuracy без объяснения.

## Что реально проверено программно

- скачан и разобран официальный ASVspoof 5 protocol archive (20,7 MB);
- посчитаны точные label/attack/speaker/gender/codec/attack-adaptation
  распределения train/dev/eval;
- скачан и разобран официальный ASVspoof 2021 DF keys archive (26,0 MB);
- посчитаны labels, 93 speakers, 110 spoof attack labels, source, codec,
  vocoder, subset, trim, gender-pair и language-code distributions;
- через HF API проверены объёмы официальных repos: In-the-Wild 8,16 GB,
  SpoofCeleb 134,16 GB, RuASD ≈250,45 GB;
- через HF tree подтверждены Russian directories и современные model names
  MLAAD, но `meta.csv` оказался gated без token;
- через HF Dataset Viewer получены реальные JSON metadata rows RuASD без
  загрузки аудио;
- через Zenodo API проверены размеры ASVspoof 5, ASVspoof 2021 DF и WaveFake.

Воспроизводимый код находится в
`audio_deepfake_detector/research/dataset_metadata.py` и notebook
`audio_deepfake_detector/notebooks/dataset_research.ipynb`.

## Неизвестные или требующие ручного действия сведения

- точное число файлов, speakers и sample rate RuASD не опубликованы в card;
- MLAAD metadata/audio требует принятия gated license и `HF_TOKEN`;
- SpoofCeleb требует manual access с institutional email;
- In-the-Wild labels не вынесены отдельным маленьким файлом;
- SpeechFake требует проверки лицензии конкретных subsets и языка `ru` по
  metadata перед включением;
- для полного ASVspoof 5/2021 теста нужен большой диск; отдельный FLAC из Zenodo
  tar без скачивания соответствующей tar-части получить нельзя.
