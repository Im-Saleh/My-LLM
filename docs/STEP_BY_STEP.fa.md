# ۳) پیاده‌سازی گام‌به‌گام

هر گام یک لایه‌ی مستقل و قابل تست است. ترتیب زیر همان ترتیبی است که پروژه ساخته و
تست شده — می‌توانی همین مسیر را برای خواندن کد هم دنبال کنی.

---

## گام ۰ — بیلد و تست سلامت

```bash
git clone https://github.com/Im-Saleh/My-LLM.git && cd My-LLM
sudo ./scripts/install_deps.sh          # کامپایلر، cmake، OpenGL/X11 برای GUI
./build.sh --run-tests
```

`slm_gradcheck` باید ۲۶/۲۶ را پاس کند: مقایسه‌ی GEMM با مرجع، و **گرادیان عددی
مرکزی** برای همه‌ی عملگرها. این مهم‌ترین شبکه‌ی امنیت پروژه است؛ اگر backward جایی
غلط باشد اینجا لو می‌رود، نه بعد از سه ساعت آموزش.

```
[ OK ] gemm vs reference        max rel err 0.000004
[ OK ] layernorm                max rel err 0.000055
[ OK ] matmul batched 4d        max rel err 0.000142
[ OK ] cross_entropy            max rel err 0.000015
[ OK ] seq_logprob              max rel err 0.000038
[ OK ] mini transformer block   max rel err 0.000161
[ OK ] checkpoint == plain grads max rel err 0.000000
26/26 checks passed
```

---

## گام ۱ — موتور تنسور (`src/core/tensor*.cpp`)

یک autograd معکوس با نوار (tape) ساده:

- هر تنسور یک گره است: `shape` + حافظه‌ی پیوسته + `grad` + والدها + یک closure برای backward.
- `backward()` یک پیمایش post-order معکوس روی DAG انجام می‌دهد (که برای DAG یک ترتیب
  توپولوژیک معتبر است) و **بازگشتی‌پذیر (reentrant)** است — همین ویژگی است که
  gradient checkpointing را ممکن می‌کند.
- عملگرها: `matmul` بچ‌دار، `linear`، `layernorm`، `softmax`، `gelu`، `causal_mask`،
  `embedding`، `cross_entropy`، `seq_logprob`، `logsigmoid`، `transpose/slice/reshape`.
- `gemm.cpp`: کرنل ۴×۱۶ با AVX2+FMA و بلوک‌بندی N/K، انتخاب در زمان اجرا
  (`__builtin_cpu_supports`) تا باینری روی هر CPU x86-64 اجرا شود.
  اندازه‌گیری: **۶۱ GFLOP/s تک‌هسته، ۱۸۰ GFLOP/s روی ۸ هسته**.

> نکته‌ی مهم پیاده‌سازی: بدنه‌ای که به `checkpoint()` می‌دهی باید همه‌چیز را
> **by value** کپچر کند، چون در backward دوباره اجرا می‌شود. کپچر by-reference
> از متغیرهای محلی = دنگلینگ (اولین باگی بود که در همین پروژه گیر افتاد).

بهینه‌سازی‌ای که ۲.۷ برابر سرعت داد: `transpose` اولیه برای هر عنصر `nd` تقسیم صحیح
انجام می‌داد. با جایگزینی با یک شمارنده‌ی odometer + `memcpy` برای بازه‌های پیوسته و
افزودن OpenMP به softmax/layernorm/gelu/cross-entropy:
**۱۶۷۹ → ۴۵۱۸ توکن بر ثانیه** (مدل ۱M) و **۳۶۵ → ۸۳۸** (مدل ۱۱.۵M).

---

## گام ۲ — توکنایزر BPE (`src/tokenizer.cpp`)

- بایت‌محور: هیچ متنی نمی‌تواند خطا بدهد (فارسی، ایموجی، کد).
- پیش‌توکن‌سازی سبک GPT-2: فاصله‌ی ابتدایی به کلمه می‌چسبد، کلاس کاراکتر
  (حرف/رقم/فاصله/علامت) مرز می‌سازد، سقف ۴۸ بایت برای هر قطعه.
- آموزش با شاخص معکوس `pair → {word ids}` تا هر merge به‌جای بازشماری کل، فقط
  کلمه‌های متأثر را به‌روز کند.
- توکن‌های کنترلی: `<|endoftext|> <|user|> <|assistant|> <|system|>`؛ `encode` املای
  آن‌ها را در متن تشخیص می‌دهد (چت و ساخت پرامپت بر همین بنا است).

```bash
slm tokenizer --input data/sample_corpus.txt --out run/tok.slmtok --vocab 2048
# roundtrip: "Hello world! سلام دنیا." -> 25 tokens -> "Hello world! سلام دنیا." [exact]
```

---

## گام ۳ — مدل پایه و پیش‌آموزش (`src/model.cpp`)

```bash
slm pretrain --data data/sample_corpus.txt --tokenizer run/tok.slmtok \
             --out run/base.slm --config configs/slm-demo.conf --steps 500
```

مشاهده‌ی واقعی روی کورپوس نمونه (مدل ۰.۹۴M، ۱۲۰ گام، ۷۷ ثانیه):

```
step 0   loss 6.98  ppl 1076
step 60  loss 1.53  ppl 4.6
step 119 loss 1.22  ppl 3.4      holdout 1.136 (ppl 3.11)
```

سپس تولید با KV-cache (۱۴۰ توکن بر ثانیه روی همین مدل کوچک):

```bash
slm chat --ckpt run/base.slm --tokenizer run/tok.slmtok \
         --prompt '<|user|>what is the capital of France?<|assistant|>'
```

`slm quantize --in run/base.slm --out run/base.q8.slm --dtype q8` هم اندازه‌ی
چک‌پوینت را ~۳.۹ برابر کم می‌کند و خطای RMS نسبی را گزارش می‌دهد.

---

## گام ۴ — زیرساخت مشترک تردها (`src/trainer.cpp`)

هر ترد یک حلقه دارد:

```
while (زنده) {
    اگر Emergency Stop → "halted"، خواب
    اگر غیرفعال شده → "disabled"، خواب
    اگر فاصله از دور قبل < min_interval_s → خواب
    اگر ready() نه → "waiting for data"، خواب
    round():  sync() → آموزش محلی → submit_delta()
}
```

- `sync()`: وزن‌ها را از snapshot منتشرشده می‌گیرد، نقطه‌ی مبنا را ذخیره می‌کند و
  **وضعیت AdamW را ریست می‌کند** تا Δ فقط تابع داده‌ی همین دور باشد.
- `submit_delta()`: تفاضل flat را می‌سازد، در audit log با `samples/steps/loss/delta_norm`
  ثبت می‌کند و به coordinator می‌دهد.

---

## گام ۵ — (الف) ترد یادگیری مداوم (`train_continual.cpp`)

- ورودی‌های کاربر را بافر می‌کند؛ با رسیدن `min_samples` نمونه (یا `max_wait_s` ثانیه)
  یک دور کوچک fine-tune می‌زند.
- `lr = 2e-5` در برابر `1.2e-3` پیش‌آموزش (۶۰ برابر کمتر).
- `replay_percent = 50`: نیمی از بچ‌های هر دور از کورپوس اصلی نمونه‌برداری می‌شود
  (دفاع دوم در برابر فراموشی، در کنار میرایی فیشر در coordinator).
- `last_k_blocks = 2`: فقط دو بلوک آخر + `ln_f` + head حرکت می‌کنند.

---

## گام ۶ — (ب) ترد داده‌ی خودتولید (`train_selfgen.cpp`)

استخراج پرامپت: کورپوس برای الگوی `<|user|> … <|assistant|>` اسکن می‌شود و همان‌ها
دانه‌ی تولید می‌شوند (اگر کورپوس چنین ساختاری نداشت، پنجره‌های تصادفی توکن).

فیلتر کیفیت — هر نمونه باید از هر چهار گیت بگذرد:

| گیت | پیش‌فرض | چه چیزی را می‌گیرد |
|---|---|---|
| `ppl_min` | 1.15 | حلقه‌ی تکراری/دژنره (perplexity بیش از حد کم) |
| `ppl_max` | 30 | خروجی بی‌معنا |
| `max_repeat_ratio` | 0.34 | نسبت ۴-گرام‌های تکراری |
| `min_tokens` | 8 | جواب‌های ناقص |

از بازمانده‌ها فقط `keep_best` نمونه‌ی برتر (بر اساس نزدیکی به مرکز باند perplexity و
کمی تکرار) وارد بافر augmentation می‌شوند. **هر تصمیم** — پذیرش یا رد با دلیل — در
`audit.jsonl` ثبت می‌شود:

```json
{"level":"augment","source":"self-generated","message":"accepted synthetic sample",
 "ppl":"2.09","repeat":"0.00","tokens":"20","text":"People use Go because it is fast, portab..."}
{"level":"filtered","source":"self-generated","message":"dropped: too short (5 tokens)","text":"blue, yellow."}
```

---

## گام ۷ — (ج) ترد بازخورد (DPO) {#ج-ترد-بازخورد-dpo}

**چرا DPO و نه PPO؟** PPO به reward model + value head + rollout + کنترل KL نیاز دارد:
چهار جزء متحرک که هر کدام می‌تواند مدل ۱۰M پارامتری را با چند ده امتیاز ناپایدار کند.
DPO همه‌ی این‌ها را با یک loss بسته‌فرم روی *جفت‌ها* جایگزین می‌کند:

```
L = −log σ( β · [ (log π(y_w|x) − log π_ref(y_w|x)) − (log π(y_l|x) − log π_ref(y_l|x)) ] )
```

- `π_ref` همان snapshot ابتدای دور است. چون مدل محلی در لحظه‌ی `sync()` دقیقاً برابر
  مرجع است، `log π_ref` **یک‌بار و رایگان** قبل از اولین گام محاسبه و کش می‌شود →
  هیچ مدل دومی در حافظه لازم نیست.
- یک جمله‌ی SFT کوچک (`sft_weight = 0.25`) روی پاسخ برنده اضافه می‌شود؛ برای مدل‌های
  کوچک به‌طور محسوسی پایدارتر است.
- اگر برای یک پرامپت فقط *یک* امتیاز وجود داشته باشد (جفت نداریم)، به
  **reward-weighted fine-tuning** برمی‌گردیم: cross-entropy وزن‌دار با امتیاز نرمال‌شده.
- فقط توکن‌های *پاسخ* امتیازدهی می‌شوند (موقعیت‌های پرامپت با `-100` ماسک می‌شوند).

نکته‌ی عملی که در تست کشف شد: برای ساختن جفت باید پرسش تکراری *پاسخ متفاوت* بگیرد،
پس seed نمونه‌برداری هر نوبت تازه می‌شود. با این تغییر:

```
DPO: 3 preference pairs, beta=0.100, 2 steps (session pairs=16 rwft=0)
```

---

## گام ۸ — Coordinator

سند مستقل: **[COORDINATOR.fa.md](COORDINATOR.fa.md)**.

---

## گام ۹ — رابط زنده

```bash
# داشبورد گرافیکی (ImGui)
slm dashboard --ckpt run/base.slm --tokenizer run/tok.slmtok \
              --data data/sample_corpus.txt --config configs/slm-demo.conf

# بدون نمایشگر / روی SSH: داشبورد ترمینالی با همان داده‌ها
slm live --ckpt run/base.slm --tokenizer run/tok.slmtok \
         --data data/sample_corpus.txt --config configs/slm-demo.conf --autopilot
```

پنل‌ها: نمودار زنده‌ی loss سه ترد + holdout (هر کدام با رنگ مستقل، پنجره‌ی زمانی و
مقیاس log)، نقشه‌ی حرارتی توجه لایه‌به‌لایه/هد‌به‌هد با tooltip، بارهای توزیع توکن بعدی،
کنترل on/off هر ترد با نمایش lr و credit و آخرین checkpoint، آمار coordinator با ماتریس
شباهت سه‌گانه، چت با امتیازدهی ۱..۵، لاگ رنگی، و دکمه‌ی سراسری **EMERGENCY STOP**.

`--autopilot` یک «کاربر مصنوعی» است: سؤالی از کورپوس می‌پرسد (پس جواب درست معلوم است)،
پاسخ را با هم‌پوشانی کلمات امتیاز می‌دهد و ۴۵٪ مواقع سؤال قبلی را تکرار می‌کند تا جفت
ترجیحی ساخته شود. برای soak-test و دموی بدون انسان.

---

## گام ۱۰ — بسته‌بندی

```bash
./build.sh --deb          # → dist/slm_0.1.0_amd64.deb
sudo apt install ./dist/slm_0.1.0_amd64.deb
```

`scripts/make_deb.sh` اگر `dpkg-deb` موجود باشد از آن استفاده می‌کند و در غیر این صورت
بسته را دستی با `ar` + `tar` می‌سازد (یک `.deb` دقیقاً همین سه عضو است:
`debian-binary`, `control.tar.gz`, `data.tar.gz`) — یعنی روی توزیع‌های غیر دبیانی هم
می‌توان بسته ساخت.
