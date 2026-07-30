# ۱) نمای کلی معماری سیستم

> این سند بخش اول از سه سند فنی است:
> **ARCHITECTURE** (همین فایل) ← **[COORDINATOR](COORDINATOR.fa.md)** ← **[STEP_BY_STEP](STEP_BY_STEP.fa.md)**

## ۱.۱ نمودار کلی

```
                        ┌──────────────────────── GUI (Dear ImGui) ───────────────────────┐
                        │  loss ۳ خط + holdout │ attention heatmap │ توزیع توکن بعدی      │
                        │  کنترل هر ترد (on/off)│ آمار coordinator  │ Emergency Stop      │
                        │  چت + امتیازدهی ۱..۵  │ audit log زنده                          │
                        └───────▲───────────────────────────────────────────┬─────────────┘
                                │ فقط خواندن (Telemetry)                    │ فقط نوشتن
                                │                                          │ (InteractionHub)
   ┌────────────────────────────┴──────────────────────────────────────────▼─────────────┐
   │                                   Telemetry                InteractionHub           │
   │        ring-buffer loss / state هر ترد / audit JSONL        متن کاربر + امتیازها     │
   └───────▲──────────────▲──────────────▲────────────────────────┬──────────┬───────────┘
           │              │              │                        │          │
   ┌───────┴──────┐ ┌─────┴────────┐ ┌───┴──────────┐             │          │
   │ (الف)        │ │ (ب)          │ │ (ج)          │◄────────────┘          │
   │ Continual    │ │ Self-Gen     │ │ Feedback     │  امتیازها               │
   │ Thread       │ │ Thread       │ │ Thread (DPO) │                        │
   │ replica مدل  │ │ replica مدل  │ │ replica مدل  │                        │
   └───────┬──────┘ └─────┬────────┘ └───┬──────────┘                        │
           │ Δθ₁          │ Δθ₂          │ Δθ₃                               │
           └──────────────┴──────────────┴───────────┐                       │
                                                     ▼                       │
   ┌─────────────────────────────────────────────────────────────────────┐   │
   │                        COORDINATOR (تک‌نویسنده)                      │   │
   │  1 Fisher damping → 2 trust-region → 3 TIES trim → 4 PCGrad         │   │
   │  → 5 sign-election + میانگین وزنی → 6 rate-limit                    │   │
   │  → line-search روی α∈{1,½,¼} با گیت holdout(loss + entropy)         │   │
   │  → accept: publish اتمیک   |   reject: rollback خودکار              │   │
   └───────────────────────────────┬─────────────────────────────────────┘   │
                                   │ ParamStorePtr (RCU / atomic swap)       │
                                   ▼                                         │
                    ┌───────────────────────────────┐    متن کاربر           │
                    │ ChatEngine (ترد جدا، inference)├────────────────────────┘
                    │ KV-cache، attention capture   │
                    └───────────────────────────────┘
```

## ۱.۲ لایه‌های کد

| لایه | فایل‌ها | نقش |
|---|---|---|
| موتور تنسور | `core/tensor.h` (façade)، `core/tensor_native.cpp`، `core/tensor_torch.cpp`، `core/gemm.*` | تنسور + autograd معکوس، GEMM با AVX2/FMA و OpenMP، gradient checkpointing |
| بهینه‌سازی | `core/optim.*` | AdamW + clip + زمان‌بندی cosine با warmup |
| ذخیره‌سازی | `core/serialize.*` | فرمت `.slm` با fp32/fp16/int8(گروهی) |
| داده | `core/dataset.*`، `tokenizer.*` | BPE بایت‌محور، بچ‌گیری، holdout ثابت |
| مدل | `model.h/.cpp` | GPT decoder-only، freeze جزئی، KV-cache، capture توجه |
| وزن‌ها | `core/params.*` | `ParamStore` غیرقابل‌تغییر + فضای flat برای coordinator |
| خودآموزی | `trainer.*`، `train_continual.*`، `train_selfgen.*`، `train_feedback.*` | سه مکانیزم یادگیری |
| هماهنگی | `coordinator.*` | ادغام و گیت‌کردن آپدیت‌ها (سند دوم) |
| رصد | `telemetry.*` | سری‌های زمانی، وضعیت، audit JSONL |
| رابط | `gui.cpp` (ImGui)، `gui_terminal.cpp` (ANSI)، `chat.*`، `app.cpp` | داشبورد و چت |

## ۱.۳ مدل پایه

```
ids ──embedding(tok_emb) + pos_emb[offset..offset+T]
     └─► N × Block:
            x + Attn( LN(x) )              ← pre-LN، causal، H هد، KV-cache اختیاری
            x + MLP( LN(x) )               ← 4× با GELU (tanh approx)
     └─► LN_f ──► logits = x · tok_embᵀ    ← weight tying (پیش‌فرض)
```

- هر بلوک می‌تواند داخل `checkpoint(fn, x)` اجرا شود: در forward بدون گراف محاسبه می‌شود و در
  backward دوباره اجرا می‌شود. حافظه‌ی فعال‌سازی از `O(L)` به `O(1)` بلوک می‌رسد
  (اندازه‌گیری‌شده: از ۴۳۲MiB به ۹۰MiB برای batch=8, ctx=256, dim=384).
- **freeze جزئی**: `FreezePolicy` مشخص می‌کند فقط `last_k_blocks` آخر + `ln_f` + head آموزش ببینند.
  این همان چیزی است که اجازه می‌دهد سه ترد هم‌زمان روی ۲GB VRAM کار کنند (وضعیت AdamW
  فقط برای پارامترهای trainable ساخته می‌شود).

## ۱.۴ چرا سه replica جدا؟

هر ترد یک نمونه‌ی کامل `GPT` دارد که از snapshot منتشرشده مقدار می‌گیرد. دلیل:

1. **هیچ ترد آموزشی هرگز روی وزن‌های زنده نمی‌نویسد** → مسابقه‌ی داده (race) در سطح وزن ممکن نیست.
2. آپدیت هر منبع به شکل «پیشنهاد» (`Δθ`) قابل بازرسی، مقیاس‌پذیر و قابل رد کردن است.
3. inference (چت) هم‌زمان با آموزش، از نسخه‌ی منتشرشده می‌خواند و هیچ‌وقت وزنِ نیم‌آپدیت‌شده نمی‌بیند.

هزینه‌ی حافظه با freeze کنترل می‌شود: برای مدل ۶.۴M پارامتری دمو، هر replica ≈ ۲۵MB وزن +
وضعیت AdamW فقط برای ~۲.۵M پارامتر trainable.

## ۱.۵ همزمانی و قفل‌ها

| منبع مشترک | مکانیزم | توضیح |
|---|---|---|
| وزن‌های منتشرشده | `std::shared_mutex` + `shared_ptr<const ParamStore>` (RCU) | نوشتن = swap اشاره‌گر؛ خواننده‌ها نسخه‌ی خودشان را زنده نگه می‌دارند |
| صف پیشنهادها | `mutex` + `condition_variable` | coordinator با پنجره‌ی `merge_window_s` منتظر بقیه می‌ماند |
| آمار/تلمتری | چند `mutex` کوچک مستقل | GUI هرگز روی I/O یا محاسبه بلاک نمی‌شود (لاگ و ذخیره‌سازی بیرون از قفل انجام می‌شود) |
| ورودی کاربر و امتیازها | `mutex` در `InteractionHub` | صف‌های محدود (bounded) |
| Emergency Stop | `std::atomic<bool>` | در حلقه‌ی هر ترد، داخل line-search و حتی داخل callback تولید توکن بررسی می‌شود |

## ۱.۶ محدودیت سخت‌افزار هدف (۱۶GB RAM / ۲GB VRAM)

`slm info --config <conf>` بودجه‌ی حافظه را چاپ می‌کند. خروجی واقعی برای GPT-2 small شکل:

```
weights f32 496MiB | f16 248MiB | q8 132MiB
gradients f32 496MiB | AdamW 992MiB | مجموع آموزش کامل f32 ≈ 2GiB
```

نتیجه‌گیری صادقانه:

- **۱۶GB RAM**: آموزش کامل تا ~۱۲۴M پارامتر عملی است (کندی CPU مسئله است، نه حافظه).
- **۲GB VRAM**: آموزش کامل ۱۲۴M جا نمی‌شود؛ اما آموزش *جزئی* (۲ بلوک آخر + head) با
  `grad_checkpointing=true` و micro-batch=1 حدود ۰.۷GB می‌شود و جا می‌شود. دقیقاً همان
  حالتی که سه ترد خودآموزی در آن کار می‌کنند.
- ۵۰۰M پارامتر روی ۲GB VRAM حتی با int8 برای *آموزش* واقع‌بینانه نیست؛ برای *inference* با
  وزن int8 (≈۵۳۰MB) بله.
- بکند libtorch (`--libtorch`) روی CPU حدود **۴ برابر** سریع‌تر از موتور بومی است
  (۳۴۸۱ در مقابل ۸۳۸ توکن بر ثانیه، مدل ۱۱.۵M، batch=8، ctx=256، ۸ هسته) و CUDA هم دارد.
