# OCR "4 hours" wrong-answer fix – summary

## What’s going on

1. **New code is deployed**  
   - Response includes `ocr_build: "v2-english-page-512tokens"`.  
   - Logs show: `OCR using English pipeline (recognize_english_page + max_new_tokens=512)`.

2. **Pipeline in use**  
   - For `lang=english`, the API calls `recognize_english_page(image_path)`.

3. **Why you still got "4 hours"**  
   - On Railway, region detection for this image either fails or returns no usable crops.  
   - So the code falls back to **full-page** OCR: one TrOCR call on the **entire** image.  
   - TrOCR (handwritten) is meant for **lines/small patches**, not full forms.  
   - On a full QSL card it often **hallucinates** a short phrase like "4 hours" instead of the real text.

4. **Root cause**  
   - Feeding the **whole page** to TrOCR in one go causes bad output.  
   - Fix: when that fallback runs, **don’t** send the full image once; split the page into **horizontal strips**, run TrOCR on each strip, then concatenate.

---

## Change made

**File: `english_ocr_pipeline.py`**

- **New helper:** `_recognize_english_fullpage_by_strips(image_path, num_strips=10)`  
  - Preprocesses the page once (grayscale, deskew).  
  - Splits it into **10 horizontal strips**.  
  - Runs TrOCR on each strip (each strip is a smaller, line-like region).  
  - Joins the results in order.  
  - If something fails, it falls back to the old single full-page call.

- **Fallbacks now use strips**  
  - Whenever we would have called `recognize_english(path)` for a full page (detection failed, or no regions, or no text from regions), we now call `_recognize_english_fullpage_by_strips(path)` instead.

So whenever the pipeline would have done “one big TrOCR call on the full image”, it now does “10 smaller TrOCR calls on strips” and concatenates. That should remove the "4 hours" hallucination and give real content from the QSL card.

---

## What you do next

1. **Commit and push** the updated `english_ocr_pipeline.py`.
2. **Redeploy** on Railway.
3. **Test again** with the same image; you should see real extracted text (callsigns, date, remarks, address, etc.) instead of "4 hours".

---

## Quick reference

| Item | Status |
|------|--------|
| New API code deployed | Yes (`ocr_build` in response) |
| English pipeline used | Yes (log: recognize_english_page + max_new_tokens=512) |
| Wrong output | TrOCR on full page → hallucination |
| Fix | Full-page fallback now uses strip-based OCR (10 strips) |
