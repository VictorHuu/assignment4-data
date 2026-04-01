## 2 Filtering Common Crawl
### 2.1 Looking at the data
#### (a)
- http://0371rykj.com/ipfhsb/34.html
- Limited Access due to being marked as dangerous site
- It's the main page of 上海林頻儀器股份有限公司 which is used to advertise and seek commercial cooperation.
#### (b)
- pornographic ads and messy SEO information
- Usage and sales information about a thermo controller brand of a Chinese company.
#### (c)
- Supply Chain Management ,Business Management.

- Education.
#### (d)

### 2.2 HTML to text conversion
#### (a)
```python
HTTP_SPLIT_RE = re.compile(br"\r?\n\r?\n", re.MULTILINE)
...
        if payload.startswith(b"HTTP/"):
            parts = HTTP_SPLIT_RE.split(payload, maxsplit=1)
            if len(parts) == 2:
                payload = parts[1]
```
There are HTTP header in some of the so-called html_bytes,so you should strip them then.  
#### (b) 
- My extraction and the WET extraction are similar in that both preserve much of the visible text from the webpage, including navigation menus, repeated links, and other boilerplate. 

- Across these samples, my WARC-based extraction averaged 273.6 words and the WET extraction averaged 223.0 words. Overall, the WET extraction seems slightly cleaner, but neither method perfectly isolates only the main page content.

### 2.3 Language Identification
#### (a)
You should strip all the ```Enter``` with whitespace since models like fastText assume each input is a single-line sample. Newlines can be interpreted as separators or disrupt tokenization, causing a mismatch with training data. So we strip or replace them to keep input consistent with the training format.
#### (b)
- If the language identification step is inaccurate, the training set may include the wrong languages or exclude useful in-language data, which can hurt fluency, increase code-switching, and degrade model behavior on the target language. It can also introduce systematic bias if short, noisy, or mixed-language pages are misclassified more often than cleaner text. 

- In a higher-stakes setting, I would mitigate this by combining language ID with additional safeguards such as confidence thresholds, manual audits on sampled data, secondary classifiers or ensemble checks, and monitoring performance separately across languages and dialects. I would also treat low-confidence or mixed-language documents conservatively rather than relying on a single classifier decision.

#### (c)

- For example, a Polish-heavy multilingual page was labeled as English, a mixed Russian/Ukrainian e-commerce page was labeled as English, and one garbled/encoding-corrupted page was incorrectly labeled as Chinese. 

- About **40%** of the sampled documents were predicted to be English (8 of 20), though some of those English predictions were low-information redirect or menu-heavy pages rather than rich content. 

- Based on the sample, low-confidence predictions were much more likely to be weak, corrupted, or mixed-language text, so a conservative confidence threshold around **0.7** seems like a reasonable choice for filtering.

### 2.4 Personal identifiable information
#### 4.
- Naïve PII masking can destroy useful structure in the data: for example, replacing every email, phone number, or IP address with the same token may remove distinctions that matter for syntax, formatting, debugging, cybersecurity, or instruction-following tasks. It can also create distribution shift, where the model over-learns artificial mask tokens and performs poorly on naturally occurring patterns or on contexts where such strings are benign and necessary. 

- To mitigate this, I would use high-precision detectors, preserve coarse type information and local formatting when possible, audit false positives and false negatives on real samples, and keep separate policies for different domains rather than applying one uniform masking rule everywhere
#### 5.
- We inspected 20 randomly sampled documents where at least one PII replacement was made. The email masking was generally accurate, but the phone number regex produced false positives on structured numeric strings such as **repeated contact numbers, product identifiers, or formatted text in navigation-heavy pages**. We also observed false negatives for **non-U.S. or irregular phone formats** (e.g., international numbers or partially formatted numbers) that were not captured by the regex. 
- Overall, while the regex-based approach is effective for standard patterns, it is brittle on noisy web text and fails to generalize well to diverse real-world formats.
### 2.5 Harmful content

#### 3.
- Applying harmful-content filters naively can remove **legitimate educational, medical, legal, journalistic, or safety-critical discussion that contains flagged terms** but is not itself harmful. This can bias the model toward being less capable of recognizing, analyzing, or responding appropriately to harmful language in context, and may disproportionately suppress dialectal or minority-language text if the classifiers are miscalibrated. 
- To mitigate this, I would calibrate thresholds on domain-specific validation data, use separate policies for classification versus outright removal, and manually audit false positives and false negatives across different domains and demographic varieties.
#### 4.
- In a 20-example manual inspection, the harmful-content classifiers correctly flagged clearly explicit pornographic pages, but they also produced false positives on benign pages such as product listings, diet-plan pages, and noisy non-text inputs like PDF/binary dumps.

- Across the full WARC-derived dataset, about **1.04%** of documents were predicted as harmful. 

- The main failure mode was not missing obvious harmful content, but overreacting to boilerplate-heavy, repetitive, or corrupted text, so I would use conservative confidence thresholds around **0.7** for both NSFW and toxic-speech filtering. Better upstream text cleaning would likely reduce these false positives substantially.

### 2.6 Quality Rules
#### 2.
- We ran the rule-based Gopher quality filter on text extracted from the WARC files and found that about 70.8% of documents passed the filter. In a manual review of 20 sampled examples, agreement with the rule-based filter was 70%, with 6 disagreements. 

- The filter made false positives on pages that were long enough to satisfy the surface heuristics but were still low-value for language modeling, such as **placeholder pages, e-commerce navigation pages, law-firm service listings, and issue-tracker index pages**. 

- It also made false negatives on some informative pages, including **a substantive FAQ-style accessibility page and a noisy but still useful news page**, showing that simple heuristics can miss semantically valuable text while allowing some boilerplate-heavy pages through.

## 2.7 Quality Classifier
The quality classifier is available at [hf-repo](https://huggingface.co/rahulrao493/cs336-data-collection)

Collect inputs: one positive WARC (Wikipedia-linked pages) and one negative WARC (Common Crawl pages). The pipeline has defaults for both. 

Extract + clean text from WARC responses:

- positives use stricter filtering (length, English ID, Gopher quality),

- negatives use lighter filtering (length + optional English, no Gopher) to preserve low-quality signals. 

- Build supervised training file (quality_train.txt) in fastText format with ```__label__hq``` and ```__label__lq```. 

- Balance classes by truncating to the smaller side, then train fastText and save ```quality_classifier.bin```. 

- Run inference with optional threshold control (infer --threshold ...) or call runtime ```classify_quality``` from the helper module.

