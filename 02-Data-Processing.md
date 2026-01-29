# 02. 數據處理 (Data Processing)

數據處理是 RAG 系統成敗的關鍵。「Garbage In, Garbage Out」，如果輸入的文本切分不當或解析錯誤，再強的模型也無法挽救檢索品質。

## 1. Chunking (切分) 策略

Chunking 是將長文檔分割成較小的片段 (Chunks) 以適應 Embedding 模型和 LLM 上下文窗口的過程。

### 1.1 固定長度切分 (Fixed-size Chunking)
最基礎的方法，設定一個固定的字符數或 Token 數進行切分。
*   **優點**：簡單、計算開銷低。
*   **缺點**：容易切斷語義。例如將一句話切成兩半，導致上下文丟失。
*   **改進**：通常會設定 `chunk_overlap` (重疊區間)，例如每塊 500 tokens，重疊 50 tokens，以保留邊界語義。

### 1.2 語義切分 (Semantic Chunking)
基於內容的語義變化來進行切分，而不是硬性規定長度。
*   **原理**：利用 Embedding 模型計算句子間的相似度。當相鄰句子的相似度低於某個閾值時，視為語義轉折點，進行切分。
*   **優點**：保持語義完整性，檢索精準度高。
*   **缺點**：計算成本較高，需要對全文進行 Embedding 計算。

### 1.3 結構化切分 (Structured Chunking / Markdown Split)
利用文檔原本的結構（標題、段落、列表）進行切分。
*   **方法**：識別 Markdown 的 `# Header` 或 HTML 的 `<h1>`, `<div>` 標籤。
*   **優點**：天然符合人類閱讀邏輯，能保留層級關係（例如將 H1 標題作為 Metadata 附加到該章節的所有 Chunks 中）。

### 範例代碼 (Python/LangChain)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

# 1. 結構化切分 (Markdown)
markdown_document = "# RAG 介紹\n\n## 定義\nRAG 是一種...\n\n## 優點\n解決幻覺..."
headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_splits = md_splitter.split_text(markdown_document)

# 2. 遞歸字符切分 (作為 fallback 或二次切分)
# 優先嘗試用段落符、換行符切分，最後才用字符
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)
final_splits = text_splitter.split_documents(md_splits)

print(f"切分後數量: {len(final_splits)}")
# 每個 Chunk 都會保留 Header 作為 metadata
print(final_splits[0].metadata) 
# 輸出: {'Header 1': 'RAG 介紹', 'Header 2': '定義'}
```

---

## 2. PDF 與複雜文檔處理

真實世界的文檔（如 PDF）通常包含複雜的排版，直接轉 Text 會丟失大量資訊。

### 2.1 雙欄排版 (Dual Column) 問題
*   **問題**：PDF 解析器若按「從左到右」讀取，會將左欄的一行和右欄的一行拼在一起，導致語句錯亂。
*   **解決**：使用支援版面分析 (Layout Analysis) 的工具，識別區塊座標，按區塊順序讀取。

### 2.2 表格處理 (Table Processing)
表格是 RAG 的噩夢。簡單的 `text-extraction` 會將表格變成一串無意義的數字和文字。

**策略**：
1.  **轉 Markdown/HTML**：保留表格結構。許多 LLM 能很好地理解 Markdown 表格。
2.  **表格摘要 (Table Summary)**：
    *   將表格單獨提取出來。
    *   送給 LLM 生成一段文字摘要（例如：「這張表顯示了 2023 年 Q1-Q4 的營收，其中 Q3 增長最高...」）。
    *   對「摘要」進行 Embedding 檢索，檢索到時返回原始表格內容。
3.  **OCR 識別**：對於圖片格式的表格，必須使用 OCR。

### 2.3 版面分析工具

*   **PDFPlumber**: 
    *   Python 庫，適合提取結構化較好的 PDF 中的文字和表格。
    *   優點：輕量、易用。
    *   缺點：對掃描件、複雜排版無力。

*   **LayoutLM (Microsoft)**:
    *   多模態模型，同時利用文字 (Text) 和 視覺 (Image/Layout) 資訊。
    *   能識別文檔中的「標題」、「段落」、「表格」、「圖片」、「頁眉頁腳」。
    *   **流程**：PDF -> 圖片 -> LayoutLM 識別區域 -> OCR 提取文字 -> 按閱讀順序重組。

*   **Unstructured.io**:
    *   強大的開源 ETL 工具，整合了多種策略（包括 Tesseract OCR, LayoutLM 等）來處理各種格式 (PDF, PPT, Word)。
    *   支援 "Hi-Res" 模式，能精準切割表格和圖片。
