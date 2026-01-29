# 01. 核心基礎 (Core Concepts)

本章節探討檢索增強生成 (RAG) 的核心概念、底層架構 Transformer 以及支撐語義檢索的 Embedding 技術。

## 1. 什麼是 RAG (Retrieval-Augmented Generation)

**RAG (Retrieval-Augmented Generation)**，中文稱為「檢索增強生成」，是一種結合了**資訊檢索 (Information Retrieval)** 與 **大型語言模型 (LLM)** 生成能力的技術架構。

### 1.1 定義
RAG 的核心思想是在讓 LLM 回答問題之前，先從外部知識庫（Knowledge Base）中檢索出相關的資訊，並將這些資訊作為「上下文 (Context)」提供給 LLM，讓模型依據這些事實依據來生成答案，而不是僅憑模型訓練時的內部記憶。

### 1.2 解決的核心問題

RAG 主要解決了 LLM 的兩大痛點：

1.  **幻覺問題 (Hallucination)**：
    *   **現象**：LLM 常常會一本正經地胡說八道，生成看似合理但事實錯誤的內容。
    *   **RAG 解決方案**：透過強制模型基於檢索到的真實文檔回答，大幅降低了憑空捏造的可能性。如果檢索不到相關資訊，系統可以設計為回答「不知道」，而非瞎編。

2.  **知識時效性 (Knowledge Cutoff)**：
    *   **現象**：LLM 的知識截止於訓練數據的收集時間（例如 GPT-4 的訓練數據截止於某個年份），無法回答最新的新聞或企業內部動態數據。
    *   **RAG 解決方案**：外部知識庫可以隨時更新（插入新的 PDF、網頁、數據庫記錄），無需重新訓練模型即可讓 AI 掌握最新資訊。

### 1.3 核心流程
1.  **User Query**: 用戶提出問題。
2.  **Retrieve**: 系統將問題向量化，從向量資料庫中檢索相關文檔塊 (Chunks)。
3.  **Augment**: 將檢索到的文檔塊與原始問題組合成新的提示詞 (Prompt)。
4.  **Generate**: LLM 接收增強後的提示詞，生成最終回答。

---

## 2. Transformer 架構

Transformer 是現代 LLM (如 GPT, BERT, Llama) 的基石。理解 Transformer 對於理解 RAG 中的 Embedding 和生成過程至關重要。

### 2.1 Encoder 與 Decoder

Transformer 架構最初由 Google 於 "Attention Is All You Need" 論文中提出，包含兩個主要部分：

*   **Encoder (編碼器)**：
    *   負責理解和提取輸入序列的特徵。
    *   **在 RAG 中的應用**：Embedding 模型（如 BERT, sentence-transformers）通常基於 Encoder 架構。它們將輸入文本轉化為富含語義的向量 (Vector)。
    *   特點：雙向注意力 (Bidirectional Attention)，能同時看到上下文，適合理解語義。

*   **Decoder (解碼器)**：
    *   負責根據特徵生成輸出序列。
    *   **在 RAG 中的應用**：生成式 LLM（如 GPT 系列）通常基於 Decoder-only 架構。它們根據輸入的 Prompt（包含檢索到的 Context）逐字預測下一個 Token。
    *   特點：單向注意力 (Causal Attention)，只能看到當前和之前的詞，適合文本生成。

### 2.2 Self-Attention 機制 (核心思想)

**Self-Attention (自注意力機制)** 是 Transformer 的靈魂。它的作用是計算序列中每個詞與其他所有詞之間的關聯強度。

*   **原理**：對於句子中的每個詞，模型會計算它與句子中其他詞的「權重」。
    *   例如句子：「**銀行**裡的**錢**被偷了」 vs 「河邊的**銀行**風景很好」。
    *   Self-Attention 能讓「銀行」這個詞在第一句中與「錢」高度關聯（理解為金融機構），在第二句中與「河邊」高度關聯（理解為河岸）。
*   **Q, K, V 矩陣**：
    *   **Query (Q)**: 查詢向量，代表「我在找什麼」。
    *   **Key (K)**: 鍵向量，代表「我有什麼特徵」。
    *   **Value (V)**: 值向量，代表「我的實際內容」。
    *   Attention 分數即為 $Q \times K$ 的相似度，最後再乘以 $V$ 得到加權後的表示。

---

## 3. Embedding (向量化)

Embedding 是 RAG 系統中檢索環節的基石，它將人類的語言轉化為機器可計算的數學形式。

### 3.1 什麼是向量化？

**Embedding (詞嵌入/向量化)** 是將文字（詞、句子或段落）映射到一個高維連續向量空間的過程。

*   **特性**：在這個高維空間中，**語義相似**的文本，其對應的向量距離會非常近。
*   **維度**：常見的維度有 768 (BERT base), 1536 (OpenAI text-embedding-3-small), 1024 (Cohere) 等。

### 3.2 如何計算語義相似度？

當文本被轉化為向量後，我們通常使用數學距離來衡量它們的相似程度。最常用的是 **Cosine Similarity (餘弦相似度)**。

#### 餘弦相似度 (Cosine Similarity)
*   **定義**：測量兩個向量在多維空間中夾角的餘弦值。
*   **公式**：
    $$ \text{similarity} = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|} $$
    其中 $A \cdot B$ 是點積，$\|A\|$ 是向量長度。
*   **直觀理解**：
    *   值為 1：兩個向量方向完全相同（語義完全一致）。
    *   值為 0：兩個向量正交（語義無關）。
    *   值為 -1：兩個向量方向相反（語義完全相反）。

在 RAG 檢索時，我們將用戶的 Query 轉為向量，計算它與資料庫中所有 Document 向量的餘弦相似度，取分數最高的 Top-K 個文檔作為上下文。
