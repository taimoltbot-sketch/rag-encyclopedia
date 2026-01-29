# RAG 知識百科 (RAG Knowledge Wiki)

本知識庫旨在系統化地整理 **檢索增強生成 (Retrieval-Augmented Generation)** 的核心技術、實作細節及進階應用。內容涵蓋從基礎概念到工程落地的全方位知識。

## 目錄 (Table of Contents)

### 1. [核心基礎 (Core Concepts)](01-Core-Concepts.md)
*   **RAG 定義**: 解決幻覺與時效性問題。
*   **Transformer**: Encoder/Decoder 架構與 Self-Attention 機制。
*   **Embedding**: 向量化原理與餘弦相似度計算。

### 2. [數據處理 (Data Processing)](02-Data-Processing.md)
*   **Chunking 策略**: 固定長度、語義切分、Markdown 結構化切分。
*   **複雜文檔處理**: PDF 雙欄排版、表格摘要與 OCR。
*   **工具介紹**: LayoutLM, PDFPlumber, Unstructured.io。

### 3. [索引與檢索優化 (Retrieval Optimization)](03-Retrieval-Optimization.md)
*   **Hybrid Search**: 結合 BM25 (關鍵字) 與 Dense Vector (語義)。
*   **Re-ranking**: Cross-Encoder 原理與檢索漏斗 (Recall vs Precision)。
*   **Query Rewrite**: HyDE (假設性文檔嵌入) 與查詢改寫。

### 4. [進階 RAG 模式 (Advanced Patterns)](04-Advanced-Patterns.md)
*   **RAG-Fusion**: 多查詢生成與 RRF 排序融合。
*   **Self-RAG**: Retrieve -> Generate -> Critique 自我反思機制。
*   **GraphRAG**: 知識圖譜與多跳躍推理 (Multi-hop Reasoning)。

### 5. [評測與工程 (Evaluation & Ops)](05-Evaluation-Ops.md)
*   **RAGAS 框架**: Faithfulness, Answer Relevance, Context Precision 指標。
*   **幻覺處理**: 檢測與緩解策略 (CoT, 引用檢查)。
*   **模型微調**: LoRA 技術與 RAG vs SFT 的選擇策略。

### 6. [面試與職涯 (Career & Interview)](06-Career-Interview.md)
*   **常見面試題**: 檢索優化、向量資料庫選型、RAG vs Long Context。
*   **職涯發展**: AI 工程師技能樹與學習路徑。

### 7. [LangGraph 與 Agentic RAG](07-LangGraph-Agentic-RAG.md)
*   **Stateful Agent**: 取代 AgentExecutor，引入狀態機與循環流程。
*   **LangGraph 核心**: StateGraph, Nodes, Edges, Checkpointer。
*   **Agentic Pattern**: 構建具備自我修正能力的 RAG 系統 (Retrieval -> Grade -> Generate)。

---

## 參考來源 (References)

*   [LangChain v0.3 Docs](https://python.langchain.com/v0.3/docs/introduction/)
*   [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
*   [LangSmith Tracing](https://docs.smith.langchain.com/)

---

*建立時間: 2024*
*最後更新: 2026*
*維護者: RAG 學習小組*
