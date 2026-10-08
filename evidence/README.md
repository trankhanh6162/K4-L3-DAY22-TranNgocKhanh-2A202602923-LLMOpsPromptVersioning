# Evidence - Day 22

## Thong tin hoc vien

- Ho va ten: Tran Ngoc Khanh
- MSSV: `2A202602923`
- Repository: `K4-L3-DAY22-TranNgocKhanh-2A202602923-LLMOpsPromptVersioning`

## LangSmith Project

- Project: `day22-lab`
- URL: https://smith.langchain.com/o/cd252b5f-add9-424d-8950-a2991bdb2464/projects/p/92b728a6-c994-4645-99c0-2d34e9058b76
- Evidence: 1,008 traces tren giao dien, gom ca `rag-query` va `ab-rag-query`.

## Phan tich Prompt V1 va V2

Hai prompt deu dat faithfulness tren 0.9, vuot muc tieu 0.8. V2 dat faithfulness cao hon
(0.9570 so voi 0.9299), cho thay chi dan doc ky context va khong suy doan giup cau tra
loi bam sat tai lieu hon.

V1 dat answer relevancy cao hon (0.9151 so voi 0.9003). Phong cach ngan gon 2-4 cau
giup cau tra loi tap trung truc tiep hon vao cau hoi. Hai prompt co context recall 1.0 va
context precision 0.9450 vi dung chung retriever, chunking va top-k=3; prompt chi tac dong
den buoc sinh cau tra loi, khong thay doi cac doan context duoc truy xuat.
