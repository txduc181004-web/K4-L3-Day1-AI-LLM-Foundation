# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature tăng dần từ 0.0 lên 1.5, phản hồi của model sẽ đa dạng và khó đoán hơn. Ở temperature thấp, câu trả lời thường ổn định và nhất quán. Ở temperature cao hơn, model thường sáng tạo hơn nhưng khả năng xuất hiện nhiều thông tin không nhất quán hoặc kém phù hợp cũng tăng .

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ chọn temperature khoảng 0.2–0.3 cho chatbot hỗ trợ khách hàng. Vì mức này giúp câu trả lời tương đối ổn định, nhất quán và ít tạo ra thông tin không chính xác, đồng thời vẫn đủ linh hoạt để diễn đạt tự nhiên.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Workload mỗi ngày là 10.000 × 3 × 350 = 10.500.000 token đầu ra. Theo mức giá output thông thường của GPT-4o, GPT-4o-mini hay GPT-4o có chi phí cao hơn khoảng 5 lần GPT-4o-mini cho cùng lượng output. GPT-4o nên sử dụng khi cần chất lượng suy luận và độ chính xác cao, ví dụ xử lý các yêu cầu phức tạp; trong khi GPT-4o-mini phù hợp với các tác vụ đơn giản, số lượng request lớn như FAQ, phân loại yêu cầu hoặc chatbot hỗ trợ cơ bản.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> *Hai phản hồi thể hiện rõ sự khác nhau về độ dài, từ vựng và cách đưa ra ví dụ. Với system prompt dành cho giáo viên tiểu học, model thường sử dụng câu ngắn, từ ngữ đơn giản và các ví dụ gần gũi để trẻ 8 tuổi dễ hiểu. Với system prompt dành cho chuyên gia tài chính, model sử dụng nhiều thuật ngữ hàn lâm, kỹ thuật . Điều này cho thấy system prompt có thể định hướng persona, mức độ chuyên sâu, cách diễn đạt và phong cách trả lời của model.*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với một đoạn văn tiếng Việt khoảng 100 từ, giả sử count_tokens cho kết quả khoảng 150 token, trong khi cách ước lượng số từ / 0.75 cho khoảng 133 token. Chênh lệch khoảng 12–13%, cho thấy cách đếm từ chỉ là một ước lượng thô và không phản ánh chính xác cách model token hóa văn bản. Tiếng Việt thường tốn nhiều token hơn tiếng Anh cùng độ dài vì cách tokenizer xử lý dấu tiếng Việt, các âm tiết và cách biểu diễn Unicode có thể khiến một từ hoặc âm tiết được chia thành nhiều token hơn.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi model cần tạo ra phản hồi dài hoặc mất nhiều thời gian xử lý. ì người dùng có thể nhìn thấy kết quả từng phần ngay khi model bắt đầu sinh output thay vì phải chờ toàn bộ câu trả lời hoàn tất. Điều này giúp giảm cảm giác chờ đợi và tạo trải nghiệm tương tác tốt hơn, đặc biệt với chatbot và các ứng dụng AI hội thoại. Ngược lại, non-streaming phù hợp với các tác vụ cần nhận toàn bộ kết quả một lần, chẳng hạn như xử lý dữ liệu phía backend, gọi API để lấy một JSON hoàn chỉnh hoặc khi ứng dụng không cần hiển thị kết quả theo thời gian thực.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giúp giảm áp lực lên API khi hệ thống đang quá tải bằng cách tăng dần thời gian chờ giữa các lần retry. Nếu sử dụng delay cố định 1 giây, hàng nghìn client có thể retry cùng một thời điểm, tạo ra một đợt request mới và khiến tình trạng quá tải trở nên nghiêm trọng hơn. Exponential backoff giúp phân tán các lần retry theo thời gian, cho server cơ hội phục hồi và giảm hiện tượng retry storm.
---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Tôi chọn persona trợ lý học tập AI bằng tiếng Việt, hỗ trợ người dùng hiểu các khái niệm về lập trình và AI. System prompt: "Bạn là một trợ lý học tập AI chuyên về Python, Machine Learning và AI. Hãy giải thích kiến thức bằng tiếng Việt, rõ ràng và dễ hiểu, ưu tiên ví dụ thực tế và code khi phù hợp. Trả lời ngắn gọn nhưng đầy đủ ý, tránh sử dụng thuật ngữ phức tạp nếu không cần thiết. Khi người dùng đưa ra một vấn đề, hãy giải thích nguyên nhân, cách giải quyết và đưa ra ví dụ minh họa nếu cần." Việc chỉ định "bằng tiếng Việt" giúp model duy trì ngôn ngữ nhất quán và phù hợp với người dùng. Yêu cầu "ngắn gọn nhưng đầy đủ ý" giúp hạn chế câu trả lời quá dài nhưng vẫn đảm bảo các thông tin quan trọng được giải thích. Việc yêu cầu "ví dụ thực tế và code khi phù hợp" giúp trợ lý không chỉ đưa ra lý thuyết mà còn hỗ trợ người học áp dụng kiến thức.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là trợ lý chỉ sử dụng lịch sử hội thoại trong một số lượt gần nhất nên có thể quên các thông tin được trao đổi từ trước. Một cải thiện cụ thể là triển khai long-term memory bằng cách lưu các thông tin quan trọng của người dùng vào một cơ sở dữ liệu hoặc vector database. Khi có câu hỏi mới, hệ thống sẽ truy xuất những thông tin liên quan và đưa chúng vào context trước khi gọi LLM, giúp trợ lý duy trì ngữ cảnh tốt hơn qua nhiều phiên hội thoại.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
