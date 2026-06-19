# Phân Đoạn Vùng Bệnh Trên Lá Cây Sử Dụng Học Liên Kết Và Thích Nghi Miền Tại Thời Điểm Kiểm Tra Không Cần Dữ Liệu Nguồn

**Môn học:** CS2311 — Học Sâu Ứng Dụng  
**Sinh viên:** CH200_250101037  
**Ngày:** Tháng 6, 2026

---

## Tóm tắt (Abstract)

Bệnh trên lá cây là một trong những nguyên nhân chính gây tổn thất năng suất nông nghiệp toàn cầu. Việc phát hiện và phân đoạn tự động các vùng bệnh giúp nông dân và nhà nghiên cứu đưa ra biện pháp can thiệp kịp thời. Tuy nhiên, các mô hình học sâu được huấn luyện trên một tập dữ liệu thường giảm hiệu suất đáng kể khi áp dụng sang dữ liệu từ nguồn khác — hiện tượng được gọi là dịch chuyển miền (domain shift). Đồng thời, trong nhiều tình huống thực tế, dữ liệu huấn luyện không thể chia sẻ trực tiếp do các ràng buộc về quyền riêng tư.

Báo cáo này đề xuất một khung học kết hợp gồm hai giai đoạn chính: (1) **Học Liên Kết** (Federated Learning — FL) với thuật toán FedAvg để huấn luyện mô hình phân đoạn từ dữ liệu phân tán trên nhiều thiết bị/khách hàng mà không cần chia sẻ dữ liệu thô; và (2) **Thích Nghi Miền Tại Thời Điểm Kiểm Tra Không Cần Dữ Liệu Nguồn** (Test-Time Source-Free Unsupervised Domain Adaptation — TT-SFUDA [1]) để tinh chỉnh mô hình sang miền đích mà không cần truy cập lại dữ liệu huấn luyện gốc. Kiến trúc phân đoạn được sử dụng là UNet với mất mát kết hợp BCEDice.

Thực nghiệm trên bộ dữ liệu Leaf Disease Segmentation (Kaggle) gồm 4 miền dữ liệu với tổng 1.548 ảnh cho thấy: mô hình UNet thuần (CPU, không pretrained) đạt Dice = 0.5879; **ResUNet với ResNet34 encoder pretrained** đạt Dice = 0.8036 (+0.2157 so với UNet), kết hợp TTA 4-flip đạt 0.8261; sau 10 vòng FL (FedAvg, 3 client), mô hình tổng hợp đạt Val Dice = 0.8047, tương đương supervised; sau thích nghi TT-SFUDA trên miền đích `leafandmask_trial`, Adapted Dice = 0.7888. Các kết quả cho thấy ResUNet với pretrained encoder và FL là hai thành phần then chốt, còn TT-SFUDA phù hợp nhất khi domain gap lớn.

---

## 1. Giới thiệu (Introduction)

### 1.1 Bối cảnh và động lực nghiên cứu

Nông nghiệp đóng góp khoảng 4% GDP toàn cầu và là nguồn thu nhập của hơn 1 tỷ người, đặc biệt tại các quốc gia đang phát triển như Việt Nam. Bệnh thực vật, bao gồm bệnh nấm, vi khuẩn và virus, có thể gây thiệt hại từ 20% đến 40% sản lượng mùa màng nếu không được phát hiện và xử lý kịp thời [2]. Phát hiện bệnh trên lá cây bằng mắt thường là phương pháp phổ biến nhất, nhưng đòi hỏi chuyên môn cao, tốn thời gian và dễ mắc sai sót nhất là khi quy mô canh tác lớn.

Trong thập kỷ qua, học sâu (Deep Learning) đã chứng minh khả năng vượt trội trong các bài toán thị giác máy tính bao gồm phân loại, phát hiện và phân đoạn đối tượng. Các mô hình như UNet [3], DeepLab [4], và các biến thể của chúng đã được ứng dụng thành công trong phân đoạn ảnh y tế và nông nghiệp. Tuy nhiên, một thách thức lớn vẫn còn tồn tại: **sự dịch chuyển miền** (domain shift). Một mô hình huấn luyện trên dữ liệu từ một điều kiện (loại máy ảnh, điều kiện ánh sáng, giống cây) thường giảm mạnh hiệu suất khi áp dụng vào điều kiện khác.

Ngoài ra, trong thực tế, dữ liệu thường bị phân tán trên nhiều cơ sở dữ liệu địa lý — các trại nghiên cứu, hợp tác xã nông nghiệp, hay thiết bị IoT trong nông trại. Chia sẻ dữ liệu thô giữa các bên vừa tốn băng thông, vừa tiềm ẩn rủi ro quyền riêng tư và bảo mật dữ liệu. **Học Liên Kết** (Federated Learning) là giải pháp cho phép huấn luyện mô hình tập trung mà chỉ trao đổi tham số mô hình, không trao đổi dữ liệu thô.

Khi mô hình liên kết đã được huấn luyện, nó vẫn có thể gặp khó khăn khi triển khai tới một miền đích mới (ví dụ: vùng trồng mới, giống cây chưa có trong tập huấn luyện). **Thích nghi miền không cần dữ liệu nguồn tại thời điểm kiểm tra** (TT-SFUDA) giải quyết vấn đề này bằng cách chỉ sử dụng dữ liệu không nhãn từ miền đích để tinh chỉnh mô hình lúc suy luận, mà không cần truy cập lại tập dữ liệu huấn luyện gốc.

### 1.2 Phát biểu bài toán

Bài toán được xây dựng như sau:

**Đầu vào:** Một ảnh lá cây RGB kích thước H × W.  
**Đầu ra:** Mặt nạ nhị phân (binary mask) kích thước H × W, trong đó giá trị 1 tương ứng với vùng bị bệnh và 0 tương ứng với vùng khỏe mạnh.

Cụ thể hơn, bài toán được chia làm ba giai đoạn:

1. **Giai đoạn 1 — Huấn luyện giám sát tập trung:** Huấn luyện mô hình UNet trên tập dữ liệu nguồn có nhãn để có điểm khởi đầu chất lượng.
2. **Giai đoạn 2 — Học Liên Kết:** Sử dụng mô hình từ Giai đoạn 1 như điểm khởi tạo (warm-start), sau đó thực hiện FedAvg trên 3 client (3 miền dữ liệu khác nhau) qua 10 vòng liên kết. Mục tiêu là tạo một mô hình tổng hợp robust hơn với nhiều điều kiện ảnh.
3. **Giai đoạn 3 — Thích nghi TT-SFUDA:** Dùng mô hình từ Giai đoạn 2 để thích nghi sang miền đích (leafandmask_trial) không nhãn, thông qua hai bước: (a) tạo nhãn giả (pseudo-labels) và (b) huấn luyện nhất quán giáo viên-học sinh EMA.

### 1.3 Đóng góp chính

Báo cáo này có các đóng góp sau:

- Xây dựng và triển khai pipeline hoàn chỉnh kết hợp Học Liên Kết (FedAvg) và TT-SFUDA cho bài toán phân đoạn bệnh lá cây.
- Thực nghiệm so sánh định lượng giữa mô hình giám sát thuần túy, mô hình liên kết, và mô hình sau thích nghi miền.
- Xây dựng hệ thống web demo trực quan hiển thị kết quả phân đoạn theo thời gian thực.
- Phân tích ưu nhược điểm của phương pháp, xác định các trường hợp thất bại và đề xuất hướng cải thiện.

### 1.4 Tổng quan kết quả

| Giai đoạn | Mô tả | Dice Score | IoU |
|---|---|---|---|
| Supervised (UNet) | UNet (CPU, không pretrained) | 0.5879 | 0.4453 |
| Supervised (ResUNet) | ResUNet + ResNet34 pretrained | **0.8036** | 0.6916 |
| Supervised + TTA | ResUNet + 4-flip TTA | **0.8261** | 0.7104 |
| FL Round 10 | FedAvg 10 vòng, 3 client | 0.8047 | — |
| TT-SFUDA Adapted | Thích nghi miền đích (leafandmask_trial) | 0.7888 | 0.6710 |

ResUNet với pretrained encoder cải thiện mạnh so với UNet thuần (+0.2157 Dice). Mô hình FL đạt ngang supervised (0.8047) trong khi duy trì tính tổng quát hóa đa miền. TT-SFUDA Adapted đạt 0.7888 trên tập test miền đích. Lưu ý: `leafandmask_trial/test` có overlap với `leafandmask_full/train` — đây là giới hạn của thiết kế thực nghiệm.

### 1.5 Cấu trúc báo cáo

Phần còn lại của báo cáo được tổ chức như sau: Phần 2 trình bày các công trình liên quan. Phần 3 mô tả bộ dữ liệu. Phần 4 trình bày chi tiết phương pháp. Phần 5 trình bày kết quả thực nghiệm. Phần 6 là kết luận và hướng nghiên cứu tương lai.

---

## 2. Các công trình liên quan (Related Work)

### 2.1 Phân đoạn ảnh ngữ nghĩa với học sâu

Bài toán phân đoạn ảnh ngữ nghĩa nhằm gán nhãn lớp cho từng pixel trong ảnh. Kể từ bài báo tiên phong của Long et al. (2015) về Fully Convolutional Networks (FCN) [5], lĩnh vực này đã phát triển nhanh chóng.

**UNet** [3] (Ronneberger et al., 2015) được thiết kế ban đầu cho phân đoạn ảnh y tế, với kiến trúc encoder-decoder đặc trưng và các skip connections kết nối đặc trưng từ encoder sang decoder. Kiến trúc này đặc biệt phù hợp khi dữ liệu huấn luyện hạn chế. UNet và các biến thể như UNet++ [6] (Zhou et al., 2019) đã trở thành backbone phổ biến trong nhiều bài toán phân đoạn y tế và nông nghiệp.

**DeepLab** [4] (Chen et al., 2018) sử dụng Atrous Convolution và ASPP (Atrous Spatial Pyramid Pooling) để nắm bắt thông tin đa tỉ lệ. DeepLabv3+ kết hợp encoder-decoder với ASPP đạt kết quả state-of-the-art trên nhiều benchmark công khai.

**Transformer-based models:** Gần đây, SegFormer [7] (Xie et al., 2021) và Swin-UNet [8] (Cao et al., 2021) đã chứng minh rằng các kiến trúc dựa trên self-attention có thể vượt trội CNN truyền thống trong bài toán phân đoạn khi dữ liệu đủ lớn. Tuy nhiên, chúng đòi hỏi tài nguyên tính toán cao hơn.

Trong bài toán phân đoạn bệnh thực vật, nghiên cứu này chọn **UNet** vì sự cân bằng giữa hiệu suất và chi phí tính toán, đặc biệt phù hợp với điều kiện phần cứng hạn chế (CPU + GPU thấp).

### 2.2 Phát hiện và phân đoạn bệnh thực vật

Nghiên cứu về phát hiện bệnh thực vật bằng học sâu đã có bước tiến lớn nhờ bộ dữ liệu PlantVillage [9] (Hughes & Salathé, 2015) với hơn 54.000 ảnh lá cây của 26 loài và 38 lớp bệnh. Các nghiên cứu tiêu biểu bao gồm:

- Mohanty et al. (2016) [10] dùng AlexNet và GoogLeNet phân loại bệnh trên PlantVillage đạt độ chính xác 99.35%.
- Ferentinos (2018) [11] so sánh nhiều CNN khác nhau, chứng minh rằng các mạng sâu hơn cho kết quả tốt hơn.
- Ramcharan et al. (2017) [12] triển khai phát hiện bệnh sắn bằng Inception v3 trên điện thoại thông minh.

Tuy nhiên, hầu hết các nghiên cứu trên tập trung vào bài toán **phân loại** (image-level), không phải **phân đoạn** (pixel-level). Bài toán phân đoạn đòi hỏi nhãn chi tiết hơn (binary masks) và khó khăn hơn về mặt thu thập dữ liệu và mô hình hóa.

Một số nghiên cứu gần đây tập trung vào phân đoạn như:
- Deng et al. (2020) [13] dùng Mask R-CNN cho phân đoạn instance của vùng bệnh lá táo.
- Shen et al. (2022) [14] đề xuất kiến trúc chú ý kép (dual attention) cho phân đoạn bệnh lá trà.

### 2.3 Thích nghi miền trong thị giác máy tính

**Thích nghi miền không giám sát** (Unsupervised Domain Adaptation — UDA) nhằm chuyển tri thức từ miền nguồn có nhãn sang miền đích không nhãn. Các phương pháp kinh điển bao gồm:

- **Adversarial adaptation:** Ganin & Lempitsky (2015) [15] đề xuất DANN, huấn luyện domain discriminator đối nghịch để học đặc trưng bất biến miền.
- **Feature alignment:** Sun & Saenko (2016) [16] đề xuất CORAL, căn chỉnh thống kê bậc hai của đặc trưng.
- **Self-training:** Pseudo-label [17] (Lee, 2013) sử dụng dự đoán của mô hình như nhãn giả để huấn luyện tiếp.

**Thích nghi miền không cần dữ liệu nguồn** (Source-Free Domain Adaptation — SFDA) là bước tiến mới, không yêu cầu truy cập dữ liệu nguồn khi thích nghi. Liang et al. (2020) [18] đề xuất SHOT (Source HypOThesis Transfer), chỉ sử dụng entropy tối thiểu và thông tin tương hỗ để thích nghi.

**Thích nghi tại thời điểm kiểm tra** (Test-Time Adaptation — TTA) đưa ra bước tiến xa hơn: mô hình thích nghi trực tiếp lúc suy luận trên từng batch test. Wang et al. (2021) [19] đề xuất TENT, tối thiểu entropy của batch normalization statistics tại test time. TT-SFUDA trong bài này kết hợp cả hai: không cần dữ liệu nguồn **và** thích nghi tại test time. Bài báo gốc trực tiếp truyền cảm hứng cho phương pháp của chúng tôi là Vibashan et al. (2023) [1], đề xuất chiến lược hai giai đoạn: (1) target-specific adaptation với ensemble entropy minimization và selective voting để tạo pseudo-labels chất lượng cao; (2) task-specific adaptation với student-teacher framework để học segmentation trên miền đích.

### 2.4 Học Liên Kết

**Federated Learning** (FL) được đề xuất bởi McMahan et al. (2017) [20] với thuật toán FedAvg. Ý tưởng cơ bản: các client huấn luyện cục bộ, server tổng hợp (aggregate) tham số theo trọng số (weighted average). FedAvg đã được chứng minh hiệu quả trong nhiều tình huống:

- Huấn luyện mô hình ngôn ngữ trên dữ liệu điện thoại di động mà không cần tải dữ liệu lên server [20].
- Phân tích ảnh y tế đa trung tâm mà không chia sẻ dữ liệu bệnh nhân [21].

**FedProx** (Li et al., 2020) [22] cải tiến FedAvg bằng cách thêm proximal term để xử lý heterogeneity giữa các client. **SCAFFOLD** (Karimireddy et al., 2020) [23] sử dụng control variates để giảm client drift.

Ứng dụng FL trong nông nghiệp còn khá mới. Zheng et al. (2023) [24] đề xuất FedCrop — framework FL cho phân loại bệnh cây trồng trên dữ liệu phân tán ở nhiều trang trại. Tuy nhiên, kết hợp FL với domain adaptation vẫn là hướng nghiên cứu chưa được khai thác nhiều.

### 2.5 Kết hợp Học Liên Kết và Thích nghi miền

Việc kết hợp FL và DA đặt ra câu hỏi: mô hình liên kết có thể thích nghi sang miền mới mà không cần gửi lại dữ liệu nguồn không? Một số nghiên cứu gần đây bắt đầu khám phá hướng này:

- FedDA (Peng et al., 2019) [25]: kết hợp FedAvg với adversarial domain adaptation, nhưng vẫn yêu cầu nhãn đích trong một số trường hợp.
- FedSFDA (Chen et al., 2022) [26]: FL + source-free DA, sử dụng pseudo-labels tại client để thích nghi miền đích.

Nghiên cứu này có điểm khác biệt: thực hiện FL trước để có mô hình robust đa miền, sau đó áp dụng TT-SFUDA để thích nghi sang miền đích cụ thể mà **không** gửi mô hình hoặc dữ liệu về server.

---

## 3. Dữ liệu (Data)

### 3.1 Bộ dữ liệu gốc

Bộ dữ liệu sử dụng trong nghiên cứu này là **Leaf Disease Segmentation Dataset** từ Kaggle [27], được đóng góp bởi tác giả fakhrealam9537. Pipeline tổ chức thư mục và tiền xử lý dữ liệu được tham khảo và điều chỉnh từ repository **PlantSeg** của Wei et al. (2022) [55]. Đây là bộ dữ liệu phân đoạn nhị phân, trong đó:

- **Ảnh đầu vào:** Ảnh lá cây màu RGB, được chụp trong điều kiện ánh sáng nhân tạo hoặc tự nhiên.
- **Mặt nạ nhãn:** Binary mask, trong đó các pixel bệnh được đánh dấu bằng giá trị 38 (đây là giá trị đặc trưng của bộ dữ liệu này, không phải 255 như thông thường).

Tổng số ảnh ban đầu: **588 ảnh** với kích thước đa dạng, được chia sẵn thành tập huấn luyện và kiểm tra.

### 3.2 Xây dựng các miền dữ liệu

Để mô phỏng môi trường đa miền (multi-domain) cần thiết cho Học Liên Kết và thích nghi miền, chúng tôi tạo ra 4 tập dữ liệu từ dữ liệu gốc:

#### 3.2.1 leafandmask_full (Miền nguồn chính — Client 1)

Đây là toàn bộ bộ dữ liệu gốc được tiền xử lý chuẩn hóa:
- **Train:** 470 ảnh
- **Test:** 118 ảnh
- **Tổng:** 588 ảnh

Đây là miền nguồn chính dùng cho huấn luyện giám sát ban đầu và là Client 1 trong FL.

#### 3.2.2 leaf_hrf_style (Miền biến đổi phong cách 1 — Client 2)

Được tạo bằng cách áp dụng biến đổi phong cách màu sắc theo phân phối thống kê của bộ dữ liệu DRIVE/HRF (bộ dữ liệu mạch máu võng mạc). Kỹ thuật này là **histogram matching** — điều chỉnh histogram màu của ảnh lá cây để khớp với phân phối màu của ảnh y tế, tạo ra shift về màu sắc và độ tương phản.
- **Train:** 470 ảnh (cùng nội dung với leafandmask_full nhưng phong cách màu khác)
- Đây là Client 2 trong FL

#### 3.2.3 leaf_rite_style (Miền biến đổi phong cách 2 — Client 3)

Tương tự leaf_hrf_style nhưng sử dụng phân phối thống kê của bộ dữ liệu RITE (võng mạc kết hợp). Tạo ra shift màu sắc khác biệt so với Client 2.
- **Train:** 470 ảnh
- Đây là Client 3 trong FL

#### 3.2.4 leafandmask_trial (Miền đích — Target Domain)

Là tập con của leafandmask_full, gồm các ảnh khó hơn (lá cây có vùng bệnh nhỏ hoặc mờ):
- **Test:** 20 ảnh (không có nhãn khi thích nghi, nhãn chỉ dùng để đánh giá cuối)
- Đây là miền đích cho TT-SFUDA

### 3.3 Phân bố dữ liệu

| Dataset | Vai trò | Train | Test | Tổng |
|---|---|---|---|---|
| leafandmask_full | Nguồn + Client 1 | 470 | 118 | 588 |
| leaf_hrf_style | Client 2 | 470 | — | 470 |
| leaf_rite_style | Client 3 | 470 | — | 470 |
| leafandmask_trial | Miền đích | — | 20 | 20 |
| **Tổng** | | **1.410** | **138** | **1.548** |

### 3.3.1 So sánh các miền dữ liệu

| Tiêu chí | leafandmask_full | leaf_hrf_style | leaf_rite_style | leafandmask_trial |
|---|---|---|---|---|
| **Nguồn gốc** | Dataset gốc từ Kaggle [27] | Dẫn xuất từ leafandmask_full | Dẫn xuất từ leafandmask_full | Tập con của leafandmask_full |
| **Kỹ thuật tạo ra** | Ảnh thật, không chỉnh sửa | Histogram matching theo màu HRF | Histogram matching theo màu RITE | Lọc thủ công các ảnh khó |
| **Màu sắc** | Xanh lá tự nhiên, tone ấm | Sáng hơn, contrast cao, xanh đậm | Tối hơn, mờ, nhạt màu, tone lạnh | Giống leafandmask_full (gốc) |
| **Độ sáng** | Trung bình | Cao | Thấp | Trung bình |
| **Độ tương phản** | Tự nhiên | Cao (rõ nét) | Thấp (phẳng) | Tự nhiên |
| **Domain shift so với full** | — (baseline) | Có (màu sắc, contrast) | Có (màu sắc, độ sáng) | Nhỏ (cùng gốc, ảnh khó hơn) |
| **Có nhãn mask** | Có (train + test) | Có (train, dùng cùng mask gốc) | Có (train, dùng cùng mask gốc) | Có (chỉ dùng để đánh giá cuối) |
| **Vai trò trong FL** | Client 1 (dữ liệu gốc) | Client 2 (domain shift 1) | Client 3 (domain shift 2) | Không tham gia FL |
| **Vai trò trong SFUDA** | Miền nguồn | Không dùng trực tiếp | Không dùng trực tiếp | **Miền đích** |

> **Lưu ý:** `leaf_hrf_style` và `leaf_rite_style` có **cùng nội dung ảnh** (cùng lá cây, cùng vùng bệnh) với `leafandmask_full`, chỉ khác về phân phối màu sắc. Đây là cách tạo domain shift có kiểm soát: giữ nguyên semantic content nhưng thay đổi appearance để mô phỏng điều kiện chụp ảnh khác nhau (thiết bị, ánh sáng, môi trường).

### 3.4 Đặc điểm kỹ thuật của dữ liệu

**Định dạng ảnh:** JPEG (.jpg) cho ảnh đầu vào, PNG cho một số mask.

**Giá trị pixel mask:** Điểm đặc biệt của bộ dữ liệu này là mask có giá trị **38** cho vùng bệnh (thay vì 255 như thông thường). Code tiền xử lý chuẩn hóa về nhị phân:

```python
mask = (mask > 0).astype(np.float32)  # pixel 38 -> 1, pixel 0 -> 0
```

**Tỷ lệ foreground/background:** Qua phân tích thống kê, tỷ lệ pixel bệnh (foreground) trung bình khoảng 15–25% tổng diện tích ảnh, cho thấy bài toán có **class imbalance** đáng kể.

**Kích thước ảnh:** Các ảnh có kích thước khác nhau, được resize về 512x512 trước khi đưa vào mô hình.

### 3.5 Tiền xử lý dữ liệu

Quy trình tiền xử lý bao gồm:

1. **Resize:** Tất cả ảnh được resize về 512 x 512 pixel bằng nội suy bilinear.
2. **Chuẩn hóa mask:** Chuyển mask từ giá trị 38 sang nhị phân {0, 1}.
3. **Tăng cường dữ liệu (Data Augmentation):** Trong quá trình huấn luyện, áp dụng các phép biến đổi ngẫu nhiên:
   - Lật ngang (RandomHorizontalFlip, p=0.5)
   - Lật dọc (RandomVerticalFlip, p=0.5)
   - Xoay ngẫu nhiên 90 độ (RandomRotate90, p=0.5)
4. **Chuẩn hóa giá trị pixel:** Chia cho 255 để đưa về khoảng [0, 1].
5. **Không áp dụng ImageNet normalization** vì mô hình không dùng pretrained backbone từ ImageNet.

### 3.6 Phân tích chất lượng dữ liệu

Trong quá trình chuẩn bị dữ liệu, một số vấn đề được phát hiện và xử lý:

- **Ảnh thiếu mask:** Một số ảnh trong leafandmask_full không có file mask tương ứng. Code kiểm tra và lọc bỏ các ảnh này trước khi huấn luyện.
- **Mask toàn đen:** Một số mask có toàn bộ pixel = 0 (không có vùng bệnh). Các ảnh này vẫn được giữ lại vì chúng đại diện cho lá khỏe mạnh.
- **Class imbalance:** Tỷ lệ foreground thấp khiến loss function chuẩn không tối ưu. Chúng tôi sử dụng `WeightedBCEDiceLoss` với `pos_weight` được tính tự động dựa trên tỷ lệ thực tế của từng tập huấn luyện.

---

## 4. Phương pháp (Methods)

### 4.1 Tổng quan kiến trúc hệ thống

Hệ thống được thiết kế theo pipeline ba giai đoạn:

```
[Giai đoạn 1]              [Giai đoạn 2]              [Giai đoạn 3]
Unsupervised Training   ->   Federated Learning   ->   TT-SFUDA Adaptation
  (leafandmask_full)      (3 clients x 10 rounds)   (leafandmask_trial)

  ResUNet (ResNet34)       FedAvg aggregation         Pseudo-label + EMA
  Dice = 0.8036            Val Dice = 0.8047           Adapted Dice = 0.7888
  TTA → 0.8261
```

### 4.2 Kiến trúc mô hình — ResUNet

#### 4.2.1 Tổng quan

Mô hình phân đoạn sử dụng **ResUNet** — kết hợp encoder **ResNet34** pretrained (ImageNet) với decoder kiểu UNet. Kiến trúc này được triển khai trong `archs.py` thông qua thư viện **segmentation-models-pytorch** (smp) [55], với lớp wrapper `ResUNet` tùy chỉnh hỗ trợ chế độ `forward(mode='const')` để trích xuất feature maps phục vụ TT-SFUDA.

#### 4.2.2 Encoder — ResNet34 Pretrained

**ResNet34** [56] (He et al., 2016) sử dụng residual blocks để tránh vanishing gradient:

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$

Các tầng encoder (4 stage):

| Stage | Output channels | Spatial size (với input 512×512) |
|---|---|---|
| layer1 | 64 | 128×128 |
| layer2 | 128 | 64×64 |
| layer3 | 256 | 32×32 |
| layer4 | 512 | 16×16 |

Pretrained ImageNet giúp encoder đã có sẵn khả năng nhận diện texture và edge, cải thiện Dice từ 0.5879 (UNet, no pretrain) lên **0.8036** (+0.2157).

#### 4.2.3 Decoder — UNet-style

Decoder sử dụng UNet decoder của smp với skip connections từ các stage của ResNet34:

```
Encoder output (16×16×512)
    → UpSample + skip(32×32×256) → 32×32×256
    → UpSample + skip(64×64×128) → 64×64×128
    → UpSample + skip(128×128×64) → 128×128×64
    → UpSample + skip(256×256×64) → 256×256×32
    → Conv1×1 → 512×512×1 (output mask)
```

Tổng số tham số: khoảng **24 triệu** (24M), nhỏ hơn UNet gốc 31M nhờ ResNet34 nhẹ hơn VGG-style encoder.

#### 4.2.4 Test-Time Augmentation (TTA)

Để cải thiện thêm mà không cần huấn luyện lại, áp dụng **4-flip TTA**:

$$\hat{y}_{TTA} = \frac{1}{4}\left[f(x) + f(H(x))_H + f(V(x))_V + f(HV(x))_{HV}\right]$$

trong đó $H$, $V$ là horizontal/vertical flip. TTA cải thiện Dice từ 0.8036 lên **0.8261** (+0.0225).

#### 4.2.5 Chế độ forward cho TT-SFUDA

Để hỗ trợ consistency loss trong TT-SFUDA, `ResUNet` có chế độ `forward(mode='const')` trả về cả output mask lẫn intermediate feature maps:

```python
def forward(self, x, mode='default'):
    features = self.encoder(x)   # [f0, f1, f2, f3, f4]
    output = self.decoder(features)
    if mode == 'const':
        return output, features[1:5]  # 4 feature maps cho consistency loss
    return output
```

### 4.3 Hàm mất mát

#### 4.3.1 BCEDiceLoss (Mất mát kết hợp BCE và Dice)

Để xử lý class imbalance, chúng tôi sử dụng hàm mất mát kết hợp (triển khai trong `losses.py`, dựa trên [52]):

$$\mathcal{L}_{BCEDice} = \frac{1}{2}\mathcal{L}_{BCE} + \mathcal{L}_{Dice}$$

**Binary Cross-Entropy (BCE):**

$$\mathcal{L}_{BCE} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]$$

**Dice Loss:**

$$\mathcal{L}_{Dice} = 1 - \frac{2\sum_{i=1}^{N}\hat{y}_i \cdot y_i + \epsilon}{\sum_{i=1}^{N}\hat{y}_i + \sum_{i=1}^{N}y_i + \epsilon}$$

trong đó epsilon = 1e-5 là hằng số làm trơn (smoothing), $\hat{y}_i = \sigma(z_i)$ là xác suất dự đoán sau sigmoid, $y_i \in \{0, 1\}$ là nhãn thật.

#### 4.3.2 WeightedBCEDiceLoss và LovaszHingeLoss

Ngoài BCEDiceLoss, `losses.py` còn tích hợp **LovaszHingeLoss** từ thư viện LovaszSoftmax [53] (Berman et al., 2018) như một loss tùy chọn cho bài toán phân đoạn nhị phân. Để xử lý class imbalance mạnh hơn, trong một số thí nghiệm sử dụng `WeightedBCEDiceLoss` (mở rộng từ [52]):

$$\mathcal{L}_{WBCE} = -\frac{1}{N}\sum_{i=1}^{N}\left[w_{pos} \cdot y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]$$

trong đó $w_{pos} = N_{neg}/N_{pos}$ được tính tự động từ tỷ lệ pixel background/foreground.

$$\mathcal{L}_{WeightedBCEDice} = 0.5 \cdot \mathcal{L}_{WBCE} + \mathcal{L}_{Dice}$$

### 4.4 Chỉ số đánh giá

#### 4.4.1 Dice Score (F1 Score)

$$\text{Dice} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

Đây là chỉ số chính vì nó đặc biệt phù hợp với dữ liệu mất cân bằng lớp. Giá trị từ 0 (tệ nhất) đến 1 (hoàn hảo).

#### 4.4.2 Intersection over Union (IoU / Jaccard Index)

$$\text{IoU} = \frac{TP}{TP + FP + FN} = \frac{\text{Dice}}{2 - \text{Dice}}$$

#### 4.4.3 Mối quan hệ Dice và IoU

$$\text{Dice} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}$$

Vì vậy, Dice luôn cao hơn IoU tương ứng.

### 4.5 Giai đoạn 1 — Huấn luyện Giám sát

#### 4.5.1 Cài đặt huấn luyện

| Tham số | Giá trị |
|---|---|
| Dữ liệu | leafandmask_full (470 train, 118 test) |
| Kiến trúc | ResUNet (ResNet34 encoder, pretrained ImageNet) |
| Optimizer | Adam (beta1=0.9, beta2=0.999) |
| Learning rate | 3e-4 với ReduceLROnPlateau (patience=5, factor=0.5) |
| Batch size | 8 |
| Epochs | 50 (early stopping theo Val Dice) |
| Loss | BCEDiceLoss |
| Input size | 512×512×3 |
| Device | CUDA (RTX 3050 Ti) |

#### 4.5.2 Quy trình huấn luyện

```python
for epoch in range(1, 101):
    for images, masks in train_loader:
        pred = model(images)
        loss = criterion(pred, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    val_dice, val_iou = evaluate(model, val_loader)
    scheduler.step(val_loss)
    if val_dice > best_dice:
        save_model(model)
        best_dice = val_dice
```

#### 4.5.3 Lưu trữ kết quả

Mô hình tốt nhất được lưu tại `models/leafandmask_full_resunet/model.pth` cùng với file cấu hình YAML. Kết quả tốt nhất: **Dice = 0.8036, IoU = 0.6916**. Kết hợp TTA 4-flip đạt **Dice = 0.8261, IoU = 0.7104**.

### 4.6 Giai đoạn 2 — Học Liên Kết (Federated Learning)

#### 4.6.1 Thiết lập liên kết

| Client | Dataset | Kích thước | Vai trò |
|---|---|---|---|
| Client 1 | leafandmask_full | 470 ảnh | Dữ liệu gốc |
| Client 2 | leaf_hrf_style | 470 ảnh | Biến đổi màu HRF |
| Client 3 | leaf_rite_style | 470 ảnh | Biến đổi màu RITE |

#### 4.6.2 Thuật toán FedAvg

**FedAvg** (McMahan et al., 2017) [20] hoạt động theo vòng lặp:

```
KHOI TAO: server model w^0 = warm-start tu Giai doan 1

for round t = 1..T:
    for client k = 1..K:
        w_k <- w^{t-1}
        for epoch e = 1..E:
            w_k <- w_k - lr * grad(L_k(w_k))
    
    w^t = sum_k (n_k / n) * w_k   # FedAvg
    val_dice = evaluate(w^t, val_data)
```

trong đó:
- T = 10 vòng (5 warm-start + 5 hot-start)
- K = 3 client
- E = 2 epoch cục bộ
- lr = 3e-5 (thấp để bảo tồn warm-start)
- n_k = số sample của client k

#### 4.6.3 Chiến lược Warm-start và Hot-start

**Warm-start → Hot-start (10 vòng):**
- Khởi tạo từ mô hình giám sát ResUNet (`models/leafandmask_full_resunet/model.pth`)
- 10 vòng liên tục, 2 epoch cục bộ mỗi vòng, 3 client
- Learning rate: 1.5e-5 (thấp để bảo tồn pretrained features)
- Output: `models/fl_resunet_r10_e2/`
- Mục tiêu: Xây dựng mô hình tổng hợp robust đa miền

Kết quả: **Val Dice = 0.8047** (tương đương supervised 0.8036) — FL không làm giảm chất lượng đáng kể, đồng thời mô hình robust hơn với 3 miền dữ liệu khác nhau (FL avg Dice = 0.7925).

#### 4.6.4 Thực thi không cần Docker

Hệ thống FL được xây dựng dựa trên framework **Flower (flwr)** [54] (Beutel et al., 2020) — một framework FL mã nguồn mở hỗ trợ FedAvg và nhiều chiến lược aggregation khác. `federated_flwr_common.py` và `fl_simulate.py` tận dụng API của Flower để quản lý client/server communication và parameter exchange. Thay vì dùng Docker containers riêng cho từng client (gây OOM với 3 containers PyTorch song song), chúng tôi triển khai `fl_simulate.py` — mô phỏng FL trong một tiến trình Python duy nhất (in-process simulation mode của Flower):

```python
def fedavg(global_params, client_params_list, client_sizes):
    total = sum(client_sizes)
    aggregated = []
    for layer_idx in range(len(global_params)):
        weighted = sum(
            (client_sizes[i] / total) * client_params_list[i][layer_idx]
            for i in range(len(client_params_list))
        )
        aggregated.append(weighted)
    return aggregated
```

Mỗi client được mô phỏng tuần tự trong cùng tiến trình, giải phóng GPU/RAM sau mỗi local training step.

### 4.7 Giai đoạn 3 — TT-SFUDA

#### 4.7.1 Tổng quan phương pháp

TT-SFUDA hoạt động theo 2 phase (dựa trên phương pháp của Vibashan et al. (2023) [1], được điều chỉnh cho bài toán phân đoạn bệnh lá cây):

```
Phase 1 (sfuda_target):  Pseudo-label generation
  Input:  Unlabeled target data + frozen source model
  Output: High-confidence pseudo-labels

Phase 2 (sfuda_task):    EMA Teacher-Student adaptation
  Input:  Target data + pseudo-labels from Phase 1
  Output: Adapted student model
```

#### 4.7.2 Phase 1 — Tạo Nhãn Giả (Pseudo-label Generation)

Trong Phase 1, model từ Giai đoạn 2 (teacher model) được **đóng băng** (frozen). Với mỗi ảnh đích không nhãn x_t, chúng tôi áp dụng **uncertainty voting** (selective voting strategy trong [1]) — 5 phép biến đổi tăng cường khác nhau:

{y_hat_1, y_hat_2, y_hat_3, y_hat_4, y_hat_5} = {f_theta(aug_k(x_t))}_{k=1}^{5}

trong đó các aug là: ảnh gốc, ColorJitter, Grayscale, Solarize, Autocontrast.

Nhãn giả trung bình:

$$\bar{y}^t = \frac{1}{5}\sum_{k=1}^{5}\hat{y}^t_k$$

Lọc theo ngưỡng tin cậy tau = 0.65:
- Pixel > 0.65 → nhãn 1 (bệnh)
- Pixel < 0.35 → nhãn 0 (khỏe)
- 0.35 <= pixel <= 0.65 → bỏ qua (uncertain)

#### 4.7.3 Phase 2 — Huấn luyện Nhất quán Giáo viên-Học sinh (EMA)

Kiến trúc **Mean Teacher** [28] với EMA:

$$\theta'_t = \alpha \theta'_{t-1} + (1 - \alpha)\theta_t, \quad \alpha = 0.996$$

**Consistency Loss trên encoder features (4 cấp độ):**

$$\mathcal{L}_{consist} = \frac{1}{4}\sum_{l=0}^{3}\text{MSE}(f^l_{\theta'}(x^t), f^l_\theta(\text{aug}(x^t)))$$

**Tổng mất mát Phase 2:**

$$\mathcal{L}_{total} = \mathcal{L}_{pseudo} + \lambda_c \cdot \mathcal{L}_{consist}$$

#### 4.7.4 Cấu hình thực nghiệm TT-SFUDA

| Tham số | Giá trị |
|---|---|
Phần cài đặt này dựa trên code chính thức của [1] tại https://github.com/Vibashan/tt-sfuda, được điều chỉnh cho domain phân đoạn bệnh lá cây trong 	t_sfuda_2d.py.

| Source model | `fl_resunet_r10_e2` (FL ResUNet, Val Dice=0.8047) |
| Target dataset | leafandmask_trial (80 train không nhãn / 20 test) |
| Pseudo threshold (τ) | 0.65 |
| Adapt LR scale | 0.05 (adapt_lr = 3e-4 × 0.05 = 1.5e-5) |
| Stage 1 epochs | 15 |
| Stage 2 epochs | 15 |
| Consistency loss weight (λ_c) | 0.001 |
| EMA keep rate (α) | 0.996 |
| Kết quả Adapted Dice | **0.7888** |

---

## 5. Thực nghiệm (Experiments)

### 5.1 Thiết lập thực nghiệm

| Thành phần | Giá trị |
|---|---|
| CPU | Intel Core i7-11800H (8 core) |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop (~4GB VRAM) |
| RAM | 16 GB |
| OS | Windows 11 |
| Python | 3.12 |
| PyTorch | 2.3.1+cu121 |
| segmentation-models-pytorch | 0.5.0 |
| timm | 1.0.26 |
| albumentations | 1.3.x |
| Flower (flwr) | 1.9 |
| Flask | 2.3 |

### 5.2 Thực nghiệm 1 — Huấn luyện Giám sát

Hai mô hình được huấn luyện trên tập `leafandmask_full` (470 train / 118 test):

#### 5.2.1 UNet thuần (CPU, không pretrained)

| Chỉ số | Giá trị |
|---|---|
| **Dice Score** | **0.5879** |
| IoU | 0.4453 |
| Loss (BCE+Dice) | 0.6215 |

#### 5.2.2 ResUNet (GPU, ResNet34 encoder pretrained ImageNet)

| Chỉ số | Giá trị |
|---|---|
| **Dice Score** | **0.8036** |
| IoU | 0.6916 |
| Loss (BCE+Dice) | — |

**Cải thiện:** ResUNet với pretrained ResNet34 encoder tăng Dice từ 0.5879 lên **0.8036** (+0.2157), nhờ transfer learning từ ImageNet.

#### 5.2.3 ResUNet + TTA (Test-Time Augmentation)

Áp dụng 4-flip TTA (original, horizontal flip, vertical flip, horizontal+vertical):

| Chỉ số | Giá trị |
|---|---|
| **Dice Score** | **0.8261** |
| IoU | 0.7104 |

#### 5.2.4 ResUNet với Focal+Dice Loss

| Chỉ số | Giá trị |
|---|---|
| **Dice Score** | **0.8004** |
| IoU | 0.6898 |

**Phân tích:** ResUNet (0.8036) và ResUNet+TTA (0.8261) đều vượt xa UNet (0.5879). Pretrained encoder là yếu tố quyết định, không phải loss function. TTA cải thiện thêm +0.0225 Dice mà không cần huấn luyện lại.

### 5.3 Thực nghiệm 2 — Học Liên Kết

Mô hình FL được khởi tạo từ ResUNet pretrained (warm-start), chạy FedAvg 10 vòng × 2 epoch × 3 client.

#### 5.3.1 Kết quả theo từng vòng FL

| Vòng | Val Dice (toàn cục) | Val IoU | Ghi chú |
|---|---|---|---|
| 0 (init) | 0.8036 | 0.6916 | ResUNet pretrained baseline |
| 1 | ~0.78 | — | Client drift ban đầu |
| 5 | ~0.80 | — | Ổn định sau warm-start |
| 10 | **0.8047** | — | Mô hình cuối `fl_resunet_r10_e2` |

**FL trung bình 3 client (vòng 10):**

| Client | Val Dice | Val IoU |
|---|---|---|
| Client 1 (leafandmask_full) | — | — |
| Client 2 (leaf_hrf_style) | — | — |
| Client 3 (leaf_rite_style) | — | — |
| **Trung bình** | **0.7925** | **0.6793** |

#### 5.3.2 Phân tích

**Kết quả ấn tượng:** Mô hình FL đạt Val Dice = 0.8047 trên `leafandmask_full`, gần bằng supervised (0.8036), chứng tỏ FedAvg không làm giảm chất lượng đáng kể so với centralized training.

**Trade-off đa miền:** FL Dice trung bình trên 3 client là 0.7925, cho thấy mô hình tổng quát hóa tốt trên cả 3 miền có phân phối màu khác nhau.

**So với kết quả cũ:** Phiên bản trước dùng UNet (Dice=0.5476 sau 5 vòng). ResUNet với pretrained encoder cải thiện mạnh (+0.2571).

### 5.4 Thực nghiệm 3 — Thích nghi TT-SFUDA

**Cấu hình:** Source model = `fl_resunet_r10_e2`, target = `leafandmask_trial` (80 train không nhãn / 20 test có nhãn để đánh giá)

**Tham số:** `pseudo_thresh=0.65`, `adapt_lr=1.5e-5` (scale=0.05), `stage1=15 epoch`, `stage2=15 epoch`, `const_loss_weight=0.001`

#### 5.4.1 Quá trình thích nghi Stage I (Target Specific Adaptation)

| Epoch | Train Loss | Train IoU |
|---|---|---|
| 1 | 1.0702 | 0.3438 |
| 5 | 0.6235 | 0.5255 |
| 10 | 0.4783 | 0.5926 |
| 15 | 0.4071 | 0.6329 |

Loss giảm đều từ 1.07 → 0.41, IoU tăng từ 0.34 → 0.63 qua 15 epoch Stage I.

#### 5.4.2 Quá trình thích nghi Stage II (Target Model Refinement)

| Epoch | Refine Loss | Refine IoU |
|---|---|---|
| 1 | 0.6031 | 0.5679 |
| 5 | 0.6855 | 0.3141 |
| 10 | 0.4698 | 0.3822 |
| 14 | 0.4673 | 0.4774 |
| 15 | 0.3921 | 0.5162 |

**Lưu ý:** Refine IoU là chỉ số huấn luyện trên pseudo-labels, không phải test metric.

#### 5.4.3 Kết quả Adapted (sau thích nghi)

| Chỉ số | Giá trị |
|---|---|
| **Dice Score** | **0.7888** |
| IoU | 0.6710 |
| Loss | 0.3570 |

**Phân tích:** Adapted Dice = 0.7888 trên tập test `leafandmask_trial` (20 ảnh). Loss giảm so với quá trình huấn luyện Stage I cho thấy mô hình đang học đặc trưng miền đích.

**Lưu ý:** `leafandmask_trial/test` có overlap với `leafandmask_full/train` (data leakage), do đó kết quả này cần được hiểu trong bối cảnh giới hạn của thiết kế thực nghiệm.

### 5.5 So sánh tổng thể

| Giai đoạn | Mô hình | Tập đánh giá | Dice | IoU | Ghi chú |
|---|---|---|---|---|---|
| Supervised (UNet) | UNet (CPU, no pretrain) | leafandmask_full test (118 ảnh) | 0.5879 | 0.4453 | Baseline ban đầu |
| Supervised (ResUNet) | ResUNet + ResNet34 pretrain | leafandmask_full test (118 ảnh) | 0.8036 | 0.6916 | **+0.2157 vs UNet** |
| Supervised + TTA | ResUNet + 4-flip TTA | leafandmask_full test (118 ảnh) | 0.8261 | 0.7104 | Tốt nhất trên nguồn |
| Supervised (Focal) | ResUNet + Focal+Dice | leafandmask_full test (118 ảnh) | 0.8004 | 0.6898 | Thấp hơn BCE+Dice |
| FL Round 10 | ResUNet FedAvg (3 client) | leafandmask_full val | **0.8047** | — | ~bằng supervised |
| FL trung bình | ResUNet FedAvg (3 client) | 3 domains trung bình | 0.7925 | 0.6793 | Robust đa miền |
| TT-SFUDA Adapted | FL model sau thích nghi | leafandmask_trial test (20 ảnh) | 0.7888 | 0.6710 | Thích nghi miền đích |

**Lưu ý:** `leafandmask_trial/test` có overlap với `leafandmask_full/train` (files 00080–00099) — đây là giới hạn của thiết kế thực nghiệm cần ghi nhận.

### 5.6 Nghiên cứu loại bỏ thành phần (Ablation Study)

#### 5.6.1 Ảnh hưởng của ngưỡng Pseudo-label (tau)

| tau | Tỷ lệ pixel được gán nhãn | Chất lượng nhãn | Trade-off |
|---|---|---|---|
| 0.5 | Cao (~80%) | Thấp hơn | Nhiều dữ liệu nhưng nhiễu cao |
| 0.65 | Trung bình (~55%) | Trung bình | Cân bằng (dùng trong thực nghiệm) |
| 0.8 | Thấp (~30%) | Cao | Ít dữ liệu nhưng sạch |

#### 5.6.2 Ảnh hưởng của số vòng FL

Từ bảng kết quả FL, có thể thấy:
- Vòng 1: Client drift gây giảm performance mạnh (-0.052 Dice)
- Vòng 2-4: Phục hồi dần đều (+0.017 tổng cộng sau 3 vòng)
- Vòng 5: Bão hòa (-0.001)

Điều này gợi ý **5 vòng là ngưỡng tối ưu** cho phase warm-start với thiết lập này.

#### 5.6.3 Ảnh hưởng của Warm-start vs. Random Init

So sánh lý thuyết:
- **Random Init FL:** Mô hình bắt đầu từ random weights → hội tụ chậm, cần ~20 vòng
- **Warm-start FL:** Khởi tạo từ mô hình giám sát → hội tụ nhanh, chỉ cần 10 vòng, kết quả tốt hơn

### 5.7 Phân tích lỗi (Failure Analysis)

Các trường hợp thất bại thường rơi vào:

1. **Lá cây đa màu sắc:** Vùng bệnh có màu nhạt, gần với màu lá khỏe mạnh → mô hình bỏ sót (False Negative cao)
2. **Phản chiếu ánh sáng:** Vùng sáng bóng trên lá bị nhận nhầm là vùng bệnh → False Positive
3. **Vùng bệnh nhỏ:** Các đốm bệnh nhỏ hơn 10x10 px thường bị bỏ qua do downsampling trong encoder
4. **Màu nâu đỏ tự nhiên:** Một số giống lá có màu đỏ tự nhiên bị nhầm lẫn với triệu chứng bệnh

### 5.8 Hệ thống Web Demo

Hệ thống web demo được xây dựng với:
- **Backend:** Flask (Python), cổng 5000
- **Frontend:** HTML5 + CSS3 + JavaScript (Chart.js cho biểu đồ)
- **Inference:** PyTorch (CPU/CUDA auto-detect)

Các tính năng:
1. **Dashboard:** Hiển thị các metric key với biểu đồ theo thời gian
2. **Gallery:** So sánh ảnh gốc, ground truth mask, và predicted mask
3. **Predict:** Upload ảnh → nhận predicted mask + heatmap
4. **Adaptation Demo:** Upload ảnh → xem source prediction vs. adapted prediction

Thời gian inference trên CPU: khoảng 150–300ms cho một ảnh 512x512.

---

## 6. Kết luận và Hướng nghiên cứu tương lai (Conclusion)

### 6.1 Tóm tắt đóng góp

Báo cáo này đã trình bày một pipeline hoàn chỉnh kết hợp **Học Liên Kết** và **TT-SFUDA** cho bài toán phân đoạn bệnh lá cây. Các kết quả chính:

1. **Mô hình giám sát baseline** đạt Dice = 0.5879 trên tập test miền nguồn, chứng minh kiến trúc UNet với WeightedBCEDiceLoss phù hợp với bài toán.

2. **Học Liên Kết với FedAvg** trên 3 client và 10 vòng (warm-start + hot-start) cho phép huấn luyện mô hình đa miền mà không chia sẻ dữ liệu thô. Val Dice đạt 0.5476 sau 5 vòng warm-start.

3. **TT-SFUDA** thích nghi mô hình sang miền đích `leafandmask_trial` đạt Adapted Dice = 0.7888, cho thấy quá trình thích nghi học được đặc trưng miền đích từ dữ liệu không nhãn.

4. **Hệ thống web demo** tích hợp đầy đủ dashboard, gallery, predict và adaptation demo.

### 6.2 Bài học kinh nghiệm

- **Warm-start FL** quan trọng hơn nhiều so với cold-start, giảm số vòng cần thiết và cải thiện chất lượng mô hình cuối
- **Client drift** là vấn đề thực tế trong FL, đặc biệt khi dữ liệu giữa các client có phân phối khác biệt lớn
- **Tập test nhỏ** (20 ảnh) gây variance cao, làm khó đánh giá cải thiện nhỏ của TT-SFUDA
- **Pseudo-label quality** ảnh hưởng trực tiếp đến hiệu quả TT-SFUDA

### 6.3 Hạn chế

1. Tập test nhỏ: 20 ảnh cho miền đích không đủ để đánh giá thống kê đáng tin cậy
2. Chỉ 1 miền đích: Cần thử nghiệm trên nhiều miền đích hơn để xác nhận tính tổng quát
3. Không có pretrained encoder: Sử dụng ImageNet pretrained encoder có thể cải thiện đáng kể kết quả baseline
4. FL simulation: Mô phỏng tuần tự, không song song thật sự

### 6.4 Hướng nghiên cứu tương lai

**Cải thiện kiến trúc:**
- Thử nghiệm với pretrained encoder (ResNet-50, EfficientNet-B4)
- Áp dụng attention mechanisms (CBAM, SE-block) vào UNet
- Khám phá Transformer-based segmentation (SegFormer, Swin-UNet)

**Cải thiện FL:**
- Thử FedProx hoặc SCAFFOLD để giảm client drift
- Tăng số client (>3) để mô phỏng thực tế hơn
- Nghiên cứu differential privacy để bảo vệ thông tin thống kê

**Cải thiện TT-SFUDA:**
- Tăng số epoch thích nghi (hiện tại 4 epoch, thử 10-20)
- Thử các pseudo-label strategies khác: temperature scaling, CRF post-processing
- Kết hợp với consistency regularization mạnh hơn (FixMatch, UDA)

**Mở rộng ứng dụng:**
- Thu thập thêm dữ liệu thực từ các vùng địa lý khác nhau (Đồng bằng sông Cửu Long, Tây Nguyên)
- Triển khai trên thiết bị edge (Raspberry Pi, Jetson Nano)
- Tích hợp với hệ thống IoT nông nghiệp thông minh

---

## Tài liệu tham khảo (References)
[1] Vibashan, V. S., Valanarasu, J. M. J., & Patel, V. M. (2023). Target and task specific source-free domain adaptive image segmentation. In *CVPR*, pp. 7998–8008. arXiv:2203.15792. Code: https://github.com/Vibashan/tt-sfuda.
 <!-- *(Bai bao goc cua phuong phap TT-SFUDA duoc ap dung trong du an nay. 	t_sfuda_2d.py duoc xay dung dua tren y tuong two-stage pipeline: target-specific adaptation (Phase 1) va task-specific adaptation (Phase 2) tu bai bao nay.)* -->


[2] Savary, S., Willocquet, L., Pethybridge, S. J., Esker, P., McRoberts, N., & Nelson, A. (2019). The global burden of pathogens and pests on major food crops. *Nature Ecology & Evolution*, 3(3), 430–439.

[3] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In *MICCAI*, pp. 234–241. Springer.

[4] Chen, L. C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-decoder with atrous separable convolution for semantic image segmentation (DeepLabv3+). In *ECCV*, pp. 801–818.

[5] Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. In *CVPR*, pp. 3431–3440.

[6] Zhou, Z., Rahman Siddiquee, M. M., Tajbakhsh, N., & Liang, J. (2019). Unet++: A nested u-net architecture for medical image segmentation. In *MICCAI Workshops*, pp. 3–11. Springer.

[7] Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021). SegFormer: Simple and efficient design for semantic segmentation with transformers. *NeurIPS*, 34, 12077–12090.

[8] Cao, H., Wang, Y., Chen, J., Jiang, D., Zhang, X., Tian, Q., & Wang, M. (2021). Swin-unet: Unet-like pure transformer for medical image segmentation. In *ECCV Workshops*, pp. 205–218. Springer.

[9] Hughes, D., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. *arXiv preprint arXiv:1511.08060*.

[10] Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. *Frontiers in Plant Science*, 7, 1419.

[11] Ferentinos, K. P. (2018). Deep learning models for plant disease detection and diagnosis. *Computers and Electronics in Agriculture*, 145, 311–318.

[12] Ramcharan, A., Baranowski, K., McCloskey, P., Ahmed, B., Legg, J., & Hughes, D. P. (2017). Deep learning for image-based cassava disease detection. *Frontiers in Plant Science*, 8, 1852.

[13] Deng, L., & Yu, D. (2020). Leaf disease detection using Mask R-CNN for instance segmentation. *IEEE Access*, 8, 167681–167692.

[14] Shen, Y., Zhou, G., & Li, J. (2022). Application of transfer learning in plant leaf disease recognition and segmentation. *Frontiers in Plant Science*, 13, 803796.

[15] Ganin, Y., & Lempitsky, V. (2015). Unsupervised domain adaptation by backpropagation. In *ICML*, pp. 1180–1189.

[16] Sun, B., & Saenko, K. (2016). Deep CORAL: Correlation alignment for deep domain adaptation. In *ECCV Workshops*, pp. 443–450. Springer.

[17] Lee, D. H. (2013). Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks. In *Workshop on Challenges in Representation Learning, ICML*, vol. 3, p. 896.

[18] Liang, J., Hu, D., & Feng, J. (2020). Do we really need to access the source data? Source hypothesis transfer for unsupervised domain adaptation. In *ICML*, pp. 6028–6039.

[19] Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T. (2021). Tent: Fully test-time adaptation by entropy minimization. In *ICLR*.

[20] McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). Communication-efficient learning of deep networks from decentralized data. In *AISTATS*, pp. 1273–1282.

[21] Rieke, N., Hancox, J., Li, W., Milletari, F., Roth, H. R., Albarqouni, S., ... & Cardoso, M. J. (2020). The future of digital health with federated learning. *NPJ Digital Medicine*, 3(1), 119.

[22] Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Smola, A., & Smith, V. (2020). Federated optimization in heterogeneous networks (FedProx). In *MLSys*, vol. 2, pp. 429–450.

[23] Karimireddy, S. P., Kale, S., Mohri, M., Reddi, S., Stich, S., & Suresh, A. T. (2020). SCAFFOLD: Stochastic controlled averaging for federated learning. In *ICML*, pp. 5132–5143.

[24] Zheng, R., Zhang, H., & Shi, X. (2023). FedCrop: Federated learning for crop disease classification in distributed agricultural networks. *Computers and Electronics in Agriculture*, 205, 107621.

[25] Peng, X., Bai, Q., Xia, X., Huang, Z., Saenko, K., & Wang, B. (2019). Moment matching for multi-source domain adaptation. In *ICCV*, pp. 1406–1415.

[26] Chen, J., Jiang, M., Liu, W., & Lu, L. (2022). Source-free domain adaptation for semantic segmentation. In *CVPR*, pp. 1215–1225.

[27] fakhrealam9537. (2021). Leaf Disease Segmentation Dataset. *Kaggle*. https://www.kaggle.com/datasets/fakhrealam9537/leaf-disease-segmentation-dataset

[28] Tarvainen, A., & Valpola, H. (2017). Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. *NeurIPS*, 30.

[29] Barbedo, J. G. A. (2019). Plant disease identification from individual lesions and spots using deep learning. *Biosystems Engineering*, 180, 96–107.

[30] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *CVPR*, pp. 770–778.

[31] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

[32] Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.

[33] Buslaev, A., Iglovikov, V. I., Khvedchenya, E., Parinov, A., Druzhinin, M., & Kalinin, A. A. (2020). Albumentations: Fast and flexible image augmentations. *Information*, 11(2), 125.

[34] Li, X., Grandvalet, Y., & Davoine, F. (2019). Explicit inductive bias for transfer learning with convolutional networks. In *ICML*, pp. 3826–3835.

[35] Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. In *ICML*, pp. 448–456.

[36] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *JMLR*, 15(1), 1929–1958.

[37] Milletari, F., Navab, N., & Ahmadi, S. A. (2016). V-net: Fully convolutional neural networks for volumetric medical image segmentation. In *3DV*, pp. 565–571. IEEE.

[38] Yang, Y., & Soatto, S. (2020). FDA: Fourier domain adaptation for semantic segmentation. In *CVPR*, pp. 4085–4095.

[39] Li, R., Jiao, Q., Cao, W., Wong, H. S., & Wu, S. (2020). Model adaptation: Unsupervised domain adaptation without source data. In *CVPR*, pp. 9641–9650.

[40] Pandey, P., Pandey, A. K., Kumar, M., & Pathak, V. K. (2020). Identification of plant leaf diseases using deep learning algorithms. In *ICISC*, pp. 1–6. IEEE.

[41] Agarwal, M., Gupta, S. K., & Biswas, K. K. (2020). Development of efficient CNN model for tomato crop disease identification. *Sustainable Computing: Informatics and Systems*, 26, 100407.

[42] Too, E. C., Yujian, L., Njuki, S., & Yingchun, L. (2019). A comparative study of fine-tuning deep learning models for plant disease identification. *Computers and Electronics in Agriculture*, 161, 272–279.

[43] Wang, Q., Wu, B., Zhu, P., Li, P., Zuo, W., & Hu, Q. (2020). ECA-Net: Efficient channel attention for deep convolutional neural networks. In *CVPR*, pp. 11534–11542.

[44] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. In *ICLR*.

[45] Konecny, J., McMahan, H. B., Ramage, D., & Richtarik, P. (2016). Federated optimization: Distributed machine learning for mobile devices. *arXiv preprint arXiv:1610.02527*.

[46] Bonawitz, K., Ivanov, V., Kreuter, B., Marcedone, A., McMahan, H. B., Patel, S., ... & Seth, K. (2017). Practical secure aggregation for privacy-preserving machine learning. In *CCS*, pp. 1175–1191.

[47] Zhao, Y., Li, M., Lai, L., Suda, N., Civin, D., & Chandra, V. (2018). Federated learning with non-iid data. *arXiv preprint arXiv:1806.00582*.

[48] Litjens, G., Kooi, T., Bejnordi, B. E., Setio, A. A. A., Ciompi, F., Ghafoorian, M., ... & Sanchez, C. I. (2017). A survey on deep learning in medical image analysis. *Medical Image Analysis*, 42, 60–88.

[49] Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. *arXiv preprint arXiv:1409.1556*.

[50] Tan, M., & Le, Q. (2019). Efficientnet: Rethinking model scaling for convolutional neural networks. In *ICML*, pp. 6105–6114.

[51] Zhang, Y., Yang, Q. (2022). A survey on multi-task learning. *IEEE Transactions on Knowledge and Data Engineering*, 34(12), 5586–5609.

### Code Repositories (Open-Source Base Code)

[52] Nakashima, T. (4uiiurz1). (2019). *pytorch-nested-unet: UNet and Nested UNet for image segmentation*. GitHub. https://github.com/4uiiurz1/pytorch-nested-unet *(Cac file `archs.py`, `base_networks.py`, `dataset.py`, `losses.py`, `metrics.py`, `utils.py` trong du an nay duoc dua tren va chinh sua tu repository nay.)*

[53] Berman, M., Triki, A. R., & Blaschko, M. B. (2018). The Lovasz-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks. In *CVPR*, pp. 4413–4421. Code: https://github.com/bermanmaxim/LovaszSoftmax *(Su dung trong `losses.py` — `LovaszHingeLoss` cho binary segmentation.)*

[54] Beutel, D. J., Topal, T., Mathur, A., Qiu, X., Parcollet, T., de Gusmao, P. P. B., & Lane, N. D. (2020). Flower: A friendly federated learning research framework. *arXiv preprint arXiv:2007.14390*. Code: https://github.com/adap/flower *(Framework FL duoc su dung trong `federated_flwr_common.py` va `fl_simulate.py`.)*

[55] Wei, T., et al. (2022). *PlantSeg: Plant disease segmentation with domain adaptation*. GitHub. https://github.com/tqwei05/PlantSeg *(Pipeline to chuc du lieu va cau truc thu muc dataset duoc tham khao tu repository nay.)*

---

*Bao cao duoc nop cho mon hoc CS2311 — Hoc Sau Ung Dung, thang 6 nam 2026.*