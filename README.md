# BACKPROPAGATION
## CHƯƠNG 1 – LÝ THUYẾT VỀ THUẬT TOÁN LAN TRUYỀN NGƯỢC_ FEED FORWARD NEURAL NETWORK 
### 1	Giới thiệu về thuật toán lan truyền ngược
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Thuật toán lan truyền ngược ban đầu được giới thiệu vào những năm 1970, nhưng tầm quan trọng của nó không được đánh giá đầy đủ cho đến khi một bài báo nổi tiếng năm 1986 của David Rumelhart , Geofrey Hinton và Ronald Ưilliams. Bài báo đó mô tả một số mạng nơ-ron mà sự lan truyền ngược hoạt động nhanh hơn nhiều so với các phương pháp tiếp cận học tập trước đây, khiến nó có thể sử dụng mạng thần kinh để giải quyết các vấn đề mà trước đây không thể giải quyết được. Ngày nay, thuật toán lan truyền ngược là công cụ học tập trong mạng nơ-ron. 
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Trung tâm của lan truyền ngược là một biểu thức cho đạo hàm riêng ∂C/ ∂w của hàm chi phí C đối với bất kỳ trọng lượng nào w (hoặc thiên vị b) trong mạng. Biểu thức cho chúng ta biết chi phí thay đổi nhanh như thế nào khi chúng ta thay đổi trọng số và độ lệch. Và trong khi cách diễn đạt có phần phức tạp, nó cũng có một vẻ đẹp riêng, với mỗi yếu tố đều có cách diễn giải tự nhiên, trực quan. Và do đó, lan truyền ngược không chỉ là một thuật toán nhanh để học. Nó thực sự cung cấp cho chúng ta những hiểu biết chi tiết về cách thay đổi trọng số và độ lệch thay đổi hành vi tổng thể của mạng. Điều đó rất đáng để nghiên cứu chi tiết.
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Về cơ bản, thuật toán lan truyền ngược là dạng tổng quát của thuật toán trung bình bình phương tối thiểu (Least Means Square-LMS). Thuật toán này thuộc dạng thuật toán xấp xỉ để tìm các điểm mà tại đó hiệu năng của mạng là tối ưu. Chỉ số tối ưu (performance index) thường được xác định bởi một hàm số của ma trận trọng số và các đầu vào nào đó mà trong quá trình tìm hiểu bài toán đặt ra.
  
### 2	Feed forward neural network
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Kiến trúc mạng truyền thẳng nhiều lớp (Multi-layer Feed Forward - MLFF) là kiến trúc chủ đạo của các mạng nơ-ron hiện tại. Mặc dù có khá nhiều biến thể nhưng đặc trưng của kiến trúc này là cấu trúc và thuật toán học là đơn giản và nhanh (Masters 1993).
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Một mạng truyền thẳng nhiều lớp bao gồm một lớp vào, một lớp ra và một hoặc nhiều lớp ẩn. Các nơron đầu vào thực chất không phải các nơron theo đúng nghĩa, bởi lẽ chúng không thực hiện bất kỳ một tính toán nào trên dữ liệu vào, đơn giản nó chỉ tiếp nhận các dữ liệu vào và chuyển cho các lớp kế tiếp. Các nơron ở lớp ẩn và lớp ra mới thực sự thực hiện các tính toán, kết quả được định dạng bởi hàm đầu ra (hàm chuyển). Cụm từ “truyền thẳng” (feed forward) (không phải là trái nghĩa của lan truyền ngược) liên quan đến một thực tế là tất cả các nơron chỉ có thể được kết nối với nhau theo một hướng: tới một hay nhiều các nơron khác trong lớp kế tiếp (loại trừ các nơron ở lớp ra)
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116341718-9900b880-a80b-11eb-800d-42d112e96984.png" width="50%"/>

<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Trong đó: 
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;P: Vector đầu vào (vector cột) 
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Wi: Ma trận trọng số của các nơron lớp thứ i. 
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;(SixRi: S hàng (nơron) - R cột (số đầu vào)) 
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;bi: Vector độ lệch (bias) của lớp thứ i 
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;(Six1: cho S nơ-ron) 
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;ni: net input (Six1) 
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;fi: Hàm chuyển (hàm kích hoạt) 
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;ai: net output (Six1) 
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;⊕: Hàm tổng thông thường.
  
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Một mạng truyền thẳng là một mạng lưới thần kinh nhân tạo trong đó các kết nối giữa các nút làm không tạo chu kỳ. Như vậy, nó khác với hậu duệ của nó: Mạng nơ-tron tái phát.
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Mạng nơ-tron truyền tiếp là loại mạng nơ-ron nhân tạo đầu tiên và đơn giản nhất được phát minh ra. Trong mạng này, thông tin chỉ di chuyển theo một hướng chuyển tiếp từ các nút đầu vào, qua các nút ẩn (nếu có) và đến các nút đầu ra. Không có chu kỳ hoặc vòng lặp trong mạng. 

### 3	Cách tiếp cận
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Cấu trúc cơ bản
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116341916-f5fc6e80-a80b-11eb-9918-6c6a94f45e2a.png" width="50%"/>
  
#### Dựa trên ma trận nhanh để tính toán đầu ra từ mạng nơ-tron
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Là một cách tốt để làm quen với ký hiệu được sử dụng trong lan truyền ngược, trong một ngữ cảnh quen thuộc.
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Sử dụng wljk để biểu thị trọng lượng cho kết nối từ kth nơ-ron trong lớp ( l - 1)th đến jth nơ-ron trong lớp lth. Vì vậy, ví dụ biểu đồ dưới đây cho thấy trọng số của kết nối từ nơ-ron thứ tư trong lớp thứ hai đến nơ-ron thứ hai trong lớp thứ ba của mạng:
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116342295-8fc41b80-a80c-11eb-910c-3642a0709e61.png" width="50%"/>
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Ví dụ về các ký hiệu đang được sử dụng
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116342301-918ddf00-a80c-11eb-8fd0-320b7fa2a3e7.png" width="50%"/>
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116343614-f9ddc000-a80e-11eb-920c-bb4875e98464.PNG" width="100%"/>
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116342305-92bf0c00-a80c-11eb-9f66-deef82857c25.png" width="50%"/>
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116342859-838c8e00-a80d-11eb-88ca-c8efbe9dded6.PNG" width="100%"/>
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Áp dụng một hàm chẳng hạn như σ cho mọi phần tử trong một vectơ v. Sử dụng ký hiệu rõ ràng σ ( v ) để biểu thị loại ứng dụng nguyên tố của một hàm. Đó là các thành phần của σ ( v ) chỉ là σ( v)j= σ(vj). Ví dụ, nếu có hàm f( x ) x2 thì hình thức vectơ hóa của f có tác dụng
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116342310-93f03900-a80c-11eb-85c7-295e347e1581.png" width="50%"/>
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Nghĩa là, vectơ hóa f chỉ bình phương mọi phần tử của vectơ.
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Với những ký hiệu này, phương trình có thể được viết lại ở dạng vectơ đẹp và nhỏ gọn
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116342315-95b9fc80-a80c-11eb-9973-0c092f7bf252.png" width="50%"/>
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116343042-db2af980-a80d-11eb-950b-d11c4e822885.PNG" width=100%"/>

### 4	Hai giả định về hàm chi phí ( C )
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Mục tiêu của lan truyền ngược là tính toán các đạo hàm riêng ∂C/ ∂w và ∂C/ ∂b của hàm chi phí C đối với bất kỳ trọng lượng nào w hoặc thiên vị b trong mạng. Để lan truyền ngược hoạt động, hai giả thiết chính về dạng của hàm chi phí. Tuy nhiên, trước khi nêu những giả định đó, nên lưu ý đến một hàm chi phí mẫu. Trong ký hiệu của phần cuối cùng, chi phí bậc hai có dạng
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116343975-ae77e180-a80f-11eb-86b7-4a96e306b2a8.png" width="50%"/>
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116344085-d6ffdb80-a80f-11eb-858c-cf8eb017efde.PNG" width=100%"/>
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116344190-057db680-a810-11eb-8486-3901d6cff1ef.PNG" width=100%"/>
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116344306-378f1880-a810-11eb-87db-b2d042a74b9b.png" width="50%"/>
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Ví dụ, hàm chi phí bậc hai đáp ứng yêu cầu này, vì chi phí bậc hai cho một ví dụ đào tạo duy nhất x có thể được viết là:
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116344309-3827af00-a810-11eb-9b4e-7ad23d5e4897.png" width="50%"/>
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Do đó là một chức năng của các kích hoạt đầu ra. Tất nhiên, hàm chi phí này cũng phụ thuộc vào sản lượng mong muốn y. Tuy nhiên, vì ví dụ đào tạo đầu vào x được cố định, và do đó đầu ra y cũng là một tham số cố định. Đặc biệt, nó không phải là thứ có thể sửa đổi bằng cách thay đổi trọng số và độ lệch theo bất kỳ cách nào tức là nó không phải là thứ mà mạng nơ-ron học được. Và do đó nó có ý nghĩa khi coi C như một chức năng của các kích hoạt đầu ra a^L một mình, với  y chỉ đơn thuần là một tham số giúp xác định chức năng đó.
  
### 5	Sản phẩm Hadamard 
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Thuật toán lan truyền ngược dựa trên các phép toán đại số tuyến tính phổ biến - những thứ như phép cộng vectơ, nhân một vectơ với ma trận, v.v. Nhưng một trong những thao tác ít được sử dụng hơn một chút. Đặc biệt, giả sử S và t là hai vectơ cùng chiều.  Sau đó, chúng tôi sử dụngs S⊙t để biểu thị tích nguyên tố của hai vectơ. Do đó, các thành phần của S⊙t chỉ là ( s ⊙ t)j=Sjtj. Ví dụ:
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116344314-39f17280-a810-11eb-84c3-16eccca460f1.png" width="50%"/>
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Loại phép nhân theo từng nguyên tố này đôi khi được gọi là tích Hadamard hoặc sản phẩm Schur. Được gọi nó là sản phẩm Hadamard. Các thư viện ma trận tốt thường cung cấp các triển khai nhanh chóng của sản phẩm Hadamard và điều đó rất hữu ích khi thực hiện lan truyền ngược. 
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116344317-3bbb3600-a810-11eb-923a-9fec08860000.png" width="100%"/>
  
### 6	Bốn phương trình cơ bản 
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116344322-3cec6300-a810-11eb-8ddc-c045eaffbcde.png" width="50%"/>
  
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116344969-6659be80-a811-11eb-916b-5c07a467c3e0.PNG" width="100%"/>
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Để hiểu cách xác định lỗi, hãy tưởng tượng có một con quỷ trong mạng nơ-ron của chúng ta:
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116344327-3d84f980-a810-11eb-8a81-428c4cd85e21.png" width="50%"/>
  
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116345062-a28d1f00-a811-11eb-8e53-97a2d40fb3df.PNG" width="100%"/>
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116344328-3eb62680-a810-11eb-879f-1cd401839dd1.png" width="50%"/>
  
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116345183-de27e900-a811-11eb-8c5c-80b19a5089d4.PNG" width="100%"/>
 
#### Kế hoạch tấn công
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Sự lan truyền ngược dựa trên bốn phương trình cơ bản. Cùng với nhau, những phương trình đó cung cấp cho một cách tính toán cả lỗi δ^l và gradient của hàm chi phí. Bốn phương trình dưới đây. Trên thực tế, các phương trình lan truyền ngược rất phong phú nên việc hiểu rõ chúng đòi hỏi thời gian và sự kiên nhẫn đáng kể, dần dần nghiên cứu sâu hơn về các phương trình. Tin tốt là sự kiên nhẫn đó được đền đáp nhiều lần.
  
#### Một chương trình cho lỗi trong lớp đầu ra δ^L
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Các thành phần của δ^L được đưa ra bởi
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116344333-41b11700-a810-11eb-8e6f-47ca27d89ba9.png" width="50%"/>
  
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116345432-4bd41500-a812-11eb-8260-c8d4aa757418.PNG" width="100%"/>
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Phương trình (BP1) là một biểu thức thành phần cho δ^L. Đó là một biểu thức hoàn toàn tốt, nhưng không phải là dạng dựa trên ma trận mà chúng ta muốn cho việc lan truyền ngược. Tuy nhiên, thật dễ dàng để viết lại phương trình ở dạng dựa trên ma trận như
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116344339-45449e00-a810-11eb-8728-710815f07781.png" width="50%"/>
  
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116345553-8fc71a00-a812-11eb-9559-ce8f3b9bab15.PNG" width="100%"/>
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Như ta có thể thấy, mọi thứ trong biểu thức này đều có dạng vector đẹp mắt và dễ dàng tính toán bằng thư viện như Numpy.

#### Một phương trình cho lỗi δ^L về lỗi trong lớp tiếp theo δ^L
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116344347-47a6f800-a810-11eb-88c0-0ca5314bbbe6.png" width="50%"/>
  
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116345696-dae12d00-a812-11eb-9a4d-040c750aa313.PNG" width="100%"/>

#### Một phương trình cho tốc độ thay đổi của chi phí đối với bất kỳ sai lệch nào trong mạng:
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116344351-483f8e80-a810-11eb-9aaf-dfa712d422cf.png" width="50%"/>
  
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116345901-3f9c8780-a813-11eb-9294-004c5b394cd8.PNG" width="100%"/>
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Nơi nó được hiểu rằng δ đang được đánh giá ở cùng một tế bào thần kinh với độ lệch b.
  
#### Một phương trình cho tốc độ thay đổi của chi phí đối với bất kỳ trọng số nào trong mạng:
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116346035-92763f00-a813-11eb-8692-13b67171a62e.PNG" width="100%"/>
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116346154-db2df800-a813-11eb-820b-edb2c0dce3e6.PNG" width="100%"/>
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;--> Tổng kết lại, một trọng số sẽ học chậm nếu nơ-ron đầu vào kích hoạt thấp hoặc nếu nơ-ron đầu ra đã bão hòa, tức là kích hoạt cao hoặc thấp.
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Không có quan sát nào trong số này là quá đáng ngạc nhiên. Tuy nhiên, chúng giúp cải thiện mô hình tinh thần về những gì đang diễn ra khi mạng thần kinh học hỏi. Hơn nữa, có thể xoay chuyển kiểu lập luận này. Bốn phương trình cơ bản hóa ra phù hợp với bất kỳ hàm kích hoạt nào, không chỉ hàm sigmoid tiêu chuẩn. Và vì vậy có thể sử dụng các phương trình này để thiết kế các hàm kích hoạt có các thuộc tính học mong muốn cụ thể. Ví dụ để cung cấp cho bạn ý tưởng, giả sử chọn một chức năng kích hoạt (không phải sigmoid) σ vậy nên σ′ luôn luôn tích cực và không bao giờ gần bằng không. Điều đó sẽ ngăn chặn quá trình học tập chậm lại xảy ra khi các tế bào thần kinh sigmoid bình thường bão hòa. Ghi nhớ bốn phương trình (BP1) - (BP4) có thể giúp giải thích lý do tại sao những sửa đổi như vậy được thử và chúng có thể có tác động gì.

<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116344369-4fff3300-a810-11eb-9415-7c7430901906.png" width="50%"/>

### 6 Vấn đề
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;(1) Chứng minh rằng (BP1) có thể viết lại thành
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116346403-64ddc580-a814-11eb-8f3a-458498506d1d.PNG" width="100%"/>

## CHƯƠNG 2 – CHỨNG MINH PHƯƠNG TRÌNH
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Cả bốn đều là hệ quả của quy tắc chuỗi từ phép tính đa biến.

### 1	Phương trình (BP1)
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Chứng minh (BP1)
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116346531-aa01f780-a814-11eb-863d-2c0b0f2cb91e.jpg" width="50%"/>
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116346532-aa9a8e00-a814-11eb-8e95-7e071ab56a6d.jpg" width="50%"/>
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Cho một biểu thức cho lỗi đầu ra δ^L. Để chứng minh đẳng thức này, hãy nhớ lại điều đó theo định nghĩa
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116346718-fcdbaf00-a814-11eb-9a40-449f8aae1406.PNG" width="100%"/>

### 2 Chứng minh (BP2)

<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116346886-4f1cd000-a815-11eb-97b2-611d6d7ff7d9.PNG" width="100%"/>
  
## CHƯƠNG 3: THUẬT TOÁN LAN TRUYỀN NGƯỢC_GRADIENT DESCENT
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116346961-770c3380-a815-11eb-85e5-e48632f76023.PNG" width="100%"/>
  
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116347056-a753d200-a815-11eb-952b-60114e1cfa97.PNG" width="100%"/>
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Quy tắc bỏ túi cần nhớ là chiều của hai ma trận ở hai vế phải như nhau.
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Đạo hàm theo một ma trận của một hàm số nhận giá trị thực (scalar) sẽ có chiều bằng với chiều của ma trận đó!
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116347179-ed109a80-a815-11eb-88ad-2522c8495f3a.PNG" width="100%"/>
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Cập nhật đạo hàm cho ma trận trọng số và vector biases:
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116347290-23e6b080-a816-11eb-92dd-625743bc2716.png" width="50%"/>
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Kiểm tra thuật toán, có thể thấy lý do tại sao nó được gọi là lan truyền ngược . Tính toán các vectơ lỗi δ^l lùi lại, bắt đầu từ lớp cuối cùng. Có vẻ kỳ lạ là ta đang trải qua mạng lạc hậu. Nhưng nếu bạn nghĩ về bằng chứng của sự lan truyền ngược, sự di chuyển ngược lại là hệ quả của thực tế rằng chi phí là một hàm của kết quả đầu ra từ mạng. Để hiểu chi phí thay đổi như thế nào với các trọng số và độ lệch trước đó, cần áp dụng nhiều lần quy tắc chuỗi, làm việc ngược lại qua các lớp để thu được các biểu thức có thể sử dụng được.
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Backpropagation: bức tranh lớn
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116347642-c868f280-a816-11eb-8128-0ec3f51a8fbe.PNG" width="100%"/>
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Sự thay đổi trọng lượng đó sẽ gây ra sự thay đổi trong kích hoạt đầu ra từ nơ-ron tương ứng:
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116347759-0403bc80-a817-11eb-92d1-cb2533da3a4d.png" width="50%"/>
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Điều đó đến lượt nó sẽ gây ra sự thay đổi trong tất cả các kích hoạt trong lớp tiếp theo:
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116347761-0534e980-a817-11eb-9148-d30be09cc347.png" width="50%"/>
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Những thay đổi đó sẽ lần lượt gây ra những thay đổi trong lớp tiếp theo và sau đó là lớp tiếp theo và cứ tiếp tục như vậy để gây ra thay đổi trong lớp cuối cùng và sau đó trong hàm chi phí:
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116347763-06661680-a817-11eb-8179-80133db35872.png" width="50%"/>
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116347988-71afe880-a817-11eb-9004-acce289f1169.PNG" width="100%"/>
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116348132-b50a5700-a817-11eb-959f-7b44ffbfe1d0.png" width="50%"/>

<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116348243-ee42c700-a817-11eb-8476-651d5c3f9d94.PNG" width="100%"/>
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116348334-21855600-a818-11eb-9a2b-1a45e7b0356d.PNG" width="100%"/>
  
## CHƯƠNG 4: CHƯƠNG TRÌNH VÍ DỤ
### 1	Bài toán áp dụng
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116348840-0535e900-a819-11eb-88cc-fdad4bf01be8.png" width="50%"/>
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Trong hình trên, đầu vào được chuyển đổi đầu tiên thông qua lớp ẩn 1, sau đó là lớp thứ hai và cuối cùng là đầu ra được dự đoán.  Mỗi phép biến đổi được điều khiển bởi một tập hợp các trọng số (và độ lệch).
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;Sử dụng gradient descent như một thuật toán tối ưu hóa, trọng số được cập nhật ở mỗi lần lặp là:
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116348844-08c97000-a819-11eb-8600-d4f314107d95.png" width="50%"/>
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;trong đó L là hàm mất mát và ϵ là tốc độ học
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116348872-1121ab00-a819-11eb-95d2-28cbf216c64f.png" width="50%"/>
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116348875-11ba4180-a819-11eb-94a2-dc0b5ed6752b.png" width="50%"/>
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116348882-14b53200-a819-11eb-9b27-6f535f0ff55c.png" width="50%"/>
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116348888-167ef580-a819-11eb-8d1f-1913995d4171.png" width="50%"/>
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116348889-17178c00-a819-11eb-8b9d-51c5ca260b6f.png" width="50%"/>
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116348904-1aab1300-a819-11eb-9426-91a6c7c7b417.png" width="50%"/>
<p align="center"> <img src ="https://user-images.githubusercontent.com/77925421/116348911-1bdc4000-a819-11eb-9f46-46487000c51a.png" width="50%"/>
  
##  TÀI LIỆU THAM KHẢO
&nbsp;&nbsp;&nbsp;&nbsp;1.	http://neuralnetworksanddeeplearning.com/chap2.html
&nbsp;&nbsp;&nbsp;&nbsp;2.	https://www.kaggle.com/romaintha/an-introduction-to-backpropagation
&nbsp;&nbsp;&nbsp;&nbsp;3.	https://123doc.net/document/706333-mang-noron-truyen-thang-va-thuat-toan-lan-truyen-nguoc.htm

