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
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;
<p align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;
