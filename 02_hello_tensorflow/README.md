## Dự án 01: Xây dựng Raspberry PI thành máy tính cho Data Scientist (PIDS)

## Bài 02. Hello, TensorFlow

##### Người soạn: Dương Trần Hà Phương

##### Website: [Mechasolution Việt Nam](https://mechasolution.vn)

##### Email: mechasolutionvietnam@gmail.com

---

TensorFlow là một hệ thống chuyên dùng để tính toán trên đồ thị (graph-based computation). Một ví dụ điển hình là sử dụng trong máy học (machine learning). Trong notebook này, chúng ta sẽ tìm hiểu những khái niệm cơ bản cuả TensorFlow và một số ví dụ đơn giản.

Cái tên TensorFlow được kết hợp từ 2 thành phần:

- [**Tensor**](https://en.wikipedia.org/wiki/Tensor), là một mảng có số chiều tùy ý. Một vector là một mảng 1 chiều (1-d array) thì được hiểu là tensor bậc 1. Một vector là một mảng 2 chiều (2-d array) thì được hiểu là tensor bậc 2.
- **Flow** được hiểu nhưng luồng xử lý trong đồ thị.

Khi bạn nghĩ sẽ làm thứ gì đó với TensorFlow, bạn có thể làm các bước sau:

- Tạo tensor (như ma trận)
- Thêm các thao tác tính toán (các phép tính trên tensor)
- Cuối cùng là thực thi việc tính toán (chạy đồ thị tính toán)

Cụ thể, khi bạn thêm các thao tác tính toán trên tensor, nó sẽ không thực thi ngay lập tức. Thay vào đó, TensorFlow sẽ đợi bạn định nghĩa tất cả các thao tác bạn muốn thực hiện trên đồ thị tính toán. Sau đó, TensorFlow tối ưu đồ thị tính toán, quyết định cách thức thực thi đồ thị tính toán trên, trước khi sinh ra dữ liệu. Bởi vì vậy, một tensor trong TensorFlow không giữ quá nhiều dữ liệu như một nơi lưu trữ dữ liệu mà nó sẽ đợi dữ liệu đến khi việc tính toán được thực thi.

## Cộng hai vector trong TensorFlow

Hãy bắt đầu với một vài ví dụ đơn giản. Hãy cộng 2 vector có 4 phần tử (2 tensor bậc 1) như sau:

[ 1.0 1.0 1.0 1.0] + [ 2.0 2.0 2.0 2.0 ] = [ 3.0 3.0 3.0 3.0 ]

```python
import tensorflow as tf

with tf.Session():
    input1 = tf.constant([1.0, 1.0, 1.0, 1.0])
    input2 = tf.constant([2.0, 2.0, 2.0, 2.0])
    output = tf.add(input1, input2)
    result = output.eval()
    print(result)
```

    [3. 3. 3. 3.]

## Giải thích ví dụ cộng 2 vector trong TensorFlow

Ví dụ trên nhìn có vẻ đơn giản, chỉ là cộng 2 vectors thôi mà ! Nhưng thực ra bản chất bên trong bao gồm nhiều thứ hơn chúng ta thấy, vì vậy hãy cũng đi sâu vào ví dụ trên để tìm hiểu.

> `import tensorflow as tf`

Import TensorFlow API vào IPython runtime environment.

> `with tf.Session():`

Khi bạn chạy một thao tác trên TensorFlow, bạn cần thực hiện thao tác đó trên một `Session`. Một session chứa đồ thị tính toán, là nơi chứa các tensors và các thao tác. Khi bạn tạo các tensors và các thao tác, nó sẽ không thực thi ngay lập tức, nhưng nó sẽ đợi các thao tác khác, các tensors khác được thêm vào đồ thị và chỉ thực thi khi được yêu cầu thực thi của session, cuối cùng là cho ra kết quả. Việc trì hoãn thực thi như trên cho ta có cơ hội bổ sung cho việc xử lý song song và tối ưu hóa. TensorFlow có thể quyết định cách kết hợp các thao tác và nơi chạy chúng sau khi TensorFlow biết về tất cả các thao tác.

> > `input1 = tf.constant([1.0, 1.0, 1.0, 1.0])`

> > `input2 = tf.constant([2.0, 2.0, 2.0, 2.0])`

Hai dòng tiếp theo, tạo tensor bằng cách sử dụng hàm `constant`, hàm này tương tự như `array` và `full` trong numpy. Tóm lại, hàm trên sẽ tạo một tensor với một shape quy định trước và điền các giá trị vào.
Cấu trúc của hàm `constant`:

```
tf.constant(
    value,
    dtype=None,
    shape=None,
    name='Const',
    verify_shape=False
)
```

> > `output = tf.add(input1, input2)`

Ở dòng code này, bạn có thể nghĩ hàm `add` sẽ cộng 2 vectors ngay lập tức, nhưng thực ra nó không phải như vậy. Nó chỉ thêm thao tác cộng vào đồ thị tính toán, nhưng đồ thị tính toán vẫn chưa thực thi ngay lúc này.

> > `result = output.eval()`

> > `print(result)`

`eval()` có vẻ đơn giản hơn sự phức tạp mà ta nhìn thấy ở trên. Nó sẽ lấy giá trị của vector (tensor) và trả về một numpy array, cái mà ta có thể in ra để kiểm tra. Nhưng nó thực sự quan trong vì ta sẽ thực thi đồ thị tính toán tại đây; để có nó, ta phải chạy đồ thị tính toán.

Vì vậy, tại điểm này, đồ thị tính toán sẽ được thực thi, không phải khi thao tác `add` được gọi mà khi ta đã thêm thao tác `add` vào đồ thị tính toán.

## Tổ hợp nhiều thao tác

Để sử dụng TensorFlow, bạn cần thêm các thao tác và các tensors vào đồ thị tính toán. Sau đó, thực thi đồ thị tính toán để chạy tất cả các thao tác và tính toán các giá trị của tất cả tensors trong đồ thị.

Dưới đây là ví dụ đơn giản với 2 thao tác:

```python
import tensorflow as tf

with tf.Session():
    input1 = tf.constant(1.0, shape=[4])
    input2 = tf.constant(2.0, shape=[4])
    input3 = tf.constant(3.0, shape=[4])
    output = tf.add(tf.add(input1, input2), input3)
    result = output.eval()
    print(result)
```

    [6. 6. 6. 6.]

Ở ví dụ này, hàm `constant` được sử dụng tương tự hàm `fill` của numpy.

Toán tử `add` hỗ trợ operator overloading, vì vậy bạn có thể viết một cách đơn giản hơn là: `input1 + input2`. Xem ví dụ sau:

```python
with tf.Session():
    input1 = tf.constant(1.0, shape=[4])
    input2 = tf.constant(2.0, shape=[4])
    output = input1 + input2
    print(output.eval())
```

    [3. 3. 3. 3.]

## Cộng hai ma trận

Next, let's do something very similar, adding two matrices:

```python
import tensorflow as tf
import numpy as np

with tf.Session():
    input1 = tf.constant(1.0, shape=[2, 3])
    input2 = tf.constant(np.reshape(np.arange(1.0, 7.0, dtype=np.float32), (2, 3)))
    output = tf.add(input1, input2)
    print(output.eval())
```

    [[2. 3. 4.]
     [5. 6. 7.]]

Trong ví dụ này, ma trận với giá trị từ 1 đến 6 được tạo bằng thư viện numpy và gán vào trong `constant`, nhưng TensorFlow cũng có các toán tử `range`, `reshape`, và `tofloat`. Sử dụng các toán tử trên với TensorFlow sẽ hiệu quả hơn nếu làm việc với các ma trận lớn.

## Nhân hai ma trận

Tiếp theo, chúng ta sẽ tìm hiểu việc nhân 2 ma trận. Chọn một vector gồm 0, 1 và ma trận chứa một vài giá trị ngẫu nhiên, đây là một thao tác quan trọng mà chúng ta cần dùng cho các bài toán như Regression và Neural networks.

```python
#@test {"output": "ignore"}
import tensorflow as tf
import numpy as np

with tf.Session():
    input_features = tf.constant(np.reshape([1, 0, 0, 1], (1, 4)).astype(np.float32))
    weights = tf.constant(np.random.randn(4, 2).astype(np.float32))
    output = tf.matmul(input_features, weights)
    print("Input:")
    print(input_features.eval())
    print("Weights:")
    print(weights.eval())
    print("Output:")
    print(output.eval())
```

    Input:
    [[1. 0. 0. 1.]]
    Weights:
    [[-0.02770738 -2.6773098 ]
     [-0.29421392 -1.3019267 ]
     [ 1.1246065  -1.4934174 ]
     [-0.6606843   1.4383485 ]]
    Output:
    [[-0.6883917 -1.2389612]]

Ví dụ trên, chúng ta dùng (1 x 4) vector [1 0 0 1] và nhân với một ma rận (4 x 2) với các giá trị được sinh ra từ phân phổi chuẩn (mean 0, stdev 1). Vì vậy, ma trận kết quả sẽ là ma trận (1 x 2).

Bạn có thể thử hiệu chỉnh ví dụ trên. Chạy cell trên nhiều lần thì ma trận weights sẽ được sinh ra mới liên tục và ta sẽ thu được các ma trận kết quả khác nhau. Hoặc thay đổi input thành \[0 0 0 1], và chạy lại cell trên một nữa. Hoặc cố gắng khởi tạo ma trận weights mà sử dụng các TensorFlow ops như: [random_normal](https://www.tensorflow.org/api_docs/python/tf/random_normal), thay cho việc sử dụng thư viện numpy.

## Variables

Xét việc cộng 2 ma trận trong một vòng lặp, chúng ta sẽ không tạo ra một tensor mới ở mỗi lần lập, mà chúng ta sẽ cập nhật những giá trị liên tục và sau đó sẽ chạy lại đồ thị tính toán với dữ liệu mới vừa được cập nhật. Điều đó xảy ra rất nhiều với các mô hình trong machine learning, chúng ta có thể thay đổi nhiều tham số ở mỗi lần lặp ví dụ như thuật toán Gradient Descent cập nhật các trọng số và sau đó thực hiện các tính toán một lần nữa và cứ thế lặp lại.

```python
#@test {"output": "ignore"}
import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    # Set up two variables, total and weights, that we'll change repeatedly.
    total = tf.Variable(tf.zeros([1, 2]))
    weights = tf.Variable(tf.random_uniform([1,2]))

    # Initialize the variables we defined above.
    tf.initialize_all_variables().run()

    # This only adds the operators to the graph right now. The assignment
    # and addition operations are not performed yet.
    update_weights = tf.assign(weights, tf.random_uniform([1, 2], -1.0, 1.0))
    update_total = tf.assign(total, tf.add(total, weights))

    for _ in range(10):
        # Actually run the operation graph, so randomly generate weights and then
        # add them into the total. Order does matter here. We need to update
        # the weights before updating the total.
        sess.run(update_weights)
        sess.run(update_total)

        print(weights.eval(), total.eval())
```

    [[ 0.8276067 -0.5568018]] [[ 0.8276067 -0.5568018]]
    [[0.7663796  0.23061371]] [[ 1.5939863 -0.3261881]]
    [[ 0.73993087 -0.58519435]] [[ 2.3339171  -0.91138244]]
    [[ 0.62350607 -0.6924634 ]] [[ 2.9574232 -1.6038458]]
    [[-0.01303411  0.88936186]] [[ 2.944389 -0.714484]]
    [[-0.69019794 -0.36287642]] [[ 2.2541912 -1.0773604]]
    [[0.73608446 0.922559  ]] [[ 2.9902756  -0.15480137]]
    [[ 0.95856285 -0.30964684]] [[ 3.9488385 -0.4644482]]
    [[-0.8023789  -0.41767144]] [[ 3.1464596  -0.88211966]]
    [[-0.37029266 -0.94749045]] [[ 2.776167  -1.8296101]]

Ví dụ trên có vẻ phức tạp hơn. Ở mức độ cao hơn, chúng ta tạo ra hai biến và thêm các thao tác lên chúng. Sau đó, trong vòng lặp, liên tục thực hiện những thao tác này. Hãy tìm hiểu ví dụ trên từng bước một (step-by-step):

- Đầu tiên, code trên tạo 2 biến `total` và `weights`. `total` được khởi tạo \[0, 0\] và `weights` được khởi tạo với các giá trị ngẫu nhiên trong [-1, 1].

- Tiếp theo, hai thao tác được thêm vào đồ thị, một để cập nhật biến weights với các giá trị được lấy ngẫu nhiên trong [-1, 1], thao tác còn lại để cập nhật tổng với giá trị trọng số (weight) mới. Một lần nữa, các toán tử sẽ không được thực thi ngay cho đến khi chúng ta gọi hàm `eval`.

- Cuối cùng, trong vòng lặp for, chúng ta chạy 2 thao tác trên. Trong mỗi lần lặp, ta sẽ thực thi thao tác mà ta đã thêm vào trước đó. Đầu tiên gán những giá trị ngẫu nhiên vào `weights`, sau đó cập nhật tổng với giá trị `weights` mới.

Thật khó để chúng ta có thể biết được chính xác khi nào các thao tác được thực thi. Do đó, điều quan trọng chúng ta cần nhớ là các thao tác chỉ thực hiện khi có yêu cầu.

[Biến (Variable)](https://www.tensorflow.org/api_docs/python/tf/Variable) có thể sử dụng một cách hữu ích trong các trường hợp có nhiều tính toán và dữ liệu của bạn muốn sử dụng nhiều lần với sự thay đổi không đáng kế ở mỗi lần.

---

Nếu có thắc mắc hoặc góp ý, các bạn hãy comment bên dưới để bài viết có thể được hoàn thiện hơn.
Xin cảm ơn,

Hà Phương - Mechasolution Việt Nam.
