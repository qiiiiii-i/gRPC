# gRPC通信
这是一个基于gRPC的客户端与服务端通信项目示例。本项目展示了如何使用gRPC完成客户端发送指令、服务端接收指令并且返回结果给客户端的一系列操作，并以辐射源个体识别为主要内容展示了通信的全过程。
## 项目结构
本项目的文件结构如下所示：
- **/client.py**：包含客户端代码，包括与服务端的gRPC通信逻辑及配置文件。
- **/sei.py**：包含服务端代码，包括gRPC服务实现、处理客户端请求、以及使用训练好的模型进行辐射源分类的代码。
- **/sei.proto**：存放gRPC协议文件，定义了服务端和客户端之间的通信格式。
- **/sei_pb2.py**：使用sei.proto生成的文件
- **/sei_pb2_grpc.py**:使用sei.proto生成的文件

## 技术栈
- **gRPC**：高性能、开源和通用的RPC框架，用于客户端与服务端之间的通信。
- **Protocol Buffers**：gRPC默认使用的序列化协议，用于定义服务和消息格式。
- **Python**：本项目使用的编程语言（推荐Python3.9）
- **PyTorch**：用于训练和部署辐射源分类模型的深度学习框架。
- **protobuf**：定义服务和消息的数据交换格式，确保客户端和服务端能正确交换信息。

## 功能概述
本项目实现了客户端与服务端的通信，具体功能如下：

1. **客户端功能**：
   - 向服务端发送包含辐射源相关特征数据的请求（文件地址、采样率、带宽、符号速率、中心频率）。
   - 发送指令，要求服务端使用训练好的模型进行辐射源分类。
   - 接收服务端返回的分类结果，并展示。

2. **服务端功能**：
   - 接收客户端发送的请求，解析其中的辐射源特征数据。
   - 使用训练好的分类模型，对辐射源数据进行分类。
   - 返回分类结果给客户端，提供分类标签。

## 安装与配置
### 1、安装依赖
安装gRPC和相关的python库。
```bash
# 安装gRPC的Python库
pip install grpcio
# 安装用于生成gRPC代码的工具
pip install grpcio-tools
```
### 2、生成gRPC代码  
- 直接下载整个仓库中的文件，配置好相关模型后，直接运行整个项目（先启动服务端，再启动客户端）。  
- 下载本仓库中提供的`sei.proto`文件，将其放置在`/proto`目录下，利用安装好的`grpcio-tools`来生成Python的gRPC客户端和服务端代码，操作如下：  
   ```bash
    python -m grpc_tools.protoc --proto_path=proto --python_out=. --grpc_python_out=. proto/sei.proto
    ```
    此命令会生成`sei_pb2.py`和`sei_pb2_grpc.py`文件，用于客户端和服务端的实现。  
- 根据需求编写`proto`文件  
    - 创建`.proto`文件
    - 定义语法版本，常见的是`proto3`
        ```proto
        syntax = "proto3";
        ```
    - 定义客户端消息类型，每个消息字段都要指定类型和字段编号（唯一），代表从客户端得到的消息，其中repeated代表将IdenParam嵌套在IdenReq中并可重复，支持批量处理消息
        ```proto
        message IdenParam {
            double               sample_rate        = 2;     // 采样率 Hz
            double               band_width         = 3;     // 带宽 Hz
            double               symbol_rate        = 4;     // 符号速率 Bd
            double               center_freq        = 5;     // 信号频偏 Hz
            string               file_path          = 6;     // 文件路径，默认为wav双通道16bit短整型格式
       }
       message IdenReq {
           repeated IdenParam   params            = 1;     // 识别参数，支持批量识别
       }
        ```
    - 定义服务端消息类型，代表服务端返回的结果
        ```proto
        message IdenResult {
            int32           code    = 1;    // 执行结果代码，0表示成功，其他表示失败（详细代码根据实际需求确定）
            string          message = 2;    // 执行结果的消息字段，一般用于说明执行成功或出错的原因
            string          object  = 3;    // 识别结果，表示识别目标
        }
        message IdenRes {
            repeated IdenResult  results           = 1;     // 识别结果，和IdenReq中的params一一对应
        }
        ```
    - 定义服务接口（服务端和客户端都需要）
       ```proto
       service IdenService {
          // 个体识别
          rpc Identify(IdenReq) returns(IdenRes);
       }
       ```
    根据需求完成上述编写后，同样可以使用`grpcio-tools`生成`.py`文件用于客户端与服务端通信。
### 3、完善`sei.py`服务端代码
- 本项目涉及模型的加载，确保将已经训练好的模型文件放在`/weights`目录下，将模型文件放在与`sei.py`文件同目录下，根据需求修改模型加载和推理部分即可。  
- 服务端代码中，最关键的是设定监听端口，才能接收到从客户端发送的信息，相关代码如下：
  ```bash
   def serve():
       server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
       sei_pb2_grpc.add_IdenServiceServicer_to_server(IdenService(), server)

       # 监听端口50051
       server.add_insecure_port('[::]:50051')
       server.start()
       print("Server is listening on port 50051")
       server.wait_for_termination()
  ```
  该代码定义了监听的端口，在该端口下接受来自客户端的指令，并使用自定义的`IdenService`函数来对指令进行处理。  
- 接收到指令后，`IdenService`函数处理指令，并得到处理结果，存储在`result`当中，使用以下指令向客户端返回结果：
  ```bash
  response = sei_pb2.IdenRes(results=results)
  ```
  
### 4、完善`client.py`客户端代码
本仓库中的`client.py`提供了一个客户端代码的示例，其中最关键的部分在于与服务端建立连接，相关代码如下：
```bash
# 与服务端建立连接
channel = grpc.insecure_channel('localhost:50051')  # 服务端地址和端口
stub = sei_pb2_grpc.IdenServiceStub(channel)
```
根据客户端需求修改好`request`，使用以下指令向服务端发送：
```bash
response = stub.Identify(request)
```

## 启动
- **启动服务端**：在`sei.py`文件路径下，运行以下命令启动服务端：
    ```bash
    python sei.py
    ```
- **启动客户端**：在`client.py`文件路径下，运行以下命令启动客户端：
   ```bash
   python  client.py
   ```
**确保服务端已成功启动，客户端才能正确连接并发送请求。**