import gc
import io
import re
import pkgutil
from collections import Counter
import grpc
from concurrent import futures
import sei_pb2
import sei_pb2_grpc
import os
import torch
import numpy as np
import time
from aaresnet1dPiles import *
import sys
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from collections import Counter
from func import sun
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler



# 读取并处理文件
def process_file(file_path):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        return -1, f"File not found: {file_path}", None  # 文件不存在错误

    try:
        t = time.perf_counter()
        fid = open(file_path, "rb")
        fid.seek(int(3200 * 0.8 * 2 * 2 * 1000))
        sample_nums = 600
        test_list = []
        for i in range(sample_nums):
            data = fid.read(1000 * 2 * 2)
            if data == '' or len(data) < 1000 * 2 * 2:
                print("PLease stop and ensure the datafile's size!")
                break
            data = np.frombuffer(data, dtype='<i2')
            dataI = data[::2]
            dataQ = data[1::2]
            IQ = np.stack((dataI, dataQ), axis=0)
            test_list.append(IQ)
        fid.close()
        x_test = np.array(test_list)
        test_data = torch.tensor(x_test, dtype=torch.float32)

        return 0, "File processed successfully", test_data  # 成功处理文件

    except Exception as e:
        return -2, f"Error processing file {file_path}: {str(e)}", None  # 其他错误



# 实现识别服务
class IdenService(sei_pb2_grpc.IdenServiceServicer):
    def Identify(self, request, context):
        # 从请求中获取文件路径
        file_paths = [param.file_path for param in request.params]
        results = [None] * len(file_paths)  # 预先分配结果空间，保持顺序

        # 1. 将所有文件处理成输入数据
        valid_data = []
        error_results = []
        valid_file = 0

        for idx, file_path in enumerate(file_paths):
            result_code, result_message, signal_tensor = process_file(file_path)

            if result_code != 0:  # 如果处理过程中出现错误
                error_results.append((idx, result_code, result_message, file_path))
            elif isinstance(signal_tensor, torch.Tensor):  # 确保只有有效的信号数据被加入
                valid_data.append((idx, signal_tensor))  # 保留文件的索引
                valid_file += 1

        # 2. 处理文件错误结果
        for idx, error_code, error_message, file_path in error_results:
            result = sei_pb2.IdenResult(
                code=error_code,
                message=error_message,
                object="Error"  # 错误信息
            )
            results[idx] = result  # 将错误信息放回对应的位置

        # 3. 使用神经网络进行预测
        if valid_data:  # 如果有有效的文件数据
            signal_tensors = torch.stack(
                [tensor for _, tensor in sorted(valid_data, key=lambda x: x[0])])

            signal_tensors = signal_tensors.view(-1, 2, 1000)
            test_loader = torch.utils.data.TensorDataset(signal_tensors)
            test_loader = torch.utils.data.DataLoader(test_loader, batch_size=128, shuffle=False)

            all_predictions = []
            try:
                t = time.perf_counter()
                for data in test_loader:
                    data = data[0].to(cuda)

                    with torch.no_grad():
                        output1 = model(data)
                    _, predicted = torch.max(output1.data, 1)
                    all_predictions.extend(predicted.cpu().numpy())
                targets = []
                for i in range(valid_file):
                    class_predictions = all_predictions[i * valid_file: (i + 1) * valid_file]
                    count = Counter(class_predictions)
                    most_common_prediction, _ = count.most_common(1)[0]
                    targets.append(most_common_prediction)

                for i, (idx, pred) in enumerate(zip([d[0] for d in valid_data], targets)):
                    result = sei_pb2.IdenResult(
                        code=0,
                        message="Recognition successful",
                        object=str(pred)  # 预测结果转为字符串
                    )
                    results[idx] = result  # 将识别结果放回对应的位置

            except Exception as e:
                error_message = f"Error during model inference: {str(e)}"
                for idx in range(len(file_paths)):
                    results[idx] = sei_pb2.IdenResult(
                        code=-1,
                        message=error_message,
                        object="Model inference failed"
                    )
        # 返回响应
        response = sei_pb2.IdenRes(results=results)
        return response


# 启动服务
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    sei_pb2_grpc.add_IdenServiceServicer_to_server(IdenService(), server)

    # 监听端口50051
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server is listening on port 50051")
    server.wait_for_termination()


def clear():
    for path in sys.path:
        if re.match(r'^_MEI\d+$', os.path.basename(path)):
            if os.path.exists(path):
                os.remove(path)

torch.cuda.init()
cuda = torch.device('cuda')

t = time.perf_counter()
model = AAResNet(num_classes=8, in_channel=2)
# base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
# model_path = os.path.join(base_dir, 'best_model_train_35USRP_QPSK_lvbo_IQ_feature_10_classes_selected_shunxu_duo.pth')

print(sys.path[0])
print(sys.argv[0])
print(os.path.dirname(os.path.realpath(sys.executable)))
print(os.path.dirname(os.path.realpath(sys.argv[0])))

# 检查是否运行在打包后的可执行文件中
# if hasattr(sys, '_MEIPASS'):
#     base_dir = sys._MEIPASS
# else:
base_dir = os.path.dirname(os.path.realpath(sys.argv[0]))#os.path.dirname(os.path.abspath(sys.executable))
model_path = os.path.join(base_dir, './weights/model1.pth')

with open(model_path, 'rb') as f:
    encrypted_weights = f.read()  # 读取剩余部分的加密权重

state_dict = sun(encrypted_weights)

# state_dict = torch.load(model_path, map_location=cuda)['model']
model.load_state_dict(state_dict)
model = model.to(cuda)

maps = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}


if __name__ == '__main__':
    print("代码开始执行")
    # serve()
    try:
        # 你的主程序逻辑
        serve()
    except Exception as e:
        print(f"程序运行出错: {e}")
        input("按回车键退出程序...")  # 仅在程序出错时等待用户确认
    finally:
        clear()
        gc.collect()
