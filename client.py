import grpc
import sei_pb2
import sei_pb2_grpc
import os


# 创建请求的帮助函数，构造识别参数
def create_file_request(file_info_list):
    request = sei_pb2.IdenReq()

    for file_info in file_info_list:
        # 从file_info中提取不同的参数
        param = sei_pb2.IdenParam(
            sample_rate=file_info['sample_rate'],
            band_width=file_info['band_width'],
            symbol_rate=file_info['symbol_rate'],
            center_freq=file_info['center_freq'],
            file_path=file_info['file_path']
        )
        request.params.append(param)

    return request


def run_client():
    # 与服务端建立连接
    channel = grpc.insecure_channel('localhost:50051')  # 服务端地址和端口
    stub = sei_pb2_grpc.IdenServiceStub(channel)

    # 准备每个文件的信息，包括不同的采样率、带宽等
    file_info_list = [
        {
            'file_path': r"G:\shanghai12\20250111_kuapi\0\0_IQ_ID11_Fs1945523_Fc1535000053_Bw486381.wav",
            'sample_rate': 2000000.0,  # 采样率，例如2 MHz
            'band_width': 1000000.0,  # 带宽，例如1 MHz
            'symbol_rate': 100000.0,  # 符号速率，例如100 kBd
            'center_freq': 5000000.0  # 信号频偏，例如5 MHz
        },
        {
            'file_path': 'E:/20250102/0_6_ID8_Fs2234983_Fc15python005977_Bw558746.dat',
            'sample_rate': 1500000.0,  # 不同的采样率
            'band_width': 900000.0,    # 不同的带宽
            'symbol_rate': 90000.0,    # 不同的符号速率
            'center_freq': 4000000.0   # 不同的频偏
        },
        {
            'file_path': 'E:/20250102/2_1_ID8_Fs2129983_Fc1535009726_Bw532496.dat',
            'sample_rate': 2500000.0,  # 不同的采样率
            'band_width': 1100000.0,   # 不同的带宽
            'symbol_rate': 120000.0,   # 不同的符号速率
            'center_freq': 6000000.0   # 不同的频偏
        },
        {
            'file_path': r"E:\20250102\2_4_ID8_Fs1976235_Fc1534999414_Bw494059.dat",
            'sample_rate': 2500000.0,  # 不同的采样率
            'band_width': 1100000.0,  # 不同的带宽
            'symbol_rate': 120000.0,  # 不同的符号速率
            'center_freq': 6000000.0  # 不同的频偏
        },
        {
            'file_path': r"E:\20250102\2_9_ID8_Fs2118734_Fc1534999414_Bw529683.dat",
            'sample_rate': 2500000.0,  # 不同的采样率
            'band_width': 1100000.0,  # 不同的带宽
            'symbol_rate': 120000.0,  # 不同的符号速率
            'center_freq': 6000000.0  # 不同的频偏
        },
        {
            'file_path': r"E:\20250102\3_1_ID8_Fs2291232_Fc1535001289_Bw572808.dat",
            'sample_rate': 2500000.0,  # 不同的采样率
            'band_width': 1100000.0,  # 不同的带宽
            'symbol_rate': 120000.0,  # 不同的符号速率
            'center_freq': 6000000.0  # 不同的频偏
        },
        {
            'file_path': r"E:\20250102\3_4_ID8_Fs2174983_Fc1534987227_Bw543746.dat",
            'sample_rate': 2500000.0,  # 不同的采样率
            'band_width': 1100000.0,  # 不同的带宽
            'symbol_rate': 120000.0,  # 不同的符号速率
            'center_freq': 6000000.0  # 不同的频偏
        },
        {
            'file_path': 'E:/20250102/2_1_ID8_Fs2129983_Fc1535009726_Bw532496.dat',
            'sample_rate': 2500000.0,  # 不同的采样率
            'band_width': 1100000.0,  # 不同的带宽
            'symbol_rate': 120000.0,  # 不同的符号速率
            'center_freq': 6000000.0  # 不同的频偏
        },
        {
            'file_path': r"E:\20250102\3_7_ID8_Fs2328732_Fc1535001289_Bw582183.dat",
            'sample_rate': 2500000.0,  # 不同的采样率
            'band_width': 1100000.0,  # 不同的带宽
            'symbol_rate': 120000.0,  # 不同的符号速率
            'center_freq': 6000000.0  # 不同的频偏
        },
        {
            'file_path': r"E:\20250102\4_6_ID8_Fs2051234_Fc1534998477_Bw512808.dat",
            'sample_rate': 2500000.0,  # 不同的采样率
            'band_width': 1100000.0,  # 不同的带宽
            'symbol_rate': 120000.0,  # 不同的符号速率
            'center_freq': 6000000.0  # 不同的频偏
        },
        {
            'file_path': r"E:\20250102\4_10_ID8_Fs2084984_Fc1535006914_Bw521246.dat",
            'sample_rate': 2500000.0,  # 不同的采样率
            'band_width': 1100000.0,  # 不同的带宽
            'symbol_rate': 120000.0,  # 不同的符号速率
            'center_freq': 6000000.0  # 不同的频偏
        },
        {
            'file_path': r"E:\20250102\5_2_ID8_Fs2129983_Fc1535019101_Bw532496.dat",
            'sample_rate': 2500000.0,  # 不同的采样率
            'band_width': 1100000.0,  # 不同的带宽
            'symbol_rate': 120000.0,  # 不同的符号速率
            'center_freq': 6000000.0  # 不同的频偏
        },
        {
            'file_path': r"E:\20250102\5_8_ID8_Fs2084984_Fc1534993789_Bw521246.dat",
            'sample_rate': 2500000.0,  # 不同的采样率
            'band_width': 1100000.0,  # 不同的带宽
            'symbol_rate': 120000.0,  # 不同的符号速率
            'center_freq': 6000000.0  # 不同的频偏
        },
        {
            'file_path': 'E:/20250102/2_1_ID8_Fs2129983_Fc1535009726_Bw532496.dat',
            'sample_rate': 2500000.0,  # 不同的采样率
            'band_width': 1100000.0,  # 不同的带宽
            'symbol_rate': 120000.0,  # 不同的符号速率
            'center_freq': 6000000.0  # 不同的频偏
        },
        {
            'file_path': r"E:\20250102\6_3_ID8_Fs2189983_Fc1534990039_Bw547496.dat",
            'sample_rate': 2500000.0,  # 不同的采样率
            'band_width': 1100000.0,  # 不同的带宽
            'symbol_rate': 120000.0,  # 不同的符号速率
            'center_freq': 6000000.0  # 不同的频偏
        },
        {
            'file_path': r"E:\20250102\6_3_ID8_Fs2189983_Fc1534990039_Bw547496.dat",
            'sample_rate': 2500000.0,  # 不同的采样率
            'band_width': 1100000.0,  # 不同的带宽
            'symbol_rate': 120000.0,  # 不同的符号速率
            'center_freq': 6000000.0  # 不同的频偏
        },
        {
            'file_path': r"E:\20250102\6_3_ID8_Fs2189983_Fc1534990039_Bw547496.dat",
            'sample_rate': 2500000.0,  # 不同的采样率
            'band_width': 1100000.0,  # 不同的带宽
            'symbol_rate': 120000.0,  # 不同的符号速率
            'center_freq': 6000000.0  # 不同的频偏
        },
        {
            'file_path': r"E:\20250102\6_3_ID8_Fs2189983_Fc1534990039_Bw547496.dat",
            'sample_rate': 2500000.0,  # 不同的采样率
            'band_width': 1100000.0,  # 不同的带宽
            'symbol_rate': 120000.0,  # 不同的符号速率
            'center_freq': 6000000.0  # 不同的频偏
        },
        {
            'file_path': r"E:\20250102\6_3_ID8_Fs2189983_Fc1534990039_Bw547496.dat",
            'sample_rate': 2500000.0,  # 不同的采样率
            'band_width': 1100000.0,  # 不同的带宽
            'symbol_rate': 120000.0,  # 不同的符号速率
            'center_freq': 6000000.0  # 不同的频偏
        },
        {
            'file_path': r"E:\20250102\6_3_ID8_Fs2189983_Fc1534990039_Bw547496.dat",
            'sample_rate': 2500000.0,  # 不同的采样率
            'band_width': 1100000.0,  # 不同的带宽
            'symbol_rate': 120000.0,  # 不同的符号速率
            'center_freq': 6000000.0  # 不同的频偏
        },
        {
            'file_path': r"E:\20250102\6_3_ID8_Fs2189983_Fc1534990039_Bw547496.dat",
            'sample_rate': 2500000.0,  # 不同的采样率
            'band_width': 1100000.0,  # 不同的带宽
            'symbol_rate': 120000.0,  # 不同的符号速率
            'center_freq': 6000000.0  # 不同的频偏
        }
    ]

    # 创建请求对象
    request = create_file_request(file_info_list)

    try:
        # 发送请求并接收响应
        response = stub.Identify(request)

        # 打印返回的识别结果
        for idx, result in enumerate(response.results):
            if result.code == 0:
                print(f"File {file_info_list[idx]['file_path']} - Recognition successful, Result: {result.object}")
            else:
                print(f"File {file_info_list[idx]['file_path']} - Error: {result.message}")

    except grpc.RpcError as e:
        print(f"gRPC Error: {e}")


if __name__ == '__main__':
    run_client()
