import random

from DataPropose import DataPrepose
from FLingredients import Server
from FLingredients import Client
from Network import ResNet
import numpy as np

if __name__ == "__main__":
    total_client = 10  # 总客户端数量
    k_clients = 10  # 选择客户端数量
    resnet = ResNet.getResnet()  # 选择模型
    communicate_round = 2

    train_data, test_data = DataPrepose.dataDownloadCIFAR10()

    # clientDataDict = DataPrepose.splitData_nonIID_Label_Dirichlet(
    #     train_data=train_data,
    #     client_num=total_client,
    #     dirichlet_alpha=0.1,
    #     is_overlap=False,
    # )

    fed_server = Server.Server(
        n = total_client,
        k = k_clients,
        initmodel = resnet
    )

    clientDataDict = DataPrepose.splitData(train_data, total_client)
    # 初始化客户端并分发数据
    client_list = fed_server.clients
    for i in range(total_client):
        client_i_traindata = []
        for idx in clientDataDict[i]:
            client_i_traindata.append(train_data[idx])
        random.shuffle(client_i_traindata)
        client_list.append(
            Client.Client(
                train_data = client_i_traindata,
                test_data = test_data,
                train_model = ResNet.getResnet(),
                server= fed_server,
            )
        )

    for i in range(total_client):
        client_list[i].setHyperparameters(epoch=5, learning_rate=1e-3)

    for t in range(communicate_round):
        client_idx = fed_server.selectKClients()
        print(r"In %d communication round, the selected client is " % t, client_idx)

        # 用本地样本训练teacher模型
        for i in range(k_clients):
            print(f"Now is training NO {client_idx[i]:>1d} client")
            client_list[client_idx[i]].trainTeacherModel()