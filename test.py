import torch

torch.manual_seed(42)

module = torch.nn.Linear(10, 10)
print(123123123)
print(123123123)
# 第一次调用
module.weight.data.normal_(mean=0.0, std=0.02)
print(module.weight.data)

# # 第二次调用
# module.weight.data.normal_(mean=0.0, std=0.02)
# print(module.weight.data)
