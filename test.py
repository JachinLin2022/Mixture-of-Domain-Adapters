import torch



x = torch.randn(256,128,2)
x = x[:,-1]
x = x.unsqueeze(1)


print(x.shape)


output = torch.nn.Softmax(dim=-1)(x)



w1 = torch.mean(output[:,:,0],dim=0)
w2 = torch.mean(output[:,:,1],dim=0)
print(w1,w2)