import torch


if __name__=="__main__":
    tensor1=torch.rand((6,2))
    tensor2=torch.tensor([
                [0,0],
                [1,0],
                [0,1],
                [-1,0],
                [0,-1]
            ])
    tensor3=tensor1[None]+tensor2[:,None]

    print("stop")