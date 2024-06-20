import cv2
import math
import torch
import numpy as np
from tqdm import tqdm
from loss import SRLoss
from datasets import VideoDatasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model_s import Generator, Descriminator


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1

    G_model = Generator().to(device)
    G_model.load_state_dict(torch.load('weights/sr_0.pth', map_location=device))
    D_model = Descriminator().to(device)
    D_model.load_state_dict(torch.load('weights/sr_0_des.pth', map_location=device))
    model_test = Generator()

    LR_G, LR_D = 1e-5, 1e-5
    optimizer_G = torch.optim.Adam(G_model.parameters(), lr=LR_G)
    optimizer_G.load_state_dict(torch.load('weights/optimizer_G.pth', map_location=device))
    optimizer_D = torch.optim.Adam(D_model.parameters(), lr=LR_D)
    optimizer_D.load_state_dict(torch.load('weights/optimizer_D.pth', map_location=device))
    # optimizer_G = torch.optim.SGD(G_model.parameters(), lr=LR_G,momentum=0.9)
    # optimizer_D = torch.optim.SGD(D_model.parameters(), lr=LR_D,momentum=0.9)

    criterion = SRLoss()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = VideoDatasets(transforms=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    need_stop = False
    for epoch in range(0, 2000):
        train_loss, val_loss, g_losses, d_losses, l2_losses = 0, 0, [], [], []
        for frame_large, frame_small in tqdm(train_loader, leave=False):
            frame_small = frame_small.to(device)
            frame_large = frame_large.to(device)
            img_generated = G_model(frame_small)  # 生成图

            l2_loss = criterion(frame_large, img_generated)
            if math.isnan(l2_loss.item()):
                continue

            des_true = D_model(frame_large)
            des_generated = D_model(img_generated)

            G_loss = l2_loss-0.001/torch.mean(torch.log(1.-des_generated))
            D_loss = -torch.mean(torch.log(des_true)+torch.log(1-des_generated))

            if math.isinf(D_loss.item()):
                continue

            if math.isinf(G_loss.item()):
                continue

            optimizer_G.zero_grad()
            G_loss.backward(retain_graph=True)

            optimizer_D.zero_grad()
            D_loss.backward()

            optimizer_G.step()
            optimizer_D.step()

            g_losses.append(G_loss.item())
            d_losses.append(D_loss.item())
            l2_losses.append(l2_loss.item())
        if need_stop:
            break
        g_loss = np.mean(g_losses)
        d_loss = np.mean(d_losses)
        dis_loss = np.mean(l2_losses)

        torch.save(G_model.state_dict(), "weights/{0}_{1}.pth".format("sr", epoch))
        torch.save(D_model.state_dict(), "weights/{0}_{1}_des.pth".format("sr", epoch))
        torch.save(optimizer_G.state_dict(), "weights/optimizer_G.pth")
        torch.save(optimizer_D.state_dict(), "weights/optimizer_D.pth")
        print('epoch:{}, dis_loss:{:.6f}, g_loss:{:.6f}, d_loss:{:.6f}'.format(epoch, dis_loss, g_loss, d_loss))

        # # 测试
        model_test.load_state_dict(torch.load("weights/{0}_{1}.pth".format("sr", epoch), map_location='cpu'))
        model_test.eval()

        dataset = VideoDatasets()
        for j in range(10):
            frame_large, frame_small = dataset.__getitem__(np.random.randint(0, 2500))

            pred = model_test(frame_small.unsqueeze(0)).squeeze()
            pred = torch.permute(pred, (2, 1, 0)).detach().numpy()*255
            frame = pred.astype(np.uint8)
            cv2.imwrite(f"tmp/sr_{epoch}_{j}.jpg", frame)
