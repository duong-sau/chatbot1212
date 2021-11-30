import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer


class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.t5 = BertModel.from_pretrained('bert').base_model
        self.liner = nn.Sequential(nn.Linear(512, 512), nn.Sigmoid())
        self.out = nn.Linear(512, 1)

    def forward_one(self, x):
        x = self.t5(x)
        x = x.view(x.size()[0], -1)
        x = self.liner(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        #  return self.sigmoid(out)
        return out


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        midst = self.margin - dist
        dist = torch.clamp(midst, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss


# preprocessing and loading the dataset
class SiameseDataset(torch.utils.data.Dataset):
    def __init__(self, training_csv=None, training_dir=None, transform=None):
        # used to prepare the labels and images path
        train_df = pd.read_csv('https://raw.githubusercontent.com/duong-sau/chatbot1212/master/Model/Data'
                               '/IntentClassification/Positive/learn_data.csv', header=0)
        self.tokenizer = BertTokenizer.from_pretrained('bert')
        self.data_column = [(lambda x: "stsb: " + train_df.iloc[x]["source"] + '</s>')
                            (x) for x in range(len(train_df))]
        self.class_column = [(lambda x: int(float(train_df.iloc[x]["target"])))
                             (x) for x in range(len(train_df))]
        self.max_len = 512

    def __getitem__(self, index):
        s = self.data_column[index]
        index1 = s.rfind('sentence1')
        index2 = s.rfind('sentence2')
        sentence1 = self.tokenizer.encode_plus(s[index1:index2 - 9], max_length=self.max_len, padding='longest',
                                               return_tensors="pt")
        sentence2 = self.tokenizer.encode_plus(s[index2:], max_length=self.max_len, padding='longest',
                                               return_tensors="pt")
        target_ids = torch.tensor(self.class_column[index], dtype=torch.int32)
        return {"ids1": sentence1,
                "ids2": sentence2,
                "label": target_ids}

    def __len__(self):
        return len(self.data_column)


siamese_dataset = SiameseDataset()
train_dataloader = DataLoader(siamese_dataset, shuffle=True, num_workers=8, batch_size=4)


def train():
    loss = []
    counter = []
    iteration_number = 0
    for epoch in range(1, 100):
        for i, data in enumerate(train_dataloader, 0):
            sentence1 = data['ids1']
            sentence2 = data['ids2']
            label = data['label']
            optimizer.zero_grad()
            output1, output2 = net(sentence1, sentence2)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
        print("Epoch {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
        iteration_number += 10
        counter.append(iteration_number)
        loss.append(loss_contrastive.item())
    return net


# set the device to cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = train()
torch.save(model.state_dict(), "output/model.pt")
print("Model Saved Successfully")
# for test
if __name__ == '__main__':
    net = SiameseNetwork().cuda()
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0005)
