import pickle
from torch import nn

class teacherTimm1(nn.Module):
    def __init__(self, backbone_name):
        super(teacherTimm1, self).__init__()
        self.feature_extractor = self.load_nsts(backbone_name)
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.conv_unit = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),nn.BatchNorm2d(num_features=64),

            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),nn.BatchNorm2d(num_features=64),

            nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        )

    def load_nsts(self, net_name):
        if net_name == 'resnet18':
            with open('/root/autodl-tmp/models/nets/resnet18-st.pkl', 'rb') as f:
                net = pickle.load(f)
            print('Loaded teacher resnet18-st.pkl')
            return net

        elif net_name == 'resnet34':
            with open('/root/autodl-tmp/models/nets/resnet34-st.pkl', 'rb') as f:
                net = pickle.load(f)
            print('Loaded teacher resnet34-st.pkl')
            return net
        else:
            print('Loaded teacher NON')

    def forward(self, img, spec, obs):

        features_t = self.feature_extractor(img)
        # list [0]: (b,128,11,11)
        # list [1]: (b,256,6,6)
        # print(len(features_t))
        # print(self.feature_extractor)

        # x: [b,1,32,6]
        obs = obs.float()
        batsz = obs.size(0)
        obs = self.conv_unit(obs)  # [b, 128, 28, 2]
        # print(obs.shape)
        features_t.append(obs)

        return features_t


class studentTimm1(nn.Module):
    def __init__(self, backbone_name="resnet18"):
        super(studentTimm1, self).__init__()
        self.feature_extractor = self.load_nsts(backbone_name)

        self.conv_unit = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),nn.BatchNorm2d(num_features=64),

            nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        )

    def load_nsts(self, net_name):
        if net_name == 'resnet18':
            with open('/root/autodl-tmp/models/nets/resnet18-st.pkl', 'rb') as f:
                net = pickle.load(f)
            print('Loaded student resnet18-st.pkl')
            return net
        elif net_name == 'resnet34':
            with open('/root/autodl-tmp/models/nets/resnet34-st.pkl', 'rb') as f:
                net = pickle.load(f)
            print('Loaded teacher resnet34-st.pkl')
            return net
        else:
            print('Loaded student NON')

    def forward(self, img, spec, obs):
        features_t = self.feature_extractor(img)
        # x: [b,1,32,6]
        obs = obs.float()
        batsz = obs.size(0)
        obs = self.conv_unit(obs)
        features_t.append(obs)
        return features_t
