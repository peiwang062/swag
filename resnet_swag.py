from PIL import Image
import numpy as np
import argparse
import torch
import torch.optim as optim
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import cv2

parser = argparse.ArgumentParser(description='PyTorch SWAG')
parser.add_argument('-a', '--arch', default='resnet50_swag', help='architecture name')
global args
args = parser.parse_args()


gpu = '0'
gpu = gpu.split(',')
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu)
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def load_image(img_path, max_size=400, shape=None):
    image = Image.open(img_path).convert('RGB')

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    in_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    image = in_transform(image)[:3, :, :].unsqueeze(0)

    return image


def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array(
        (0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image


def save_checkpoint(state, filename='checkpoint_res.pth.tar'):
    torch.save(state, filename)


def get_features(image, model, layers=None):

    if ARCHITECTURE == 'resnet50_swag':
        if layers is None:
            layers1 = {'0': 'conv1_0',
                       '1': 'conv1_1',
                       '2': 'conv1_2'
                       }
            layers2 = {'0': 'conv2_0',
                       '1': 'conv2_1',
                       '2': 'conv2_2',
                       '3': 'conv2_3'
                       }
            layers3 = {'0': 'conv3_0',
                       '1': 'conv3_1',
                       '2': 'conv3_2',
                       '3': 'conv3_3',
                       '4': 'conv3_4',
                       '5': 'conv3_5'
                       }
            layers4 = {'0': 'conv4_0',
                       '1': 'conv4_1',
                       '2': 'conv4_2'
                       }
        # softmax = nn.Softmax2d()

        # T = 100
        T = 1
        alpha = 0.001

        features = {}
        x = image
        x = model.conv1(x)

        x = model.bn1(x)
        x = model.relu(x)
        features['conv0_0'] = x
        # features['conv0_0'] = softmax(x / T)
        # features['conv0_0'] = softmax3d(x / T)
        # features['conv0_0'] = alpha * x
        x = model.maxpool(x)
        features['conv0_1'] = x
        # features['conv0_1'] = softmax(x / T)
        # features['conv0_1'] = softmax3d(x / T)
        # features['conv0_1'] = alpha * x

        # Although we found adding softmax smoothing can always improve the results, the best results sometimes are obtained by only smoothing the deeper layers,
        # as the paper suggests, deeper layers are more peaky and have small entropy

        for name, layer in enumerate(model.layer1):
            x = layer(x)
            if str(name) in layers1:
                features[layers1[str(name)]] = x
                # features[layers1[str(name)]] = softmax(x / T)
                # features[layers1[str(name)]] = softmax3d(x / T)
                # features[layers1[str(name)]] = alpha * x
        for name, layer in enumerate(model.layer2):
            x = layer(x)
            if str(name) in layers2:
                features[layers2[str(name)]] = x
                # features[layers2[str(name)]] = softmax(x / T)
                # features[layers2[str(name)]] = softmax3d(x / T)
                # features[layers2[str(name)]] = alpha * x
        for name, layer in enumerate(model.layer3):
            x = layer(x)
            if str(name) in layers3:
                # features[layers3[str(name)]] = softmax(x / T)
                # features[layers3[str(name)]] = softmax3d(x / T)
                features[layers3[str(name)]] = alpha * x
        for name, layer in enumerate(model.layer4):
            x = layer(x)
            if str(name) in layers4:
                # features[layers4[str(name)]] = softmax(x / T)
                # features[layers4[str(name)]] = softmax3d(x / T)
                features[layers4[str(name)]] = alpha * x

    else:
        if layers is None:
            layers1 = {'0': 'conv1_0',
                       '1': 'conv1_1',
                       '2': 'conv1_2'
                       }
            layers2 = {'0': 'conv2_0',
                       '1': 'conv2_1',
                       '2': 'conv2_2',
                       '3': 'conv2_3'
                       }
            layers3 = {'0': 'conv3_0',
                       '1': 'conv3_1',
                       '2': 'conv3_2',
                       '3': 'conv3_3',
                       '4': 'conv3_4',
                       '5': 'conv3_5'
                       }
            layers4 = {'0': 'conv4_0',
                       '1': 'conv4_1',
                       '2': 'conv4_2'
                       }

        features = {}
        x = image
        x = model.conv1(x)

        x = model.bn1(x)
        x = model.relu(x)
        features['conv0_0'] = x
        x = model.maxpool(x)

        for name, layer in enumerate(model.layer1):
            x = layer(x)
            if str(name) in layers1:
                features[layers1[str(name)]] = x
        for name, layer in enumerate(model.layer2):
            x = layer(x)
            if str(name) in layers2:
                features[layers2[str(name)]] = x
        for name, layer in enumerate(model.layer3):
            x = layer(x)
            if str(name) in layers3:
                features[layers3[str(name)]] = x
        for name, layer in enumerate(model.layer4):
            x = layer(x)
            if str(name) in layers4:
                features[layers4[str(name)]] = x

    return features


def softmax3d(input):
    m = nn.Softmax()
    a, b, c, d = input.size()
    input = torch.reshape(input, (1, -1))
    output = m(input)
    output = torch.reshape(output, (a, b, c, d))
    return output


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


def style_transfer(model, style, content, target, style_layer_weights, content_layer_weights, style_weight,
                   content_weight, optimizer):

    content_features = get_features(content, model)
    style_features = get_features(style, model)

    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}



    run = [0]
    while run[0] <= 1000:
        def closure():
            optimizer.zero_grad()
            target_features = get_features(target, model)

            content_loss = torch.mean((target_features[content_layer_weights] -
                                       content_features[content_layer_weights]) ** 2)
            style_loss = 0
            for layer in style_layer_weights:
                target_feature = target_features[layer]
                target_gram = gram_matrix(target_feature)
                # _, d, h, w = target_feature.shape
                style_gram = style_grams[layer]

                layer_style_loss = style_layer_weights[layer] * torch.mean(
                    (target_gram - style_gram) ** 2)
                style_loss += style_weight * layer_style_loss

            total_loss = content_weight * content_loss + style_loss
            total_loss.backward()

            if run[0] % 500 == 0:
                print("run {}:".format(run))
                print('total Loss : {:4f}'.format(total_loss.item()))

            run[0] += 1
            return content_weight * content_loss + style_loss

        optimizer.step(closure)

    final_img = im_convert(target)
    return final_img


# load model
ARCHITECTURE = args.arch  # 'resnet50', 'resnet50_swag'

resnet = models.resnet50(pretrained=True)




for param in resnet.parameters():
    param.requires_grad_(False)

torch.cuda.is_available()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet.to(device).eval()

# load style and content images combinations

style_image_list = []
for file in glob.glob('./style_images/*'):
    style_image_list.append(file)

content_image_list = []
for file in glob.glob('./content_images/*'):
    content_image_list.append(file)

for i_style in range(len(style_image_list)):
    style_img_name = style_image_list[i_style].split('/')
    style_img_name = style_img_name[-1].split('.')
    style_img_name = style_img_name[0].split('\\')[-1]
    for i_content in range(len(content_image_list)):

        print('processing content', i_content, ' style ', i_style)

        style = load_image(style_image_list[i_style]).to(device)
        content = load_image(content_image_list[i_content]).to(device)


        target = content.clone().requires_grad_(True).to(device)

        style_weights = {'conv0_0': 1.0,
                         'conv1_2': 1.0,
                         'conv2_3': 1.0,
                         'conv3_5': 1.0,
                         'conv4_2': 1.0}
        content_weights = 'conv3_5'

        content_weight = 1
        style_weight = 1e17

        optimizer = optim.LBFGS([target])

        final_styled = style_transfer(resnet, style, content, target, style_weights, content_weights,
                                                  style_weight, content_weight, optimizer)

        content_img_name = content_image_list[i_content].split('/')
        content_img_name = content_img_name[-1].split('.')
        content_img_name = content_img_name[0].split('\\')[-1]

        if not os.path.exists('./results/' + ARCHITECTURE):
            os.makedirs('./results/' + ARCHITECTURE)

        save_path_cv2 = './results/' + ARCHITECTURE + '/' + content_img_name + '_' + style_img_name + '_cv2.png'
        final_styled_cv2 = np.uint8(255 * final_styled)
        final_styled_cv2_bgr = final_styled_cv2[:, :, [2, 1, 0]]
        cv2.imwrite(save_path_cv2, final_styled_cv2_bgr)
