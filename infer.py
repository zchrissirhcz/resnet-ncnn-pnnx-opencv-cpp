import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import cv2

def infer_resnet50():
    # 读取ImageNet的类别文件
    with open('synset_words.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    # 加载预训练的ResNet50模型
    model = resnet50(pretrained=True)
    model.eval()  # 设置为评估模式

    # 图像预处理
    def preprocess_image(cv2_img):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(cv2_img).unsqueeze(0)  # 添加batch维度

    # 读取图像
    img_path = 'cat.jpg'  # 替换为你的图片路径
    cv2_img = cv2.imread(img_path)  # 使用OpenCV读取图像
    if cv2_img is None:
        print(f"Error: Unable to load image at {img_path}")
        return
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)  # 转换为RGB

    # 预处理图像
    input_tensor = preprocess_image(cv2_img)

    # 执行推理
    with torch.no_grad():
        output = model(input_tensor)

    # 应用softmax来获取概率分布
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # 获取top 5类别和置信度
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(classes[top5_catid[i]], top5_prob[i].item())

# 调用函数执行推理
infer_resnet50()
