from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from model import CustomNet  # 替换为你自己的模型类名
from torch.serialization import add_safe_globals

app = Flask(__name__)


add_safe_globals([CustomNet])

model = torch.load('E:/NeuralNetwork-TrainingReport/nndl_project/models/model.pkl', weights_only=False)
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        print("❌ 没有接收到图像文件")
        return jsonify({'error': 'No image file uploaded'}), 400

    file = request.files['image']
    print(f"✅ 收到图像文件: {file.filename}")

    try:
        img = Image.open(file).convert('RGB')
        img = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(img)
            print("✅ 模型输出:", output)
            prediction = output.argmax(dim=1).item()
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        print("❌ 处理图像或模型推理出错:", e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
