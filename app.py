from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np

# 导入你的测试函数
from test import test

app = Flask(__name__)

# 设置上传文件的保存路径
app.config['UPLOAD_FOLDER'] = 'static/uploads/'


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 获取上传的文件
        img1 = request.files['img1']
        img2 = request.files['img2']

        # 保存上传的文件
        filename1 = secure_filename(img1.filename)
        filename2 = secure_filename(img2.filename)
        img1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
        img2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))

        # 调用测试函数，获取模型的输出结果
        result = test(os.path.join(app.config['UPLOAD_FOLDER'], filename1),
                      os.path.join(app.config['UPLOAD_FOLDER'], filename2))

        # 将结果传递给index.html
        return render_template('index.html', img1=filename1, img2=filename2, result=result)

    # 显示上传页面
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)
