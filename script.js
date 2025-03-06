const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const captureButton = document.getElementById("capture");
const resultDiv = document.getElementById("result");

// 访问摄像头
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => { video.srcObject = stream; });

// 加载 ONNX 模型
async function loadModel() {
    return await ort.InferenceSession.create("best.onnx");
}

// 进行推理
async function detect() {
    const session = await loadModel();
    
    // 画面截图
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // 获取像素数据
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    
    // 转换数据格式（YOLOv8 需要 640x640）
    const inputTensor = new ort.Tensor("float32", new Float32Array(imageData.data), [1, 3, 640, 640]);

    // 运行模型
    const results = await session.run({ "images": inputTensor });

    // 解析检测结果
    const boxes = results["output"].data;  // YOLO 输出的检测框信息
    const classes = results["output_classes"].data; // 预测类别

    if (boxes.length === 0) {
        resultDiv.innerHTML = "未检测到任何目标";
        return;
    }

    drawBoundingBoxes(boxes, classes);
    
    // 统计戴头盔和未戴头盔人数
    const helmetCount = classes.filter(cls => cls === 0).length;
    const noHelmetCount = classes.filter(cls => cls === 1).length;
    
    resultDiv.innerHTML = `检测结果：戴头盔 ✅ ${helmetCount} 人, 未戴头盔 ❌ ${noHelmetCount} 人`;
}

// 绘制检测框
function drawBoundingBoxes(boxes, classes) {
    boxes.forEach((box, index) => {
        const [x1, y1, x2, y2] = box;  // YOLOv8 的坐标格式
        ctx.strokeStyle = classes[index] === 0 ? "green" : "red";  // 绿色（戴头盔），红色（未戴头盔）
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    });
}

// 点击按钮触发推理
captureButton.addEventListener("click", detect);
