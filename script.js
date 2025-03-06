const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const captureButton = document.getElementById("capture");
const resultDiv = document.getElementById("result");

let session = null;  // 预加载 ONNX 模型

// 访问摄像头
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => { video.srcObject = stream; });

// 预加载 ONNX 模型（防止点击时重复加载）
async function loadModel() {
    if (!session) {
        session = await ort.InferenceSession.create("best.onnx");
    }
    return session;
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

    try {
        // 运行 ONNX 推理
        const results = await session.run({ "images": inputTensor });

        // 获取检测框和类别
        const boxes = results["output"].data;  // 确保这个字段是正确的
        const classes = results["output_classes"].data; // 分类信息（0: with_helmet, 1: without_helmet）

        if (boxes.length === 0) {
            resultDiv.innerHTML = " 未检测到任何目标";
            return;
        }

        drawBoundingBoxes(boxes, classes);

        // 统计头盔佩戴情况
        const helmetCount = classes.filter(cls => cls === 0).length;
        const noHelmetCount = classes.filter(cls => cls === 1).length;

        resultDiv.innerHTML = `佩戴头盔: ${helmetCount} 人 | 未佩戴头盔: ${noHelmetCount} 人`;
    } catch (error) {
        console.error("推理错误:", error);
        resultDiv.innerHTML = " 检测失败，请检查控制台错误";
    }
}

// 绘制检测框
function drawBoundingBoxes(boxes, classes) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    boxes.forEach((box, index) => {
        const [x1, y1, x2, y2] = box;
        const color = classes[index] === 0 ? "green" : "red";
        const label = classes[index] === 0 ? "佩戴头盔" : "未佩戴头盔";

        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        ctx.fillStyle = color;
        ctx.font = "16px Arial";
        ctx.fillText(label, x1, y1 - 5);
    });
}

// 点击按钮触发推理
captureButton.addEventListener("click", detect);


// 点击按钮触发推理
captureButton.addEventListener("click", detect);
