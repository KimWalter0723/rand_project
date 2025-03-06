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

    // 转换数据格式
    const inputTensor = new ort.Tensor("float32", new Float32Array(imageData.data), [1, 3, 640, 640]);

    // 运行模型
    const results = await session.run({ "images": inputTensor });

    console.log(results);
    resultDiv.innerHTML = "检测结果：" + JSON.stringify(results);
}

// 点击按钮触发推理
captureButton.addEventListener("click", detect);
