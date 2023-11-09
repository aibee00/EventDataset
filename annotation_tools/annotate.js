// 初始化标注数据
var annotations = [];

var canvas = document.createElement('canvas');
var ctx = canvas.getContext('2d');
document.body.appendChild(canvas);

var imageElement = document.querySelector(".stImage > img");
var coordinatesElement = document.getElementById('coordinates');

var isDrawing = false;
var startX, startY, endX, endY;

function getCoordinates(event) {
    if (!isDrawing) {
        isDrawing = true;
        startX = event.clientX - canvas.getBoundingClientRect().left;
        startY = event.clientY - canvas.getBoundingClientRect().top;
    } else {
        isDrawing = false;
        endX = event.clientX - canvas.getBoundingClientRect().left;
        endY = event.clientY - canvas.getBoundingClientRect().top;

        // 绘制 bounding box
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.strokeRect(startX, startY, endX - startX, endY - startY);

        // 保存坐标
        annotations.push({ x1: startX, y1: startY, x2: endX, y2: endY });
        updateCoordinates();
    }
}

function updateCoordinates() {
    coordinatesElement.innerText = JSON.stringify(annotations);
}

function drawTempBoundingBox(event) {
    if (!isDrawing) return;

    // 清除之前的临时 bounding box
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 获取当前鼠标位置
    var tempEndX = event.clientX - canvas.getBoundingClientRect().left;
    var tempEndY = event.clientY - canvas.getBoundingClientRect().top;

    // 绘制临时 bounding box
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;
    ctx.strokeRect(startX, startY, tempEndX - startX, tempEndY - startY);
}

function clearTempBoundingBox() {
    // 清除临时 bounding box
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// 监听图像的单击事件
imageElement.addEventListener('click', getCoordinates);

// 监听鼠标移动事件，显示临时 bounding box
imageElement.addEventListener('mousemove', drawTempBoundingBox);

// 监听鼠标松开事件，清除临时 bounding box
imageElement.addEventListener('mouseup', clearTempBoundingBox);
