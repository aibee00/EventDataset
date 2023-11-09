var annotations = [];

var canvas = document.createElement('canvas');
var ctx = canvas.getContext('2d');
document.body.appendChild(canvas);

var imageElement = document.getElementById('image');
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

        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.strokeRect(startX, startY, endX - startX, endY - startY);

        annotations.push({ x1: startX, y1: startY, x2: endX, y2: endY });
        updateCoordinates();
    }
}

function updateCoordinates() {
    coordinatesElement.innerText = JSON.stringify(annotations);
}

imageElement.addEventListener('click', getCoordinates);
