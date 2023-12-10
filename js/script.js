document.addEventListener('DOMContentLoaded', function() {
    const canvas = document.getElementById('myCanvas');
    const ctx = canvas.getContext('2d');
    const mousePositionDisplay = document.getElementById('mousePosition');

    // Set canvas size
    canvas.width = 300;
    canvas.height = 300;

    const points = []

    const predict = document.getElementById('predict');
    predict.addEventListener('click', function(event) {
      // Remove point
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      points.pop();
      points.forEach(function(point) {
        drawPoint(ctx, point.x, point.y);
      });
  });


    

    // Event listener for canvas click
    canvas.addEventListener('click', function(event) {
        const rect = canvas.getBoundingClientRect();
        const x = ((event.clientX - rect.left));
        const y = ((event.clientY - rect.top));

        const xnorm = ((event.clientX - rect.left)/canvas.width);
        const ynorm = (canvas.height - (event.clientY - rect.top))/canvas.height;

        // Update the mouse position display
        mousePositionDisplay.textContent = `Point: (${xnorm.toFixed(3)}, ${ynorm.toFixed(3)})`;
        const xcenter = (rect.right - rect.left) / 2 + rect.left;
        const ycenter = (rect.top - rect.bottom) / 2 + rect.bottom;
        drawPoint(ctx, xcenter, ycenter);

        // Optional: Draw a point on the canvas where clicked
        point = drawPoint(ctx, x, y);
        points.push(point);
        angle = Math.atan2(event.clientY - ycenter, event.clientX - xcenter);
        createFloatingImage(event.clientX - xcenter, event.clientY - ycenter, angle);
    });

});

function queryPoint(event){
  // Make a request
  const x = event.clientX;
  const y = event.clientY;
  const rect = canvas.getBoundingClientRect();
  const xnorm = ((event.clientX - rect.left)/canvas.width);
  const ynorm = (canvas.height - (event.clientY - rect.top))/canvas.height;
  const xcenter = (rect.right - rect.left) / 2 + rect.left;
  const ycenter = (rect.top - rect.bottom) / 2 + rect.bottom;

}

function requestPrediction(ctx, mu, sigma) {
  

}


function drawPoint(ctx, x, y) {
    ctx.fillStyle = '#76b900';
    point = ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2, true);
    ctx.fill();
    return point;
}

function createFloatingImage(x, y, angle) {
    const distance = 10;
    const imgX = x + distance * Math.cos(angle);
    const imgY = y + distance * Math.sin(angle);
    const img = document.createElement('img');
    img.src = 'icons/img-icon.png'; // Replace with your image path
    img.style.position = 'absolute';
    img.style.left = x + 'px';
    img.style.top = y + 'px';
    img.style.width = '50px'; // Set any size you want
    img.style.height = 'auto';
    img.style.zIndex = 1000;

    document.body.appendChild(img);
}
