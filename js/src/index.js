$('#test').html('hi');

let canvas = $('#canvas');
let ctx = canvas[0].getContext('2d');
ctx.lineWidth = 5;
ctx.fillStyle = 'rgb(255, 0, 0)';
ctx.strokeRect(25, 25, 175, 200);
ctx.beginPath();
ctx.moveTo(50, 50);
// draw your path
ctx.fill();

