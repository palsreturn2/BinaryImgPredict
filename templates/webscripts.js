var clickX = new Array ();
var clickY = new Array ();
var clickDrag = new Array ();

var paint;

var canvas = document.getElementById  ('board1');
context = canvas.getContext('2d');
context.fillStyle="#FF0000";
context.fillRect(0,0,128,128);


$ ('#board1').mousedown(function (e)
{
	var mouseX = e.pageX - this.offsetLeft;
	var mouseY = e.pageY - this.offsetTop;
	paint = true;
	addClick (e.pageX - this.offsetLeft, e.pageY - this.offsetTop);
	redraw ();
});

$ ('#board1').mousemove(function (e)
{
	if(paint)
	{
		addClick (e.pageX - this.offsetLeft, e.pageY - this.offsetTop, true);
		redraw ();
	}
});

$ ('#board1').mouseup(function (e)
{
	paint = false;
});

$ ('#board1').mouseleave(function (e)
{
	paint = false;
});


function addClick (x, y, dragging)
{
	clickX.push (x);
	clickY.push (y);
	clickDrag.push (dragging);
}

function redraw()
{
	context.clearRect(0, 0, context.canvas.width, context.canvas.height); // Clears the canvas
  
	context.strokeStyle = "#df4b26";
	context.lineJoin = "round";
	context.lineWidth = 5;
			
	for(var i=0; i < clickX.length; i++) 
	{		
		context.beginPath();
		if(clickDrag[i] && i)
		{
			context.moveTo(clickX[i-1], clickY[i-1]);
		}
		else
		{
			context.moveTo(clickX[i]-1, clickY[i]);
		}
		context.lineTo(clickX[i], clickY[i]);
		context.closePath();
		context.stroke();
	}
}

