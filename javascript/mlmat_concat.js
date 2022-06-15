// this needs to be used in conjunction with a jit.submatrix in parent patch.
// jit.submatrix will not work properly when called from js
//autowatch = 1;

var maximum_size = 10000;
declareattribute("maximum_size");
var mode = 0;
declareattribute("mode", "get_mode", "set_mode", 1);
var index = 0;
outlets = 2;
var outmatrix;
				
var currentinfo = {};
	
function get_info(mat) {	
	return { planecount: mat.planecount, 
					type: mat.type, 
					dim: mat.dim, 
 					size: mat.size};
}

function set_mode(v) {
	if(v != mode) {
		clear();
	}
	mode = v;
}

function get_mode() {
	return mode;
}

function check_matrix(mat) {
	var minfo = get_info(mat);
	if(currentinfo.planecount == minfo.planecount && currentinfo.type == minfo.type && currentinfo.dim[0] == minfo.dim[0] &&  currentinfo.dim[1] == minfo.dim[1]) {
		return true;
	} else {
		return false;
	}
}

function post_info(minfo) {
	post(minfo.planecount, minfo.type, minfo.dim);
}

function clear() {
	index = 0;
	outmatrix = new JitterMatrix();
	currentinfo = {};
}
	
function create_matrix(mode, minfo, max_size) {
	var mat;
	var size = max_size;
	
	var x = (minfo.dim[0] === undefined) ? minfo.dim : minfo.dim[0];
	var y = (minfo.dim[1] === undefined) ? 1 : minfo.dim[1];
	
	while(minfo.size*size >= 0x3FFFFFFF) {
			size--;
	}
	
	switch (mode) {
		case 0:
			mat = new JitterMatrix(minfo.planecount, minfo.type, x, y*size);
			break;
			
		case 1: 
			if(minfo.planecount > 1) {
				error('attempting to use a ' + minfo.planecount + ' plane matrix in mode 1.');
			} else {
				mat = new JitterMatrix(minfo.planecount, minfo.type, x, y*size);
			}
			break
			
		case 2:
			if(minfo.planecount > 1) {
				error('attempting to use a ' + minfo.planecount + ' plane matrix in mode 2.');
			} else {
				mat = new JitterMatrix(minfo.planecount, minfo.type, x*size, y);
			}
			break
		
		default:
			break;
	}
	mat.usedstdim = 1;
	return mat;
}




function jit_matrix () {
	var mat = new JitterMatrix(arrayfromargs(arguments));

	if(!check_matrix(mat)) {
		index = 0;
		outmatrix = create_matrix(mode, get_info(mat), maximum_size);
	}
	
	var x = (mat.dim[0] === undefined) ? mat.dim : mat.dim[0];
	var y = (mat.dim[1] === undefined) ? 1 : mat.dim[1];
	var xdiff = Math.max(x-1, 1);
	var ydiff = Math.max(y-1, 1);
	var outdim = [];
	
	switch(mode) {
		case 0:
			outmatrix.dstdimstart = [0, index*ydiff];
			outmatrix.dstdimend = [x , index*ydiff+ydiff];
			outdim = [x, index*y+y];
			break;
		case 1: 
			outmatrix.dstdimstart = [0, index*ydiff];
			outmatrix.dstdimend = [x , index*ydiff+ydiff];
			outdim = [x, index*y+y];
			break;
		case 2:
			outmatrix.dstdimstart = [index*xdiff, 0];
			outmatrix.dstdimend = [index*xdiff+xdiff, y];
			outdim = [index*x+x, y];
			break;
		default:
			break;
		
	}
	
	outmatrix.frommatrix(mat);
	outlet(1, 'dim', outdim);
	outlet(0, 'jit_matrix', outmatrix.name);
	index++;
	
	currentinfo = get_info(mat);
}