// mlmat.nmf-component.js -- companion to the "mlmat nmf chain" example
// patch. Computes the per-component reconstruction Vhat_i = w_i x h_i
// and the total T = W*H from the named matrices mlmat.nmf.W
// (frames x rank) and mlmat.nmf.H (rank x bins), as emitted by
// mlmat.nfm @mode 1. Layout-agnostic: whatever the nmf object's input
// layout is, W's dim0 and H's dim1 define the output frame/bins.
//
//   index i        set the component (1-based; also the arguments of
//                  the component message)
//   total          compute T = W*H   -> outlet 2 (jit_matrix)
//   component [i]  compute Vhat_i    -> outlet 1 (jit_matrix)
//
// NOTE: cell-by-cell JitterMatrix access from js is slow (~seconds at
// 1300 x 512). The example is an offline workflow: bang the analysis,
// pick a component, listen.

inlets = 1;
outlets = 2;

var idx = 1;
var comp = new JitterMatrix(1, "float64", [1, 1]);
var tot = new JitterMatrix(1, "float64", [1, 1]);

function index(i) { idx = i; }

function total() {
    var W = new JitterMatrix("mlmat.nmf.W");
    var H = new JitterMatrix("mlmat.nmf.H");
    var nf = W.dim[0];
    var r = W.dim[1];
    var nb = H.dim[1];
    tot.dim = [nf, nb];
    for (var f = 0; f < nf; f++) {
        for (var b = 0; b < nb; b++) {
            var s = 0;
            for (var i = 0; i < r; i++)
                s += W.getcell(f, i)[0] * H.getcell(i, b)[0];
            tot.setcell2d(f, b, s);
        }
    }
    outlet(1, "jit_matrix", tot.getname());
}

function component(i) {
    if (arguments.length)
        idx = i;
    var W = new JitterMatrix("mlmat.nmf.W");
    var H = new JitterMatrix("mlmat.nmf.H");
    var nf = W.dim[0];
    var nb = H.dim[1];
    comp.dim = [nf, nb];
    for (var f = 0; f < nf; f++) {
        var wv = W.getcell(f, idx)[0];
        for (var b = 0; b < nb; b++)
            comp.setcell2d(f, b, wv * H.getcell(idx, b)[0]);
    }
    outlet(0, "jit_matrix", comp.getname());
}
