{
   "patcher": {
      "fileversion": 1,
      "appversion": {
         "major": 9,
         "minor": 2,
         "revision": 1,
         "architecture": "x64",
         "modernui": 1
      },
      "classnamespace": "jit",
      "rect": [
         100.0,
         100.0,
         960.0,
         800.0
      ],
      "bglocked": 0,
      "openinpresentation": 0,
      "default_fontsize": 12.0,
      "default_fontface": 0,
      "default_fontname": "Arial",
      "gridsize": [
         15.0,
         15.0
      ],
      "boxes": [
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 16,
               "id": "title",
               "maxclass": "comment",
               "numinlets": 1,
               "numoutlets": 0,
               "patching_rect": [
                  25,
                  18,
                  560,
                  20.0
               ],
               "text": "mlmat NMF separation -- the whole chain"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12,
               "id": "intro",
               "maxclass": "comment",
               "numinlets": 1,
               "numoutlets": 0,
               "patching_rect": [
                  25,
                  45,
                  900,
                  48
               ],
               "text": "buffer_stft (polar, @nodc 1) -> magnitudes to mlmat.nfm @mode 1 -> the js computes Vhat_i and T -> mask: share = Vhat_i / T, mag_i = V * share -> jit.pack with the original phases -> buffer_istft. Zeros in V stay zero: the factorization cannot refill the bins @nodc cleaned. Bang once after loading audio; pick a component with the number."
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12,
               "id": "n_src",
               "maxclass": "comment",
               "numinlets": 1,
               "numoutlets": 0,
               "patching_rect": [
                  25,
                  100,
                  300,
                  20.0
               ],
               "text": "load audio into mlmat.nmf.src, then bang"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12.0,
               "id": "buf_src",
               "maxclass": "newobj",
               "numinlets": 1,
               "numoutlets": 2,
               "outlettype": [
                  "bang",
                  ""
               ],
               "patching_rect": [
                  25,
                  120,
                  170,
                  22.0
               ],
               "text": "buffer~ mlmat.nmf.src 10."
            }
         },
         {
            "box": {
               "id": "bang",
               "maxclass": "button",
               "numinlets": 1,
               "numoutlets": 1,
               "outlettype": [
                  "bang"
               ],
               "patching_rect": [
                  225.0,
                  120.0,
                  24.0,
                  24.0
               ]
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12.0,
               "id": "stft",
               "maxclass": "newobj",
               "numinlets": 2,
               "numoutlets": 2,
               "outlettype": [
                  "jit_matrix",
                  ""
               ],
               "patching_rect": [
                  270,
                  120,
                  560,
                  22.0
               ],
               "text": "mlmat.buffer_stft mlmat.nmf.src @output_polar 1 @nodc 1 @fftsize 1024 @overlap 4"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12.0,
               "id": "unpack",
               "maxclass": "newobj",
               "numinlets": 1,
               "numoutlets": 2,
               "outlettype": [
                  "jit_matrix",
                  "jit_matrix"
               ],
               "patching_rect": [
                  270,
                  175,
                  100,
                  22.0
               ],
               "text": "jit.unpack 2"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12.0,
               "id": "nmf",
               "maxclass": "newobj",
               "numinlets": 1,
               "numoutlets": 2,
               "outlettype": [
                  "jit_matrix",
                  "jit_matrix"
               ],
               "patching_rect": [
                  270,
                  235,
                  200,
                  22.0
               ],
               "text": "mlmat.nmf @rank 3 @mode 1"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 11,
               "id": "n_w",
               "maxclass": "comment",
               "numinlets": 1,
               "numoutlets": 0,
               "patching_rect": [
                  60,
                  315,
                  140,
                  20.0
               ],
               "text": "W (frames x rank) holder"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12.0,
               "id": "w_hold",
               "maxclass": "newobj",
               "numinlets": 1,
               "numoutlets": 1,
               "outlettype": [
                  "jit_matrix"
               ],
               "patching_rect": [
                  60,
                  335,
                  230,
                  22.0
               ],
               "text": "jit.matrix mlmat.nmf.W 1 float64 1 1"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12.0,
               "id": "h_hold",
               "maxclass": "newobj",
               "numinlets": 1,
               "numoutlets": 1,
               "outlettype": [
                  "jit_matrix"
               ],
               "patching_rect": [
                  400,
                  335,
                  230,
                  22.0
               ],
               "text": "jit.matrix mlmat.nmf.H 1 float64 1 1"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 11,
               "id": "n_h",
               "maxclass": "comment",
               "numinlets": 1,
               "numoutlets": 0,
               "patching_rect": [
                  400,
                  315,
                  160,
                  20.0
               ],
               "text": "H (rank x bins) holder"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12.0,
               "id": "t_b",
               "maxclass": "newobj",
               "numinlets": 1,
               "numoutlets": 1,
               "outlettype": [
                  "bang"
               ],
               "patching_rect": [
                  400,
                  395,
                  40,
                  22.0
               ],
               "text": "t b"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12.0,
               "id": "msg_trig",
               "maxclass": "newobj",
               "numinlets": 1,
               "numoutlets": 1,
               "outlettype": [
                  ""
               ],
               "patching_rect": [
                  270,
                  395,
                  150,
                  20.0
               ],
               "text": "total, component"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12.0,
               "id": "num",
               "maxclass": "newobj",
               "numinlets": 1,
               "numoutlets": 2,
               "outlettype": [
                  "",
                  "bang"
               ],
               "patching_rect": [
                  100,
                  395,
                  60,
                  20.0
               ],
               "text": "number 1"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12.0,
               "id": "msg_idx",
               "maxclass": "newobj",
               "numinlets": 1,
               "numoutlets": 1,
               "outlettype": [
                  ""
               ],
               "patching_rect": [
                  100,
                  430,
                  130,
                  20.0
               ],
               "text": "message index $1"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12.0,
               "id": "js",
               "maxclass": "newobj",
               "numinlets": 1,
               "numoutlets": 2,
               "outlettype": [
                  "jit_matrix",
                  "jit_matrix"
               ],
               "patching_rect": [
                  270,
                  440,
                  220,
                  22.0
               ],
               "text": "js mlmat.nmf-component.js"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 11,
               "id": "n_js",
               "maxclass": "comment",
               "numinlets": 1,
               "numoutlets": 0,
               "patching_rect": [
                  500,
                  440,
                  260,
                  20.0
               ],
               "text": "outlet 1: Vhat_i; outlet 2: T = W*H (slow: cell-wise js)"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12.0,
               "id": "floor",
               "maxclass": "newobj",
               "numinlets": 1,
               "numoutlets": 2,
               "outlettype": [
                  "jit_matrix",
                  "jit_matrix"
               ],
               "patching_rect": [
                  270,
                  500,
                  170,
                  22.0
               ],
               "text": "jit.op @op + @val 1e-10"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12.0,
               "id": "div",
               "maxclass": "newobj",
               "numinlets": 2,
               "numoutlets": 2,
               "outlettype": [
                  "jit_matrix",
                  "jit_matrix"
               ],
               "patching_rect": [
                  270,
                  550,
                  100,
                  22.0
               ],
               "text": "jit.op @op div"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12.0,
               "id": "mask",
               "maxclass": "newobj",
               "numinlets": 2,
               "numoutlets": 2,
               "outlettype": [
                  "jit_matrix",
                  "jit_matrix"
               ],
               "patching_rect": [
                  270,
                  600,
                  100,
                  22.0
               ],
               "text": "jit.op @op *"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12.0,
               "id": "pack",
               "maxclass": "newobj",
               "numinlets": 2,
               "numoutlets": 2,
               "outlettype": [
                  "jit_matrix",
                  "jit_matrix"
               ],
               "patching_rect": [
                  270,
                  650,
                  100,
                  22.0
               ],
               "text": "jit.pack 2"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12.0,
               "id": "istft",
               "maxclass": "newobj",
               "numinlets": 2,
               "numoutlets": 2,
               "outlettype": [
                  "jit_matrix",
                  ""
               ],
               "patching_rect": [
                  270,
                  700,
                  520,
                  22.0
               ],
               "text": "mlmat.buffer_istft mlmat.nmf.out @input_polar 1 @fftsize 1024 @overlap 4"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12.0,
               "id": "buf_out",
               "maxclass": "newobj",
               "numinlets": 1,
               "numoutlets": 2,
               "outlettype": [
                  "bang",
                  ""
               ],
               "patching_rect": [
                  25,
                  700,
                  170,
                  22.0
               ],
               "text": "buffer~ mlmat.nmf.out 10."
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 11,
               "id": "n_mask",
               "maxclass": "comment",
               "numinlets": 1,
               "numoutlets": 0,
               "patching_rect": [
                  640,
                  605,
                  280,
                  20.0
               ],
               "text": "mask: mag_i = V * (Vhat_i / T) -- V's zeros are authoritative"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 11,
               "id": "n_pack",
               "maxclass": "comment",
               "numinlets": 1,
               "numoutlets": 0,
               "patching_rect": [
                  640,
                  655,
                  280,
                  20.0
               ],
               "text": "plane 0 = masked magnitudes, plane 1 = original phases"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 11,
               "id": "n_out",
               "maxclass": "comment",
               "numinlets": 1,
               "numoutlets": 0,
               "patching_rect": [
                  25,
                  735,
                  900,
                  20.0
               ],
               "text": "Compare against the fragment (extras/mlmat nmf masking.maxpat): same three jit.ops. Wiener variant: square Vhat_i and T before the div."
            }
         }
      ],
      "lines": [
         {
            "destination": {
               "id": "stft",
               "inlet": 0
            },
            "source": {
               "id": "bang",
               "outlet": 0
            }
         },
         {
            "destination": {
               "id": "unpack",
               "inlet": 0
            },
            "source": {
               "id": "stft",
               "outlet": 0
            }
         },
         {
            "destination": {
               "id": "nmf",
               "inlet": 0
            },
            "source": {
               "id": "unpack",
               "outlet": 0
            }
         },
         {
            "destination": {
               "id": "mask",
               "inlet": 1
            },
            "source": {
               "id": "unpack",
               "outlet": 0
            }
         },
         {
            "destination": {
               "id": "pack",
               "inlet": 1
            },
            "source": {
               "id": "unpack",
               "outlet": 1
            }
         },
         {
            "destination": {
               "id": "w_hold",
               "inlet": 0
            },
            "source": {
               "id": "nmf",
               "outlet": 0
            }
         },
         {
            "destination": {
               "id": "h_hold",
               "inlet": 0
            },
            "source": {
               "id": "nmf",
               "outlet": 1
            }
         },
         {
            "destination": {
               "id": "t_b",
               "inlet": 0
            },
            "source": {
               "id": "h_hold",
               "outlet": 0
            }
         },
         {
            "destination": {
               "id": "msg_trig",
               "inlet": 0
            },
            "source": {
               "id": "t_b",
               "outlet": 0
            }
         },
         {
            "destination": {
               "id": "js",
               "inlet": 0
            },
            "source": {
               "id": "msg_trig",
               "outlet": 0
            }
         },
         {
            "destination": {
               "id": "msg_idx",
               "inlet": 0
            },
            "source": {
               "id": "num",
               "outlet": 0
            }
         },
         {
            "destination": {
               "id": "js",
               "inlet": 0
            },
            "source": {
               "id": "msg_idx",
               "outlet": 0
            }
         },
         {
            "destination": {
               "id": "floor",
               "inlet": 0
            },
            "source": {
               "id": "js",
               "outlet": 1
            }
         },
         {
            "destination": {
               "id": "div",
               "inlet": 1
            },
            "source": {
               "id": "floor",
               "outlet": 0
            }
         },
         {
            "destination": {
               "id": "div",
               "inlet": 0
            },
            "source": {
               "id": "js",
               "outlet": 0
            }
         },
         {
            "destination": {
               "id": "mask",
               "inlet": 0
            },
            "source": {
               "id": "div",
               "outlet": 0
            }
         },
         {
            "destination": {
               "id": "pack",
               "inlet": 0
            },
            "source": {
               "id": "mask",
               "outlet": 0
            }
         },
         {
            "destination": {
               "id": "istft",
               "inlet": 0
            },
            "source": {
               "id": "pack",
               "outlet": 0
            }
         }
      ],
      "parameters": {},
      "dependency_cache": [],
      "autosave": 0,
      "styles": []
   }
}
