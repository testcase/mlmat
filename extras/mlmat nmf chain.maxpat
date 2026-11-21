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
         1000.0,
         720.0
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
                  620,
                  20.0
               ],
               "text": "mlmat NMF separation -- the whole chain (pure stock objects)"
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
                  940,
                  48
               ],
               "text": "buffer_stft (polar, @nodc 1) -> magnitudes to mlmat.nfm @mode 1 -> jit.submatrix extracts w_i and h_i (offsets from the number) -> jit.la.mult computes Vhat_i = w_i x h_i; a second jit.la.mult computes T = W*H -> mask: share = Vhat_i / floored T, mag_i = V * share -> jit.pack with the original phases -> buffer_istft. Zeros in V stay zero: the factorization cannot refill the bins @nodc cleaned. Bang once after loading audio; pick a component with the number."
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
                  230,
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
                  25,
                  285,
                  560,
                  20.0
               ],
               "text": "W (frames x rank) and H (rank x bins), mode 1"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12.0,
               "id": "sub_w",
               "maxclass": "newobj",
               "numinlets": 1,
               "numoutlets": 1,
               "outlettype": [
                  "jit_matrix"
               ],
               "patching_rect": [
                  25,
                  320,
                  190,
                  22.0
               ],
               "text": "jit.submatrix @offset 0 0"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12.0,
               "id": "sub_h",
               "maxclass": "newobj",
               "numinlets": 1,
               "numoutlets": 1,
               "outlettype": [
                  "jit_matrix"
               ],
               "patching_rect": [
                  270,
                  320,
                  190,
                  22.0
               ],
               "text": "jit.submatrix @offset 0 0"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12.0,
               "id": "copy_w",
               "maxclass": "newobj",
               "numinlets": 1,
               "numoutlets": 1,
               "outlettype": [
                  "jit_matrix"
               ],
               "patching_rect": [
                  25,
                  360,
                  240,
                  22.0
               ],
               "text": "jit.matrix mlmat.nmf.wi 1 float64 1 1"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12.0,
               "id": "copy_h",
               "maxclass": "newobj",
               "numinlets": 1,
               "numoutlets": 1,
               "outlettype": [
                  "jit_matrix"
               ],
               "patching_rect": [
                  270,
                  360,
                  240,
                  22.0
               ],
               "text": "jit.matrix mlmat.nmf.hi 1 float64 1 1"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 11,
               "id": "n_copy",
               "maxclass": "comment",
               "numinlets": 1,
               "numoutlets": 0,
               "patching_rect": [
                  520,
                  365,
                  380,
                  30
               ],
               "text": "jit.matrix copies: jit.submatrix outputs a REFERENCE, which jit.la.mult may not consume directly (per the jit.submatrix docs)"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12.0,
               "id": "la_comp",
               "maxclass": "newobj",
               "numinlets": 2,
               "numoutlets": 2,
               "outlettype": [
                  "jit_matrix",
                  ""
               ],
               "patching_rect": [
                  270,
                  410,
                  100,
                  22.0
               ],
               "text": "jit.la.mult"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12.0,
               "id": "la_total",
               "maxclass": "newobj",
               "numinlets": 2,
               "numoutlets": 2,
               "outlettype": [
                  "jit_matrix",
                  ""
               ],
               "patching_rect": [
                  270,
                  285,
                  100,
                  22.0
               ],
               "text": "jit.la.mult"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 11,
               "id": "n_la",
               "maxclass": "comment",
               "numinlets": 1,
               "numoutlets": 0,
               "patching_rect": [
                  390,
                  285,
                  470,
                  36
               ],
               "text": "T = W*H direct; Vhat_i = (w_i column) x (h_i row) via the two submatrix slices. If the component comes out transposed, swap the two offset messages below."
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
                  25,
                  410,
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
               "id": "t_num",
               "maxclass": "newobj",
               "numinlets": 1,
               "numoutlets": 3,
               "outlettype": [
                  "int",
                  "int",
                  "int"
               ],
               "patching_rect": [
                  25,
                  445,
                  50,
                  22.0
               ],
               "text": "t i i i"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12.0,
               "id": "msg_w",
               "maxclass": "message",
               "numinlets": 1,
               "numoutlets": 1,
               "outlettype": [
                  ""
               ],
               "patching_rect": [
                  105,
                  445,
                  100,
                  20.0
               ],
               "text": "offset 0 $1"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12.0,
               "id": "msg_h",
               "maxclass": "message",
               "numinlets": 1,
               "numoutlets": 1,
               "outlettype": [
                  ""
               ],
               "patching_rect": [
                  105,
                  480,
                  100,
                  20.0
               ],
               "text": "offset $1 0"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12.0,
               "id": "t_bang",
               "maxclass": "newobj",
               "numinlets": 1,
               "numoutlets": 2,
               "outlettype": [
                  "bang",
                  "bang"
               ],
               "patching_rect": [
                  25,
                  480,
                  50,
                  22.0
               ],
               "text": "t b b"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 11,
               "id": "n_order",
               "maxclass": "comment",
               "numinlets": 1,
               "numoutlets": 0,
               "patching_rect": [
                  105,
                  510,
                  430,
                  30
               ],
               "text": "h_i re-slices first, w_i last: the left-inlet arrival into jit.la.mult is the trigger, so the right input must already be current"
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
                  460,
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
                  510,
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
                  555,
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
                  600,
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
                  645,
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
                  645,
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
                  560,
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
                  605,
                  280,
                  20.0
               ],
               "text": "plane 0 = masked magnitudes, plane 1 = original phases"
            }
         }
      ],
      "lines": [
         {
            "patchline": {
               "destination": [
                  "stft",
                  0
               ],
               "source": [
                  "bang",
                  0
               ]
            }
         },
         {
            "patchline": {
               "destination": [
                  "unpack",
                  0
               ],
               "source": [
                  "stft",
                  0
               ]
            }
         },
         {
            "patchline": {
               "destination": [
                  "nmf",
                  0
               ],
               "source": [
                  "unpack",
                  0
               ]
            }
         },
         {
            "patchline": {
               "destination": [
                  "mask",
                  1
               ],
               "source": [
                  "unpack",
                  0
               ]
            }
         },
         {
            "patchline": {
               "destination": [
                  "pack",
                  1
               ],
               "source": [
                  "unpack",
                  1
               ]
            }
         },
         {
            "patchline": {
               "destination": [
                  "sub_w",
                  0
               ],
               "source": [
                  "nmf",
                  0
               ]
            }
         },
         {
            "patchline": {
               "destination": [
                  "la_total",
                  0
               ],
               "source": [
                  "nmf",
                  0
               ]
            }
         },
         {
            "patchline": {
               "destination": [
                  "sub_h",
                  0
               ],
               "source": [
                  "nmf",
                  1
               ]
            }
         },
         {
            "patchline": {
               "destination": [
                  "la_total",
                  1
               ],
               "source": [
                  "nmf",
                  1
               ]
            }
         },
         {
            "patchline": {
               "destination": [
                  "copy_w",
                  0
               ],
               "source": [
                  "sub_w",
                  0
               ]
            }
         },
         {
            "patchline": {
               "destination": [
                  "copy_h",
                  0
               ],
               "source": [
                  "sub_h",
                  0
               ]
            }
         },
         {
            "patchline": {
               "destination": [
                  "la_comp",
                  0
               ],
               "source": [
                  "copy_w",
                  0
               ]
            }
         },
         {
            "patchline": {
               "destination": [
                  "la_comp",
                  1
               ],
               "source": [
                  "copy_h",
                  0
               ]
            }
         },
         {
            "patchline": {
               "destination": [
                  "div",
                  0
               ],
               "source": [
                  "la_comp",
                  0
               ]
            }
         },
         {
            "patchline": {
               "destination": [
                  "floor",
                  0
               ],
               "source": [
                  "la_total",
                  0
               ]
            }
         },
         {
            "patchline": {
               "destination": [
                  "div",
                  1
               ],
               "source": [
                  "floor",
                  0
               ]
            }
         },
         {
            "patchline": {
               "destination": [
                  "mask",
                  0
               ],
               "source": [
                  "div",
                  0
               ]
            }
         },
         {
            "patchline": {
               "destination": [
                  "pack",
                  0
               ],
               "source": [
                  "mask",
                  0
               ]
            }
         },
         {
            "patchline": {
               "destination": [
                  "istft",
                  0
               ],
               "source": [
                  "pack",
                  0
               ]
            }
         },
         {
            "patchline": {
               "destination": [
                  "t_num",
                  0
               ],
               "source": [
                  "num",
                  0
               ]
            }
         },
         {
            "patchline": {
               "destination": [
                  "msg_w",
                  0
               ],
               "source": [
                  "t_num",
                  2
               ]
            }
         },
         {
            "patchline": {
               "destination": [
                  "msg_h",
                  0
               ],
               "source": [
                  "t_num",
                  1
               ]
            }
         },
         {
            "patchline": {
               "destination": [
                  "sub_w",
                  0
               ],
               "source": [
                  "msg_w",
                  0
               ]
            }
         },
         {
            "patchline": {
               "destination": [
                  "sub_h",
                  0
               ],
               "source": [
                  "msg_h",
                  0
               ]
            }
         },
         {
            "patchline": {
               "destination": [
                  "t_bang",
                  0
               ],
               "source": [
                  "t_num",
                  0
               ]
            }
         },
         {
            "patchline": {
               "destination": [
                  "sub_h",
                  0
               ],
               "source": [
                  "t_bang",
                  1
               ]
            }
         },
         {
            "patchline": {
               "destination": [
                  "sub_w",
                  0
               ],
               "source": [
                  "t_bang",
                  0
               ]
            }
         }
      ],
      "parameters": {},
      "dependency_cache": [],
      "autosave": 0,
      "styles": []
   }
}
