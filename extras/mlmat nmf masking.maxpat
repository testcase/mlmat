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
         740.0,
         430.0
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
               "fontsize": 16.0,
               "id": "title",
               "maxclass": "comment",
               "numinlets": 1,
               "numoutlets": 0,
               "patching_rect": [
                  25.0,
                  20.0,
                  460.0,
                  20.0
               ],
               "text": "NMF component masking -- drop-in fragment (mlmat)"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12.0,
               "id": "intro",
               "maxclass": "comment",
               "numinlets": 1,
               "numoutlets": 0,
               "patching_rect": [
                  25.0,
                  50.0,
                  540.0,
                  20.0
               ],
               "text": "Mask, don't reconstruct: W*H only computes the component's SHARE of the modeled energy; the data comes from the original magnitudes. Where V is zero (bins 0-1, cleaned by buffer_stft @nodc 1) the product stays exactly zero -- W*H alone refills those bins with spurious positives that resynthesize as DC."
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12.0,
               "id": "lbl_total",
               "maxclass": "comment",
               "numinlets": 1,
               "numoutlets": 0,
               "patching_rect": [
                  30.0,
                  120.0,
                  250.0,
                  20.0
               ],
               "text": "<-- TOTAL  T = sum of all components (or W*H)"
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
               "patching_rect": [
                  300.0,
                  115.0,
                  170.0,
                  22.0
               ],
               "text": "jit.op @op + @val 1e-10",
               "outlettype": [
                  "jit_matrix",
                  "jit_matrix"
               ]
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 11.0,
               "id": "note_floor",
               "maxclass": "comment",
               "numinlets": 1,
               "numoutlets": 0,
               "patching_rect": [
                  480.0,
                  120.0,
                  220.0,
                  20.0
               ],
               "text": "floor T: the div would NaN where all components ~ 0"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12.0,
               "id": "lbl_component",
               "maxclass": "comment",
               "numinlets": 1,
               "numoutlets": 0,
               "patching_rect": [
                  30.0,
                  180.0,
                  250.0,
                  20.0
               ],
               "text": "<-- COMPONENT  Vhat_i (your W x H step)"
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
               "patching_rect": [
                  300.0,
                  175.0,
                  170.0,
                  22.0
               ],
               "text": "jit.op @op div",
               "outlettype": [
                  "jit_matrix",
                  "jit_matrix"
               ]
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 11.0,
               "id": "note_div",
               "maxclass": "comment",
               "numinlets": 1,
               "numoutlets": 0,
               "patching_rect": [
                  480.0,
                  180.0,
                  220.0,
                  20.0
               ],
               "text": "share_i = Vhat_i / T: the component's proportion per bin/frame"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12.0,
               "id": "lbl_v",
               "maxclass": "comment",
               "numinlets": 1,
               "numoutlets": 0,
               "patching_rect": [
                  30.0,
                  240.0,
                  250.0,
                  20.0
               ],
               "text": "<-- V  ORIGINAL magnitudes (buffer_stft plane 0)"
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
               "patching_rect": [
                  300.0,
                  235.0,
                  170.0,
                  22.0
               ],
               "text": "jit.op @op *",
               "outlettype": [
                  "jit_matrix",
                  "jit_matrix"
               ]
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 11.0,
               "id": "note_mask",
               "maxclass": "comment",
               "numinlets": 1,
               "numoutlets": 0,
               "patching_rect": [
                  480.0,
                  240.0,
                  220.0,
                  20.0
               ],
               "text": "mag_i = V * share_i -- zeros in V are authoritative"
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12.0,
               "id": "lbl_phases",
               "maxclass": "comment",
               "numinlets": 1,
               "numoutlets": 0,
               "patching_rect": [
                  30.0,
                  300.0,
                  250.0,
                  20.0
               ],
               "text": "<-- PHASES  original plane 1, untouched"
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
               "patching_rect": [
                  300.0,
                  295.0,
                  170.0,
                  22.0
               ],
               "text": "jit.pack 2",
               "outlettype": [
                  "jit_matrix",
                  "jit_matrix"
               ]
            }
         },
         {
            "box": {
               "fontname": "Arial",
               "fontsize": 12.0,
               "id": "out",
               "maxclass": "comment",
               "numinlets": 1,
               "numoutlets": 0,
               "patching_rect": [
                  25.0,
                  340.0,
                  540.0,
                  20.0
               ],
               "text": "Out: 2-plane (masked magnitudes, phases) --> buffer_istft. Soft (Wiener) variant: square the component and T (jit.op @op * each with itself) before the div for smoother separation."
            }
         }
      ],
      "lines": [
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
         }
      ],
      "parameters": {
         "obj-..": {}
      },
      "dependency_cache": [],
      "autosave": 0,
      "styles": []
   }
}
