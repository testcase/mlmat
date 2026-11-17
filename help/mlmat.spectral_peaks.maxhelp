{
 "patcher": {
  "fileversion": 1,
  "appversion": {
   "major": 9,
   "minor": 0,
   "revision": 0,
   "architecture": "x64",
   "modernui": 1
  },
  "classnamespace": "box",
  "rect": [
   45.0,
   89.0,
   700.0,
   700.0
  ],
  "bglocked": 0,
  "openinpresentation": 0,
  "default_fontsize": 13.0,
  "default_fontface": 0,
  "default_fontname": "Arial",
  "gridonopen": 1,
  "gridsize": [
   5.0,
   5.0
  ],
  "gridsnaponopen": 1,
  "objectsnaponopen": 1,
  "statusbarvisible": 2,
  "toolbarvisible": 1,
  "lefttoolbarpinned": 0,
  "toptoolbarpinned": 0,
  "righttoolbarpinned": 0,
  "bottomtoolbarpinned": 0,
  "toolbars_unpinned_last_save": 0,
  "tallnewobj": 0,
  "boxanimatetime": 200,
  "enablehscroll": 1,
  "enablevscroll": 1,
  "devicewidth": 0.0,
  "description": "",
  "digest": "",
  "tags": "",
  "style": "",
  "subpatcher_template": "",
  "showrootpatcherontab": 0,
  "showontab": 0,
  "assistshowspatchername": 0,
  "lines": [],
  "autosave": 0,
  "boxes": [
   {
    "box": {
     "id": "obj-1",
     "maxclass": "newobj",
     "numinlets": 1,
     "numoutlets": 1,
     "outlettype": [
      ""
     ],
     "patching_rect": [
      445.0,
      350.0,
      200.0,
      23.0
     ],
     "saved_object_attributes": {
      "filename": "helpstarter",
      "parameter_enable": 0
     },
     "text": "js helpstarter mlmat.spectral_peaks",
     "fontname": "Arial",
     "fontsize": 13.0
    }
   },
   {
    "box": {
     "id": "obj-5",
     "maxclass": "newobj",
     "numinlets": 0,
     "numoutlets": 0,
     "patching_rect": [
      0.0,
      0.0,
      0.0,
      0.0
     ],
     "text": "p basic",
     "varname": "basic_tab",
     "patcher": {
      "fileversion": 1,
      "rect": [
       0.0,
       0.0,
       640.0,
       480.0
      ],
      "bglocked": 0,
      "openinpresentation": 0,
      "default_fontsize": 13.0,
      "default_fontface": 0,
      "default_fontname": "Arial",
      "gridonopen": 1,
      "gridsize": [
       5.0,
       5.0
      ],
      "gridsnaponopen": 1,
      "objectsnaponopen": 1,
      "statusbarvisible": 2,
      "toolbarvisible": 1,
      "lefttoolbarpinned": 0,
      "toptoolbarpinned": 0,
      "righttoolbarpinned": 0,
      "bottomtoolbarpinned": 0,
      "toolbars_unpinned_last_save": 0,
      "tallnewobj": 0,
      "boxanimatetime": 200,
      "enablehscroll": 1,
      "enablevscroll": 1,
      "devicewidth": 0.0,
      "description": "",
      "digest": "",
      "tags": "",
      "style": "",
      "subpatcher_template": "",
      "showontab": 0,
      "showrootpatcherontab": 0,
      "assistshowspatchername": 0,
      "lines": [],
      "autosave": 0,
      "boxes": [
       {
        "box": {
         "maxclass": "comment",
         "text": "Extract spectral peaks from an STFT analysis",
         "id": "c-10-10",
         "numinlets": 1,
         "numoutlets": 0,
         "patching_rect": [
          10.0,
          10.0,
          350.0,
          20.0
         ],
         "fontname": "Arial",
         "fontsize": 13.0
        }
       },
       {
        "box": {
         "maxclass": "comment",
         "text": "Feed the output of mlmat.buffer_stft",
         "id": "c-10-30",
         "numinlets": 1,
         "numoutlets": 0,
         "patching_rect": [
          10.0,
          30.0,
          300.0,
          20.0
         ],
         "fontname": "Arial",
         "fontsize": 13.0
        }
       },
       {
        "box": {
         "maxclass": "comment",
         "text": "peaks mode: plane 0 = amplitudes, plane 1 = frequencies (Hz)",
         "id": "c-10-50",
         "numinlets": 1,
         "numoutlets": 0,
         "patching_rect": [
          10.0,
          50.0,
          380.0,
          20.0
         ],
         "fontname": "Arial",
         "fontsize": 13.0
        }
       },
       {
        "box": {
         "maxclass": "newobj",
         "text": "mlmat.buffer_stft stft_help @fftsize 1024",
         "id": "box-10-80",
         "numinlets": 1,
         "numoutlets": 1,
         "outlettype": [
          ""
         ],
         "patching_rect": [
          10.0,
          80.0,
          250.0,
          23.0
         ],
         "fontname": "Arial",
         "fontsize": 13.0
        }
       },
       {
        "box": {
         "maxclass": "newobj",
         "text": "mlmat.spectral_peaks @mode peaks @max_peaks 8",
         "id": "box-10-130",
         "numinlets": 1,
         "numoutlets": 1,
         "outlettype": [
          ""
         ],
         "patching_rect": [
          10.0,
          130.0,
          280.0,
          23.0
         ],
         "fontname": "Arial",
         "fontsize": 13.0
        }
       },
       {
        "box": {
         "maxclass": "jit.pwindow",
         "id": "pw-300-130",
         "numinlets": 1,
         "numoutlets": 2,
         "outlettype": [
          "jit_matrix",
          ""
         ],
         "patching_rect": [
          300.0,
          130.0,
          160.0,
          120.0
         ]
        }
       },
       {
        "box": {
         "maxclass": "newobj",
         "text": "jit.cellblock",
         "id": "box-10-260",
         "numinlets": 1,
         "numoutlets": 1,
         "outlettype": [
          ""
         ],
         "patching_rect": [
          10.0,
          260.0,
          200.0,
          22.0
         ],
         "fontname": "Arial",
         "fontsize": 13.0
        }
       },
       {
        "box": {
         "maxclass": "comment",
         "text": "bins mode: plane 0 = magnitudes, plane 1 = center frequency",
         "id": "c-10-290",
         "numinlets": 1,
         "numoutlets": 0,
         "patching_rect": [
          10.0,
          290.0,
          380.0,
          20.0
         ],
         "fontname": "Arial",
         "fontsize": 13.0
        }
       },
       {
        "box": {
         "maxclass": "attrui",
         "attr": "mode",
         "id": "obj-attr-0",
         "numinlets": 1,
         "numoutlets": 1,
         "outlettype": [
          ""
         ],
         "patching_rect": [
          10.0,
          350.0,
          200.0,
          22.0
         ],
         "parameter_enable": 0,
         "fontname": "Arial",
         "fontsize": 13.0
        }
       },
       {
        "box": {
         "maxclass": "attrui",
         "attr": "samplerate",
         "id": "obj-attr-1",
         "numinlets": 1,
         "numoutlets": 1,
         "outlettype": [
          ""
         ],
         "patching_rect": [
          10.0,
          372.0,
          200.0,
          22.0
         ],
         "parameter_enable": 0,
         "fontname": "Arial",
         "fontsize": 13.0
        }
       },
       {
        "box": {
         "maxclass": "attrui",
         "attr": "max_peaks",
         "id": "obj-attr-2",
         "numinlets": 1,
         "numoutlets": 1,
         "outlettype": [
          ""
         ],
         "patching_rect": [
          10.0,
          394.0,
          200.0,
          22.0
         ],
         "parameter_enable": 0,
         "fontname": "Arial",
         "fontsize": 13.0
        }
       },
       {
        "box": {
         "maxclass": "attrui",
         "attr": "threshold",
         "id": "obj-attr-3",
         "numinlets": 1,
         "numoutlets": 1,
         "outlettype": [
          ""
         ],
         "patching_rect": [
          10.0,
          416.0,
          200.0,
          22.0
         ],
         "parameter_enable": 0,
         "fontname": "Arial",
         "fontsize": 13.0
        }
       },
       {
        "box": {
         "maxclass": "attrui",
         "attr": "input_polar",
         "id": "obj-attr-4",
         "numinlets": 1,
         "numoutlets": 1,
         "outlettype": [
          ""
         ],
         "patching_rect": [
          10.0,
          438.0,
          200.0,
          22.0
         ],
         "parameter_enable": 0,
         "fontname": "Arial",
         "fontsize": 13.0
        }
       }
      ],
      "dependency_cache": [],
      "bgfillcolor_type": "color",
      "bgfillcolor_color": [
       1.0,
       1.0,
       1.0,
       1.0
      ],
      "bgfillcolor_color1": [
       0.376471,
       0.384314,
       0.4,
       1.0
      ],
      "bgfillcolor_color2": [
       0.290196,
       0.309804,
       0.341176,
       1.0
      ],
      "bgfillcolor_angle": 270.0
     }
    }
   },
   {
    "box": {
     "id": "obj-7",
     "maxclass": "jsui",
     "border": 0,
     "filename": "helpname.js",
     "ignoreclick": 1,
     "jsarguments": [
      "mlmat.spectral_peaks"
     ],
     "numinlets": 1,
     "numoutlets": 1,
     "outlettype": [
      ""
     ],
     "parameter_enable": 0,
     "patching_rect": [
      10.0,
      10.0,
      325.83203125,
      57.5998535158
     ]
    }
   },
   {
    "box": {
     "id": "obj-2",
     "maxclass": "newobj",
     "numinlets": 0,
     "numoutlets": 0,
     "patching_rect": [
      0.0,
      0.0,
      0.0,
      0.0
     ],
     "text": "p ?",
     "varname": "q_tab",
     "patcher": {
      "fileversion": 1,
      "rect": [
       0.0,
       0.0,
       640.0,
       480.0
      ],
      "boxes": [],
      "lines": [],
      "autosave": 0
     }
    }
   }
  ],
  "dependency_cache": [
   {
    "name": "helpdetails.js",
    "bootpath": "C74:/help/resources",
    "type": "TEXT",
    "implicit": 1
   },
   {
    "name": "helpname.js",
    "bootpath": "C74:/help/resources",
    "type": "TEXT",
    "implicit": 1
   },
   {
    "name": "helpstarter.js",
    "bootpath": "C74:/help/resources",
    "type": "TEXT",
    "implicit": 1
   },
   {
    "name": "mlmat.spectral_peaks.mxo",
    "type": "iLaX"
   }
  ]
 }
}