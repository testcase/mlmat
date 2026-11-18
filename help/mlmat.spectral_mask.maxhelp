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
        "classnamespace": "box",
        "rect": [
            134.0,
            164.0,
            700.0,
            700.0
        ],
        "default_fontsize": 13.0,
        "gridsize": [
            5.0,
            5.0
        ],
        "showrootpatcherontab": 0,
        "showontab": 0,
        "boxes": [
            {
                "box": {
                    "fontname": "Arial",
                    "fontsize": 13.0,
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
                        213.0,
                        23.0
                    ],
                    "saved_object_attributes": {
                        "filename": "helpstarter",
                        "parameter_enable": 0
                    },
                    "text": "js helpstarter mlmat.spectral_mask"
                }
            },
            {
                "box": {
                    "id": "obj-5",
                    "maxclass": "newobj",
                    "numinlets": 0,
                    "numoutlets": 0,
                    "patcher": {
                        "fileversion": 1,
                        "appversion": {
                            "major": 9,
                            "minor": 2,
                            "revision": 1,
                            "architecture": "x64",
                            "modernui": 1
                        },
                        "classnamespace": "box",
                        "rect": [
                            134.0,
                            190.0,
                            700.0,
                            674.0
                        ],
                        "default_fontsize": 13.0,
                        "gridsize": [
                            5.0,
                            5.0
                        ],
                        "showrootpatcherontab": 0,
                        "showontab": 1,
                        "boxes": [
                            {
                                "box": {
                                    "filename": "helpdetails.js",
                                    "id": "obj-2",
                                    "ignoreclick": 1,
                                    "jsarguments": [
                                        "mlmat.spectral_mask"
                                    ],
                                    "maxclass": "jsui",
                                    "numinlets": 1,
                                    "numoutlets": 1,
                                    "outlettype": [
                                        ""
                                    ],
                                    "parameter_enable": 0,
                                    "patching_rect": [
                                        10.0,
                                        10.0,
                                        584.0,
                                        143.0
                                    ]
                                }
                            },
                            {
                                "box": {
                                    "maxclass": "comment",
                                    "id": "c-10-10",
                                    "numinlets": 1,
                                    "numoutlets": 0,
                                    "fontname": "Arial",
                                    "fontsize": 13.0,
                                    "patching_rect": [
                                        43.0,
                                        249.0,
                                        350.0,
                                        21.0
                                    ],
                                    "text": "Bin-wise masking of one spectral analysis by another"
                                }
                            },
                            {
                                "box": {
                                    "maxclass": "comment",
                                    "id": "c-10-30",
                                    "numinlets": 1,
                                    "numoutlets": 0,
                                    "fontname": "Arial",
                                    "fontsize": 13.0,
                                    "patching_rect": [
                                        43.0,
                                        269.0,
                                        300.0,
                                        21.0
                                    ],
                                    "text": "Feed the outputs of two mlmat.buffer_stft objects"
                                }
                            },
                            {
                                "box": {
                                    "maxclass": "comment",
                                    "id": "c-10-50",
                                    "numinlets": 1,
                                    "numoutlets": 0,
                                    "fontname": "Arial",
                                    "fontsize": 13.0,
                                    "patching_rect": [
                                        43.0,
                                        289.0,
                                        380.0,
                                        21.0
                                    ],
                                    "text": "plane 0/1 keep the input format; output feeds mlmat.buffer_istft"
                                }
                            },
                            {
                                "box": {
                                    "fontname": "Arial",
                                    "fontsize": 13.0,
                                    "id": "box-10-80",
                                    "maxclass": "newobj",
                                    "numinlets": 1,
                                    "numoutlets": 2,
                                    "outlettype": [
                                        "jit_matrix",
                                        ""
                                    ],
                                    "patching_rect": [
                                        43.0,
                                        319.0,
                                        250.0,
                                        23.0
                                    ],
                                    "text": "mlmat.buffer_stft stft_help @fftsize 1024"
                                }
                            },
                            {
                                "box": {
                                    "fontname": "Arial",
                                    "fontsize": 13.0,
                                    "id": "box-10-90",
                                    "maxclass": "newobj",
                                    "numinlets": 1,
                                    "numoutlets": 2,
                                    "outlettype": [
                                        "jit_matrix",
                                        ""
                                    ],
                                    "patching_rect": [
                                        43.0,
                                        344.0,
                                        265.0,
                                        23.0
                                    ],
                                    "text": "mlmat.buffer_stft stft_masker_help @fftsize 1024"
                                }
                            },
                            {
                                "box": {
                                    "fontname": "Arial",
                                    "fontsize": 13.0,
                                    "id": "box-10-130",
                                    "maxclass": "newobj",
                                    "numinlets": 2,
                                    "numoutlets": 2,
                                    "outlettype": [
                                        "jit_matrix",
                                        ""
                                    ],
                                    "patching_rect": [
                                        43.0,
                                        369.0,
                                        317.0,
                                        23.0
                                    ],
                                    "text": "mlmat.spectral_mask @mode mag_from"
                                }
                            },
                            {
                                "box": {
                                    "depthbuffer": 1,
                                    "doublebuffer": 1,
                                    "fsaa": 0,
                                    "id": "pw-300-130",
                                    "maxclass": "jit.pwindow",
                                    "numinlets": 1,
                                    "numoutlets": 2,
                                    "outlettype": [
                                        "jit_matrix",
                                        ""
                                    ],
                                    "patching_rect": [
                                        333.0,
                                        369.0,
                                        160.0,
                                        120.0
                                    ],
                                    "pwinoutlettype": 0,
                                    "stereo": 0,
                                    "sync": 1
                                }
                            },
                            {
                                "box": {
                                    "fontface": 0,
                                    "fontname": "Arial",
                                    "fontsize": 13.0,
                                    "id": "box-10-260",
                                    "maxclass": "jit.cellblock",
                                    "numinlets": 2,
                                    "numoutlets": 4,
                                    "outlettype": [
                                        "list",
                                        "",
                                        "",
                                        ""
                                    ],
                                    "patching_rect": [
                                        43.0,
                                        499.0,
                                        200.0,
                                        200.0
                                    ]
                                }
                            },
                            {
                                "box": {
                                    "attr": "mode",
                                    "fontname": "Arial",
                                    "fontsize": 13.0,
                                    "id": "obj-attr-0",
                                    "maxclass": "attrui",
                                    "numinlets": 1,
                                    "numoutlets": 1,
                                    "outlettype": [
                                        ""
                                    ],
                                    "parameter_enable": 0,
                                    "patching_rect": [
                                        43.0,
                                        589.0,
                                        200.0,
                                        23.0
                                    ]
                                }
                            },
                            {
                                "box": {
                                    "attr": "threshold",
                                    "fontname": "Arial",
                                    "fontsize": 13.0,
                                    "id": "obj-attr-1",
                                    "maxclass": "attrui",
                                    "numinlets": 1,
                                    "numoutlets": 1,
                                    "outlettype": [
                                        ""
                                    ],
                                    "parameter_enable": 0,
                                    "patching_rect": [
                                        43.0,
                                        611.0,
                                        200.0,
                                        23.0
                                    ]
                                }
                            },
                            {
                                "box": {
                                    "attr": "input_polar",
                                    "fontname": "Arial",
                                    "fontsize": 13.0,
                                    "id": "obj-attr-2",
                                    "maxclass": "attrui",
                                    "numinlets": 1,
                                    "numoutlets": 1,
                                    "outlettype": [
                                        ""
                                    ],
                                    "parameter_enable": 0,
                                    "patching_rect": [
                                        43.0,
                                        633.0,
                                        200.0,
                                        23.0
                                    ]
                                }
                            },
                            {
                                "box": {
                                    "attr": "full_spectrum",
                                    "fontname": "Arial",
                                    "fontsize": 13.0,
                                    "id": "obj-attr-3",
                                    "maxclass": "attrui",
                                    "numinlets": 1,
                                    "numoutlets": 1,
                                    "outlettype": [
                                        ""
                                    ],
                                    "parameter_enable": 0,
                                    "patching_rect": [
                                        43.0,
                                        655.0,
                                        200.0,
                                        23.0
                                    ]
                                }
                            }
                        ],
                        "lines": [],
                        "bgfillcolor_type": "color",
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
                        "bgfillcolor_color": [
                            1.0,
                            1.0,
                            1.0,
                            1.0
                        ]
                    },
                    "patching_rect": [
                        114.0,
                        126.0,
                        50.0,
                        23.0
                    ],
                    "saved_object_attributes": {
                        "fontsize": 13.0
                    },
                    "text": "p basic",
                    "varname": "basic_tab"
                }
            },
            {
                "box": {
                    "border": 0,
                    "filename": "helpname.js",
                    "id": "obj-7",
                    "ignoreclick": 1,
                    "jsarguments": [
                        "mlmat.spectral_mask"
                    ],
                    "maxclass": "jsui",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [
                        ""
                    ],
                    "parameter_enable": 0,
                    "patching_rect": [
                        10.0,
                        10.0,
                        456.10406494140625,
                        57.599853515625
                    ]
                }
            },
            {
                "box": {
                    "id": "obj-2",
                    "maxclass": "newobj",
                    "numinlets": 0,
                    "numoutlets": 0,
                    "patcher": {
                        "fileversion": 1,
                        "appversion": {
                            "major": 9,
                            "minor": 2,
                            "revision": 1,
                            "architecture": "x64",
                            "modernui": 1
                        },
                        "classnamespace": "box",
                        "rect": [
                            0.0,
                            26.0,
                            700.0,
                            674.0
                        ],
                        "showontab": 1,
                        "boxes": [],
                        "lines": []
                    },
                    "patching_rect": [
                        285.0,
                        236.0,
                        27.0,
                        23.0
                    ],
                    "text": "p ?",
                    "varname": "q_tab"
                }
            }
        ],
        "lines": [],
        "autosave": 0
    }
}
