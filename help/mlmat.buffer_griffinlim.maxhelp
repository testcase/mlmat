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
        "rect": [ 100.0, 100.0, 700.0, 700.0 ],
        "default_fontsize": 13.0,
        "gridsize": [ 5.0, 5.0 ],
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
                    "outlettype": [ "" ],
                    "patching_rect": [ 445.0, 350.0, 214.0, 23.0 ],
                    "saved_object_attributes": {
                        "filename": "helpstarter",
                        "parameter_enable": 0
                    },
                    "text": "js helpstarter mlmat.buffer_griffinlim"
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
                        "rect": [ 100.0, 126.0, 700.0, 674.0 ],
                        "default_fontsize": 13.0,
                        "gridsize": [ 5.0, 5.0 ],
                        "showrootpatcherontab": 0,
                        "showontab": 1,
                        "boxes": [
                            {
                                "box": {
                                    "border": 0,
                                    "filename": "helpdetails.js",
                                    "id": "obj-2",
                                    "ignoreclick": 1,
                                    "jsarguments": [ "mlmat.buffer_griffinlim" ],
                                    "maxclass": "jsui",
                                    "numinlets": 1,
                                    "numoutlets": 1,
                                    "outlettype": [ "" ],
                                    "parameter_enable": 0,
                                    "patching_rect": [ 7.0, 15.0, 584.0, 143.0 ]
                                }
                            },
                            {
                                "box": {
                                    "fontname": "Arial",
                                    "fontsize": 13.0,
                                    "id": "c-10-10",
                                    "maxclass": "comment",
                                    "numinlets": 1,
                                    "numoutlets": 0,
                                    "patching_rect": [ 929.0, 319.0, 380.0, 21.0 ],
                                    "text": "Feed a magnitude-only matrix from mlmat.buffer_stft"
                                }
                            },
                            {
                                "box": {
                                    "fontname": "Arial",
                                    "fontsize": 13.0,
                                    "id": "c-10-30",
                                    "maxclass": "comment",
                                    "numinlets": 1,
                                    "numoutlets": 0,
                                    "patching_rect": [ 929.0, 339.0, 380.0, 21.0 ],
                                    "text": "Griffin-Lim iteratively reconstructs the missing phases"
                                }
                            },
                            {
                                "box": {
                                    "fontname": "Arial",
                                    "fontsize": 13.0,
                                    "id": "c-10-100",
                                    "maxclass": "comment",
                                    "numinlets": 1,
                                    "numoutlets": 0,
                                    "patching_rect": [ 42.0, 324.0, 200.0, 21.0 ],
                                    "text": "buffer~ name as first argument"
                                }
                            },
                            {
                                "box": {
                                    "fontname": "Arial",
                                    "fontsize": 13.0,
                                    "id": "box-10-125",
                                    "maxclass": "newobj",
                                    "numinlets": 1,
                                    "numoutlets": 2,
                                    "outlettype": [ "float", "bang" ],
                                    "patching_rect": [ 42.0, 349.0, 160.0, 23.0 ],
                                    "text": "buffer~ gl_help 22050"
                                }
                            },
                            {
                                "box": {
                                    "fontname": "Arial",
                                    "fontsize": 13.0,
                                    "id": "box-10-155",
                                    "maxclass": "newobj",
                                    "numinlets": 1,
                                    "numoutlets": 2,
                                    "outlettype": [ "jit_matrix", "" ],
                                    "patching_rect": [ 42.0, 379.0, 258.0, 23.0 ],
                                    "text": "mlmat.buffer_stft stft_help @output_polar 1"
                                }
                            },
                            {
                                "box": {
                                    "depthbuffer": 1,
                                    "doublebuffer": 1,
                                    "fsaa": 0,
                                    "id": "pw-270-155",
                                    "maxclass": "jit.pwindow",
                                    "numinlets": 1,
                                    "numoutlets": 2,
                                    "outlettype": [ "jit_matrix", "" ],
                                    "patching_rect": [ 302.0, 379.0, 160.0, 120.0 ],
                                    "pwinoutlettype": 0,
                                    "stereo": 0,
                                    "sync": 1
                                }
                            },
                            {
                                "box": {
                                    "fontname": "Arial",
                                    "fontsize": 13.0,
                                    "id": "box-270-290",
                                    "maxclass": "newobj",
                                    "numinlets": 1,
                                    "numoutlets": 1,
                                    "outlettype": [ "" ],
                                    "patching_rect": [ 302.0, 514.0, 394.0, 23.0 ],
                                    "text": "mlmat.buffer_griffinlim gl_help @fftsize 1024 @overlap 4 @iters 50"
                                }
                            },
                            {
                                "box": {
                                    "id": "btn-270-320",
                                    "maxclass": "button",
                                    "numinlets": 1,
                                    "numoutlets": 1,
                                    "outlettype": [ "bang" ],
                                    "parameter_enable": 0,
                                    "patching_rect": [ 302.0, 544.0, 20.0, 20.0 ]
                                }
                            },
                            {
                                "box": {
                                    "fontname": "Arial",
                                    "fontsize": 13.0,
                                    "id": "box-270-350",
                                    "maxclass": "newobj",
                                    "numinlets": 3,
                                    "numoutlets": 1,
                                    "outlettype": [ "signal" ],
                                    "patching_rect": [ 302.0, 574.0, 100.0, 23.0 ],
                                    "text": "wave~ gl_help"
                                }
                            },
                            {
                                "box": {
                                    "depthbuffer": 1,
                                    "doublebuffer": 1,
                                    "fsaa": 0,
                                    "id": "pw-380-350",
                                    "maxclass": "jit.pwindow",
                                    "numinlets": 1,
                                    "numoutlets": 2,
                                    "outlettype": [ "jit_matrix", "" ],
                                    "patching_rect": [ 412.0, 574.0, 200.0, 60.0 ],
                                    "pwinoutlettype": 0,
                                    "stereo": 0,
                                    "sync": 1
                                }
                            },
                            {
                                "box": {
                                    "attr": "fftsize",
                                    "fontname": "Arial",
                                    "fontsize": 13.0,
                                    "id": "obj-attr-0",
                                    "maxclass": "attrui",
                                    "numinlets": 1,
                                    "numoutlets": 1,
                                    "outlettype": [ "" ],
                                    "parameter_enable": 0,
                                    "patching_rect": [ 42.0, 574.0, 200.0, 23.0 ]
                                }
                            },
                            {
                                "box": {
                                    "attr": "overlap",
                                    "fontname": "Arial",
                                    "fontsize": 13.0,
                                    "id": "obj-attr-1",
                                    "maxclass": "attrui",
                                    "numinlets": 1,
                                    "numoutlets": 1,
                                    "outlettype": [ "" ],
                                    "parameter_enable": 0,
                                    "patching_rect": [ 42.0, 596.0, 200.0, 23.0 ]
                                }
                            },
                            {
                                "box": {
                                    "attr": "iters",
                                    "fontname": "Arial",
                                    "fontsize": 13.0,
                                    "id": "obj-attr-2",
                                    "maxclass": "attrui",
                                    "numinlets": 1,
                                    "numoutlets": 1,
                                    "outlettype": [ "" ],
                                    "parameter_enable": 0,
                                    "patching_rect": [ 42.0, 618.0, 200.0, 23.0 ]
                                }
                            },
                            {
                                "box": {
                                    "attr": "momentum",
                                    "fontname": "Arial",
                                    "fontsize": 13.0,
                                    "id": "obj-attr-3",
                                    "maxclass": "attrui",
                                    "numinlets": 1,
                                    "numoutlets": 1,
                                    "outlettype": [ "" ],
                                    "parameter_enable": 0,
                                    "patching_rect": [ 42.0, 640.0, 200.0, 23.0 ]
                                }
                            },
                            {
                                "box": {
                                    "attr": "full_spectrum",
                                    "fontname": "Arial",
                                    "fontsize": 13.0,
                                    "id": "obj-attr-4",
                                    "maxclass": "attrui",
                                    "numinlets": 1,
                                    "numoutlets": 1,
                                    "outlettype": [ "" ],
                                    "parameter_enable": 0,
                                    "patching_rect": [ 42.0, 662.0, 200.0, 23.0 ]
                                }
                            }
                        ],
                        "lines": [],
                        "bgfillcolor_type": "color",
                        "bgfillcolor_color1": [ 0.376471, 0.384314, 0.4, 1.0 ],
                        "bgfillcolor_color2": [ 0.290196, 0.309804, 0.341176, 1.0 ],
                        "bgfillcolor_color": [ 1.0, 1.0, 1.0, 1.0 ]
                    },
                    "patching_rect": [ 121.0, 167.0, 50.0, 23.0 ],
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
                    "jsarguments": [ "mlmat.buffer_griffinlim" ],
                    "maxclass": "jsui",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 10.0, 10.0, 486.9679870605469, 57.599853515625 ]
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
                        "rect": [ 0.0, 26.0, 700.0, 674.0 ],
                        "showontab": 1,
                        "boxes": [],
                        "lines": []
                    },
                    "patching_rect": [ 252.0, 265.0, 27.0, 23.0 ],
                    "text": "p ?",
                    "varname": "q_tab"
                }
            }
        ],
        "lines": [],
        "autosave": 0
    }
}