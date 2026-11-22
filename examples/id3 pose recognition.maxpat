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
        "rect": [ 34.0, 139.0, 985.0, 770.0 ],
        "boxes": [
            {
                "box": {
                    "fontface": 1,
                    "fontsize": 48.0,
                    "id": "obj-14",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 44.0, 14.0, 790.0, 60.0 ],
                    "presentation": 1,
                    "presentation_linecount": 6,
                    "presentation_rect": [ 17.5, 13.0, 137.0, 328.0 ],
                    "text": "id3 for pose recognition"
                }
            },
            {
                "box": {
                    "id": "obj-52",
                    "maxclass": "message",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 767.0, 434.0, 50.0, 22.0 ],
                    "text": "3"
                }
            },
            {
                "box": {
                    "id": "obj-45",
                    "linecount": 2,
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 535.0, 120.0, 150.0, 33.0 ],
                    "text": "get dot products of digit segments"
                }
            },
            {
                "box": {
                    "id": "obj-41",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
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
                        "rect": [ 0.0, 0.0, 1000.0, 780.0 ],
                        "boxes": [
                            {
                                "box": {
                                    "code": "\r\ninlets = 1;\r\noutlets = 1;\r\n\r\nfunction vsub(v1, v2) {\r\n    //no checking assume both 3d vectors\r\n    return v1.map((val, index) => val - v2[index]);\r\n}\r\n\r\nfunction vlength(v1) {\r\n       //no checking \r\n    const sqr = v1.map((val) => val * val);\r\n    const sum = sqr.reduce((accum, current) => accum + current, 0);  \r\n    return Math.sqrt(sum);\r\n}\r\n\r\nfunction vnormalize(v1) {\r\n    //no checking \r\n    const len = vlength(v1);\r\n    return v1.map((val) => val / len);\r\n}\r\n\r\nfunction vdot(v1, v2) {\r\n    //no checking \r\n    \r\n    const mul = v1.map((val, index) => val * v2[index]);\r\n    return mul.reduce((accum, current) => accum + current, 0);  \r\n}\r\n\r\nfunction vdistance(v1, v2) {\r\n    //no checking \r\n    const sub = vsub(v2, v1);\r\n    return vlength(sub);\r\n}\r\n\r\n\r\n// this is called whenever a dictionary is received. i am going to assume it is always\r\n// the dictionary for both hands\r\nfunction msg_dictionary(dict) {\r\n\t\t\r\n\tconst output_arr = new Array( 10 );\r\n\t\r\n    if(dict.hand_1) {\r\n\t\tconst thumb = [dict.hand_1.thumb_cmc.xyz, dict.hand_1.thumb_mcp.xyz, dict.hand_1.thumb_ip.xyz, dict.hand_1.thumb_tip.xyz];\r\n\t\tconst index = [dict.hand_1.index_mcp.xyz, dict.hand_1.index_pip.xyz, dict.hand_1.index_dip.xyz, dict.hand_1.index_tip.xyz];\r\n\t\tconst middle = [dict.hand_1.middle_mcp.xyz, dict.hand_1.middle_pip.xyz, dict.hand_1.middle_dip.xyz, dict.hand_1.middle_tip.xyz];\r\n\t\t\r\n\t\t\r\n\t\tconst ring = [dict.hand_1.ring_mcp.xyz, dict.hand_1.ring_pip.xyz, dict.hand_1.ring_dip.xyz, dict.hand_1.ring_tip.xyz];\r\n\t\tconst little = [dict.hand_1.little_mcp.xyz, dict.hand_1.little_pip.xyz, dict.hand_1.little_dip.xyz, dict.hand_1.little_tip.xyz];\r\n\t\t\r\n\t\tconst hand = [thumb, index, middle, ring, little];\r\n\t\t\r\n\t\t\r\n\t\tfor(let i=0;i<5;i++) {\r\n\t\t\tconst digit = hand[i];\r\n\t\t\t\r\n\t\t\tfor(let j=0;j<2;j++) {\r\n\t\t\t\tconst v1 = vnormalize(vsub(digit[j+1], digit[j]));\r\n\t\t\t\tconst v2 = vnormalize(vsub(digit[j+2], digit[j+1]));\r\n\t\t\t\toutput_arr[(i*2)+j] = vdot(v1,v2);\r\n\t\t\t}\r\n\t\t}\r\n        outlet(0,output_arr);\r\n    }\r\n}",
                                    "filename": "none",
                                    "fontface": 0,
                                    "fontname": "<Monospaced>",
                                    "fontsize": 12.0,
                                    "id": "obj-6",
                                    "maxclass": "v8.codebox",
                                    "numinlets": 1,
                                    "numoutlets": 1,
                                    "outlettype": [ "" ],
                                    "patching_rect": [ 50.0, 100.0, 1177.0, 861.0 ],
                                    "saved_object_attributes": {
                                        "parameter_enable": 0
                                    }
                                }
                            },
                            {
                                "box": {
                                    "comment": "",
                                    "id": "obj-39",
                                    "index": 1,
                                    "maxclass": "inlet",
                                    "numinlets": 0,
                                    "numoutlets": 1,
                                    "outlettype": [ "" ],
                                    "patching_rect": [ 49.999941000000035, 39.99999982304764, 30.0, 30.0 ]
                                }
                            },
                            {
                                "box": {
                                    "comment": "",
                                    "id": "obj-40",
                                    "index": 1,
                                    "maxclass": "outlet",
                                    "numinlets": 1,
                                    "numoutlets": 0,
                                    "patching_rect": [ 49.999941000000035, 1021.0000098230477, 30.0, 30.0 ]
                                }
                            }
                        ],
                        "lines": [
                            {
                                "patchline": {
                                    "destination": [ "obj-6", 0 ],
                                    "source": [ "obj-39", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-40", 0 ],
                                    "source": [ "obj-6", 0 ]
                                }
                            }
                        ]
                    },
                    "patching_rect": [ 413.0, 114.0, 105.0, 22.0 ],
                    "text": "p extract_features"
                }
            },
            {
                "box": {
                    "auto_handle": 0,
                    "depthbuffer": 1,
                    "doublebuffer": 1,
                    "enable": 1,
                    "erase_color": [ 0.0, 0.0, 0.0, 1.0 ],
                    "fps": 30.0,
                    "fsaa": 0,
                    "gizmos": 0,
                    "id": "obj-19",
                    "maxclass": "jit.pworld",
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "jit_matrix", "" ],
                    "patching_rect": [ 35.0, 503.6667009592056, 154.0, 118.6666533946991 ],
                    "pwinoutlettype": 0,
                    "stereo": 0,
                    "sync": 1
                }
            },
            {
                "box": {
                    "id": "obj-13",
                    "maxclass": "dict.view",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 191.0, 500.66667288541794, 163.0, 150.0 ]
                }
            },
            {
                "box": {
                    "attr": "async",
                    "id": "obj-24",
                    "maxclass": "attrui",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 44.0, 256.0, 89.0, 22.0 ],
                    "text_width": 60.0
                }
            },
            {
                "box": {
                    "attr": "engine",
                    "id": "obj-12",
                    "maxclass": "attrui",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 44.0, 232.0, 150.0, 22.0 ],
                    "text_width": 79.0
                }
            },
            {
                "box": {
                    "attr": "smooth",
                    "id": "obj-17",
                    "maxclass": "attrui",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 44.0, 280.0, 129.0, 22.0 ],
                    "text_width": 68.0
                }
            },
            {
                "box": {
                    "attr": "debugdraw",
                    "id": "obj-18",
                    "maxclass": "attrui",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 44.0, 303.0, 129.0, 22.0 ]
                }
            },
            {
                "box": {
                    "id": "obj-20",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 44.0, 144.0, 75.0, 22.0 ],
                    "text": "loadmess 1"
                }
            },
            {
                "box": {
                    "bgmode": 0,
                    "border": 0,
                    "clickthrough": 0,
                    "enablehscroll": 0,
                    "enablevscroll": 0,
                    "id": "obj-22",
                    "lockeddragscroll": 0,
                    "lockedsize": 0,
                    "maxclass": "bpatcher",
                    "name": "jit-vis-grab-ui.maxpat",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "offset": [ 0.0, 0.0 ],
                    "outlettype": [ "jit_matrix" ],
                    "patching_rect": [ 44.0, 169.0, 250.0, 55.0 ],
                    "varname": "jit-vis-grab-ui",
                    "viewvisibility": 1
                }
            },
            {
                "box": {
                    "id": "obj-29",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 3,
                    "outlettype": [ "jit_matrix", "", "" ],
                    "patching_rect": [ 44.0, 428.0, 303.0, 22.0 ],
                    "text": "jit.vis.hands @debugdraw 1 @mirrored 1 @engine web"
                }
            },
            {
                "box": {
                    "cols": 1,
                    "fontface": 0,
                    "fontname": "Arial",
                    "fontsize": 12.0,
                    "id": "obj-3",
                    "maxclass": "jit.cellblock",
                    "numinlets": 2,
                    "numoutlets": 4,
                    "outlettype": [ "list", "", "", "" ],
                    "patching_rect": [ 451.0, 598.0, 208.0, 78.0 ],
                    "rows": 1
                }
            },
            {
                "box": {
                    "id": "obj-4",
                    "maxclass": "gswitch2",
                    "numinlets": 2,
                    "numoutlets": 2,
                    "outlettype": [ "", "" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 536.0, 478.0, 39.0, 32.0 ]
                }
            },
            {
                "box": {
                    "id": "obj-21",
                    "maxclass": "newobj",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 580.0, 338.0, 37.0, 22.0 ],
                    "text": "join 2"
                }
            },
            {
                "box": {
                    "clickedimage": 1,
                    "id": "obj-15",
                    "maxclass": "pictctrl",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "int" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 507.0, 364.0, 44.0, 44.0 ]
                }
            },
            {
                "box": {
                    "id": "obj-7",
                    "maxclass": "incdec",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "float" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 685.0, 338.0, 25.0, 31.0 ]
                }
            },
            {
                "box": {
                    "fontsize": 20.0,
                    "id": "obj-5",
                    "maxclass": "number",
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "", "bang" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 619.0, 338.0, 64.0, 31.0 ]
                }
            },
            {
                "box": {
                    "contdata": 1,
                    "ghostbar": 71,
                    "id": "obj-8",
                    "maxclass": "multislider",
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "", "" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 413.0, 148.0, 262.0, 164.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 1402.0, 634.0, 233.0, 142.0 ],
                    "setminmax": [ -3.145900011062622, 3.145900011062622 ],
                    "size": 10,
                    "spacing": 5
                }
            },
            {
                "box": {
                    "button": 1,
                    "fontsize": 10.0,
                    "id": "obj-2",
                    "maxclass": "tab",
                    "numinlets": 1,
                    "numoutlets": 3,
                    "outlettype": [ "int", "", "" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 463.0, 516.0, 88.0, 39.0 ],
                    "rounded": 0.0,
                    "spacing_x": 1.0,
                    "spacing_y": 3.0,
                    "tabs": [ "train", "clear", "read", "write" ]
                }
            },
            {
                "box": {
                    "id": "obj-84",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "jit_matrix", "" ],
                    "patching_rect": [ 555.0, 446.0, 119.0, 22.0 ],
                    "text": "jit.matrix 10 float64 1"
                }
            },
            {
                "box": {
                    "id": "obj-85",
                    "maxclass": "newobj",
                    "numinlets": 2,
                    "numoutlets": 2,
                    "outlettype": [ "", "" ],
                    "patching_rect": [ 555.0, 417.0, 61.0, 22.0 ],
                    "text": "zl slice 10"
                }
            },
            {
                "box": {
                    "id": "obj-80",
                    "maxclass": "gswitch2",
                    "numinlets": 2,
                    "numoutlets": 2,
                    "outlettype": [ "", "" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 555.0, 373.0, 39.0, 32.0 ]
                }
            },
            {
                "box": {
                    "id": "obj-79",
                    "maxclass": "button",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "bang" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 910.0, 523.0, 24.0, 24.0 ]
                }
            },
            {
                "box": {
                    "id": "obj-61",
                    "linecount": 3,
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "jit_matrix", "" ],
                    "patching_rect": [ 809.0, 622.0, 80.0, 49.0 ],
                    "text": "jit.matrix labels @thru 0"
                }
            },
            {
                "box": {
                    "id": "obj-60",
                    "linecount": 3,
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "jit_matrix", "" ],
                    "patching_rect": [ 690.0, 622.0, 89.0, 49.0 ],
                    "text": "jit.matrix coords @thru 0"
                }
            },
            {
                "box": {
                    "id": "obj-59",
                    "maxclass": "message",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 648.0, 487.0, 35.0, 22.0 ],
                    "text": "clear"
                }
            },
            {
                "box": {
                    "fontface": 0,
                    "fontsize": 12.0,
                    "id": "obj-55",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "jit_matrix" ],
                    "patching_rect": [ 809.0, 521.0, 80.0, 22.0 ],
                    "style": "default",
                    "text": "mlmat.concat",
                    "varname": "mlmat.concat[3]"
                }
            },
            {
                "box": {
                    "fontface": 0,
                    "fontsize": 12.0,
                    "id": "obj-53",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "jit_matrix" ],
                    "patching_rect": [ 690.0, 538.0, 80.0, 22.0 ],
                    "style": "default",
                    "text": "mlmat.concat",
                    "varname": "mlmat.concat[2]"
                }
            },
            {
                "box": {
                    "id": "obj-47",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "jit_matrix", "" ],
                    "patching_rect": [ 809.0, 487.0, 99.0, 22.0 ],
                    "text": "jit.matrix 1 long 1"
                }
            },
            {
                "box": {
                    "id": "obj-46",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "jit_matrix", "" ],
                    "patching_rect": [ 690.0, 487.0, 119.0, 22.0 ],
                    "text": "jit.matrix 10 float64 1"
                }
            },
            {
                "box": {
                    "id": "obj-44",
                    "maxclass": "newobj",
                    "numinlets": 2,
                    "numoutlets": 2,
                    "outlettype": [ "", "" ],
                    "patching_rect": [ 690.0, 409.0, 61.0, 22.0 ],
                    "text": "zl slice 10"
                }
            },
            {
                "box": {
                    "id": "obj-43",
                    "maxclass": "message",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 696.0, 598.0, 129.66668619209275, 22.0 ],
                    "text": "done"
                }
            },
            {
                "box": {
                    "fontface": 0,
                    "fontsize": 12.0,
                    "id": "obj-1",
                    "maxclass": "newobj",
                    "numinlets": 3,
                    "numoutlets": 3,
                    "outlettype": [ "jit_matrix", "jit_matrix", "" ],
                    "patching_rect": [ 555.0, 563.0, 143.0, 22.0 ],
                    "style": "default",
                    "text": "mlmat.id3_tree @mode 0"
                }
            }
        ],
        "lines": [
            {
                "patchline": {
                    "destination": [ "obj-3", 0 ],
                    "source": [ "obj-1", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-43", 1 ],
                    "source": [ "obj-1", 2 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-29", 0 ],
                    "source": [ "obj-12", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-80", 0 ],
                    "source": [ "obj-15", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-29", 0 ],
                    "source": [ "obj-17", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-29", 0 ],
                    "source": [ "obj-18", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-1", 0 ],
                    "source": [ "obj-2", 1 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-22", 0 ],
                    "source": [ "obj-20", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-80", 1 ],
                    "source": [ "obj-21", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-19", 0 ],
                    "midpoints": [ 53.5, 363.8333504796028, 44.5, 363.8333504796028 ],
                    "order": 1,
                    "source": [ "obj-22", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-29", 0 ],
                    "order": 0,
                    "source": [ "obj-22", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-29", 0 ],
                    "source": [ "obj-24", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-13", 0 ],
                    "midpoints": [ 195.5, 475.33333644270897, 200.5, 475.33333644270897 ],
                    "order": 1,
                    "source": [ "obj-29", 1 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-41", 0 ],
                    "midpoints": [ 195.5, 460.0, 403.58334249999996, 460.0, 403.58334249999996, 106.0, 422.5, 106.0 ],
                    "order": 0,
                    "source": [ "obj-29", 1 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-1", 0 ],
                    "source": [ "obj-4", 1 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-21", 0 ],
                    "order": 0,
                    "source": [ "obj-41", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-8", 0 ],
                    "order": 1,
                    "source": [ "obj-41", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-46", 0 ],
                    "source": [ "obj-44", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-47", 0 ],
                    "order": 0,
                    "source": [ "obj-44", 1 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-52", 1 ],
                    "order": 1,
                    "source": [ "obj-44", 1 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-53", 0 ],
                    "source": [ "obj-46", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-55", 0 ],
                    "source": [ "obj-47", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-21", 1 ],
                    "midpoints": [ 628.5, 328.8971143960953, 607.5, 328.8971143960953 ],
                    "order": 1,
                    "source": [ "obj-5", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-7", 0 ],
                    "order": 0,
                    "source": [ "obj-5", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-60", 0 ],
                    "midpoints": [ 699.5, 564.6666733026505, 699.5, 564.6666733026505 ],
                    "source": [ "obj-53", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-61", 0 ],
                    "midpoints": [ 818.5, 564.6666733026505, 818.5, 564.6666733026505 ],
                    "source": [ "obj-55", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-53", 0 ],
                    "order": 1,
                    "source": [ "obj-59", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-55", 0 ],
                    "order": 0,
                    "source": [ "obj-59", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-1", 1 ],
                    "midpoints": [ 699.5, 681.0, 648.75, 681.0, 648.75, 551.3333185315132, 626.5, 551.3333185315132 ],
                    "source": [ "obj-60", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-1", 2 ],
                    "midpoints": [ 818.5, 681.0, 725.5833430960464, 681.0, 725.5833430960464, 551.3333185315132, 688.5, 551.3333185315132 ],
                    "source": [ "obj-61", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-5", 0 ],
                    "source": [ "obj-7", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-60", 0 ],
                    "midpoints": [ 919.5, 567.0, 699.5, 567.0 ],
                    "order": 1,
                    "source": [ "obj-79", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-61", 0 ],
                    "order": 0,
                    "source": [ "obj-79", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-44", 0 ],
                    "source": [ "obj-80", 1 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-85", 0 ],
                    "source": [ "obj-80", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-4", 1 ],
                    "source": [ "obj-84", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-84", 0 ],
                    "source": [ "obj-85", 0 ]
                }
            }
        ],
        "parameters": {
            "obj-22::obj-1": [ "enabletog", "enabletog", 0 ],
            "obj-22::obj-10": [ "vdevnum", "vdevnum", 0 ],
            "obj-22::obj-13": [ "formatnum", "formatnum", 0 ],
            "obj-22::obj-18": [ "mirrortog", "mirrortog", 0 ],
            "obj-22::obj-37": [ "textog", "textog", 0 ],
            "parameterbanks": {
                "0": {
                    "index": 0,
                    "name": "",
                    "parameters": [ "-", "-", "-", "-", "-", "-", "-", "-" ],
                    "buttons": [ "-", "-", "-", "-", "-", "-", "-", "-" ]
                }
            },
            "inherited_shortname": 1
        },
        "autosave": 0
    }
}