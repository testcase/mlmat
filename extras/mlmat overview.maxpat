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
        "rect": [ 177.0, 415.0, 928.0, 929.0 ],
        "boxes": [
            {
                "box": {
                    "bgcolor": [ 0.962713, 0.938393, 0.952793, 0.0 ],
                    "button": 1,
                    "fontface": 1,
                    "fontname": "Lato Regular",
                    "fontsize": 16.0,
                    "htabcolor": [ 0.952941, 0.564706, 0.098039, 1.0 ],
                    "id": "obj-16",
                    "margin": 5,
                    "maxclass": "tab",
                    "numinlets": 1,
                    "numoutlets": 3,
                    "outlettype": [ "int", "", "" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 634.0, 336.0, 248.0, 252.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 603.0, 382.0, 192.0, 101.0 ],
                    "rounded": 9.0,
                    "spacing_x": 12.0,
                    "spacing_y": 14.0,
                    "tabcolor": [ 0.664532, 0.706344, 0.714923, 1.0 ],
                    "tabs": [ "mlmat.buffer_stft", "mlmat.buffer_istft", "mlmat.buffer_griffinlim", "mlmat.buffer_mqsynth", "mlmat.spectral_crosssynth", "mlmat.spectral_mask", "mlmat.spectral_op", "mlmat.spectral_peaks", "mlmat.specatral_reassign" ],
                    "textcolor": [ 0.29971, 0.332965, 0.409308, 1.0 ]
                }
            },
            {
                "box": {
                    "hidden": 1,
                    "id": "obj-14",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 707.0, 629.0, 79.0, 22.0 ],
                    "text": "prepend help"
                }
            },
            {
                "box": {
                    "hidden": 1,
                    "id": "obj-15",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 707.0, 656.0, 53.0, 22.0 ],
                    "text": "pcontrol"
                }
            },
            {
                "box": {
                    "fontsize": 24.0,
                    "id": "obj-13",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 183.0, 41.0, 606.0, 33.0 ],
                    "presentation": 1,
                    "presentation_linecount": 4,
                    "presentation_rect": [ 183.0, 42.0, 190.0, 114.0 ],
                    "text": "machine learning using jitter matrices as data structures"
                }
            },
            {
                "box": {
                    "bgcolor": [ 0.962713, 0.938393, 0.952793, 0.0 ],
                    "button": 1,
                    "fontface": 1,
                    "fontname": "Lato Regular",
                    "fontsize": 16.0,
                    "htabcolor": [ 0.952941, 0.564706, 0.098039, 1.0 ],
                    "id": "obj-2",
                    "margin": 5,
                    "maxclass": "tab",
                    "numinlets": 1,
                    "numoutlets": 3,
                    "outlettype": [ "int", "", "" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 320.5, 440.0, 263.0, 159.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 501.0, 391.0, 192.0, 101.0 ],
                    "rounded": 9.0,
                    "spacing_x": 12.0,
                    "spacing_y": 14.0,
                    "tabcolor": [ 0.664532, 0.706344, 0.714923, 1.0 ],
                    "tabs": [ "knn example", "kmeans example", "som example", "sparse autoencoder example" ],
                    "textcolor": [ 0.29971, 0.332965, 0.409308, 1.0 ]
                }
            },
            {
                "box": {
                    "hidden": 1,
                    "id": "obj-3",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 418.0, 640.0, 57.0, 22.0 ],
                    "text": "tosymbol"
                }
            },
            {
                "box": {
                    "hidden": 1,
                    "id": "obj-4",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 418.0, 714.0, 53.0, 22.0 ],
                    "text": "pcontrol"
                }
            },
            {
                "box": {
                    "fontsize": 24.0,
                    "id": "obj-8",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 323.0, 417.0, 183.0, 33.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 505.0, 351.0, 189.0, 33.0 ],
                    "text": "examples:"
                }
            },
            {
                "box": {
                    "hidden": 1,
                    "id": "obj-11",
                    "maxclass": "newobj",
                    "numinlets": 2,
                    "numoutlets": 2,
                    "outlettype": [ "", "" ],
                    "patching_rect": [ 418.0, 664.0, 175.0, 22.0 ],
                    "text": "combine s .maxpat @triggers 0"
                }
            },
            {
                "box": {
                    "hidden": 1,
                    "id": "obj-12",
                    "maxclass": "message",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 418.0, 688.0, 84.0, 22.0 ],
                    "text": "loadunique $1"
                }
            },
            {
                "box": {
                    "bgcolor": [ 0.962713, 0.938393, 0.952793, 0.0 ],
                    "button": 1,
                    "fontface": 1,
                    "fontname": "Lato Regular",
                    "fontsize": 16.0,
                    "htabcolor": [ 0.952941, 0.564706, 0.098039, 1.0 ],
                    "id": "obj-1",
                    "margin": 5,
                    "maxclass": "tab",
                    "numinlets": 1,
                    "numoutlets": 3,
                    "outlettype": [ "int", "", "" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 14.0, 440.0, 225.0, 100.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 37.0, 476.0, 192.0, 101.0 ],
                    "rounded": 9.0,
                    "spacing_x": 12.0,
                    "spacing_y": 14.0,
                    "tabcolor": [ 0.664532, 0.706344, 0.714923, 1.0 ],
                    "tabs": [ "mlmat data modes", "mlmat utilities", "mlmat serialization" ],
                    "textcolor": [ 0.29971, 0.332965, 0.409308, 1.0 ]
                }
            },
            {
                "box": {
                    "hidden": 1,
                    "id": "obj-37",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 117.0, 560.0, 57.0, 22.0 ],
                    "text": "tosymbol"
                }
            },
            {
                "box": {
                    "hidden": 1,
                    "id": "obj-34",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 117.0, 634.0, 53.0, 22.0 ],
                    "text": "pcontrol"
                }
            },
            {
                "box": {
                    "fontsize": 24.0,
                    "id": "obj-33",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 18.0, 400.0, 183.0, 33.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 12.0, 425.0, 189.0, 33.0 ],
                    "text": "getting started:"
                }
            },
            {
                "box": {
                    "hidden": 1,
                    "id": "obj-5",
                    "maxclass": "newobj",
                    "numinlets": 2,
                    "numoutlets": 2,
                    "outlettype": [ "", "" ],
                    "patching_rect": [ 117.0, 584.0, 175.0, 22.0 ],
                    "text": "combine s .maxpat @triggers 0"
                }
            },
            {
                "box": {
                    "hidden": 1,
                    "id": "obj-31",
                    "maxclass": "message",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 117.0, 608.0, 84.0, 22.0 ],
                    "text": "loadunique $1"
                }
            },
            {
                "box": {
                    "hidden": 1,
                    "id": "obj-30",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 392.0, 390.0, 79.0, 22.0 ],
                    "text": "prepend help"
                }
            },
            {
                "box": {
                    "hidden": 1,
                    "id": "obj-29",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 392.0, 417.0, 53.0, 22.0 ],
                    "text": "pcontrol"
                }
            },
            {
                "box": {
                    "bgcolor": [ 0.962713, 0.938393, 0.952793, 0.0 ],
                    "button": 1,
                    "fontface": 1,
                    "fontname": "Lato Regular",
                    "fontsize": 16.0,
                    "htabcolor": [ 0.952941, 0.564706, 0.098039, 1.0 ],
                    "id": "obj-7",
                    "margin": 5,
                    "maxclass": "tab",
                    "numinlets": 1,
                    "numoutlets": 3,
                    "outlettype": [ "int", "", "" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 13.0, 124.0, 878.0, 207.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 437.5, 144.0, 192.0, 101.0 ],
                    "rounded": 9.0,
                    "spacing_x": 12.0,
                    "spacing_y": 14.0,
                    "tabcolor": [ 0.664532, 0.706344, 0.714923, 1.0 ],
                    "tabs": [ "mlmat.concat", "mlmat.convert", "mlmat.gmm", "mlmat.hmm", "mlmat.hoeffding_tree", "mlmat.id3_tree", "mlmat.kfn", "mlmat.kmeans", "mlmat.knn", "mlmat.linear_regression", "mlmat.linear_svm", "mlmat.load", "mlmat.lookup", "mlmat.mean_shift", "mlmat.mlp_classifier", "mlmat.mlp_regressor", "mlmat.nmf", "mlmat.pca", "mlmat.scaling", "mlmat.som", "mlmat.sparse_autoencoder", "mlmat.split" ],
                    "textcolor": [ 0.29971, 0.332965, 0.409308, 1.0 ]
                }
            },
            {
                "box": {
                    "fontface": 1,
                    "fontsize": 48.0,
                    "id": "obj-6",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 13.0, 14.0, 163.0, 60.0 ],
                    "presentation": 1,
                    "presentation_linecount": 2,
                    "presentation_rect": [ 17.5, 13.0, 102.0, 114.0 ],
                    "text": "mlmat"
                }
            },
            {
                "box": {
                    "fontsize": 24.0,
                    "id": "obj-9",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 9.0, 85.0, 183.0, 33.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 1037.0, 410.0, 189.0, 33.0 ],
                    "text": "current objects :"
                }
            },
            {
                "box": {
                    "fontsize": 24.0,
                    "id": "obj-10",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 1007.0, 686.0, 285.0, 33.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 442.0, 76.0, 152.0, 33.0 ]
                }
            }
        ],
        "lines": [
            {
                "patchline": {
                    "destination": [ "obj-37", 0 ],
                    "hidden": 1,
                    "source": [ "obj-1", 1 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-12", 0 ],
                    "hidden": 1,
                    "source": [ "obj-11", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-4", 0 ],
                    "hidden": 1,
                    "source": [ "obj-12", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-15", 0 ],
                    "hidden": 1,
                    "source": [ "obj-14", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-14", 0 ],
                    "hidden": 1,
                    "source": [ "obj-16", 1 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-3", 0 ],
                    "hidden": 1,
                    "source": [ "obj-2", 1 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-11", 0 ],
                    "hidden": 1,
                    "source": [ "obj-3", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-29", 0 ],
                    "hidden": 1,
                    "source": [ "obj-30", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-34", 0 ],
                    "hidden": 1,
                    "source": [ "obj-31", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-5", 0 ],
                    "hidden": 1,
                    "source": [ "obj-37", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-31", 0 ],
                    "hidden": 1,
                    "source": [ "obj-5", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-30", 0 ],
                    "hidden": 1,
                    "source": [ "obj-7", 1 ]
                }
            }
        ],
        "autosave": 0
    }
}