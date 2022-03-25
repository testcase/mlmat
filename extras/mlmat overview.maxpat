{
	"patcher" : 	{
		"fileversion" : 1,
		"appversion" : 		{
			"major" : 8,
			"minor" : 2,
			"revision" : 2,
			"architecture" : "x64",
			"modernui" : 1
		}
,
		"classnamespace" : "box",
		"rect" : [ 34.0, 87.0, 1212.0, 713.0 ],
		"bglocked" : 0,
		"openinpresentation" : 0,
		"default_fontsize" : 12.0,
		"default_fontface" : 0,
		"default_fontname" : "Arial",
		"gridonopen" : 1,
		"gridsize" : [ 15.0, 15.0 ],
		"gridsnaponopen" : 1,
		"objectsnaponopen" : 1,
		"statusbarvisible" : 2,
		"toolbarvisible" : 1,
		"lefttoolbarpinned" : 0,
		"toptoolbarpinned" : 0,
		"righttoolbarpinned" : 0,
		"bottomtoolbarpinned" : 0,
		"toolbars_unpinned_last_save" : 8,
		"tallnewobj" : 0,
		"boxanimatetime" : 200,
		"enablehscroll" : 1,
		"enablevscroll" : 1,
		"devicewidth" : 0.0,
		"description" : "",
		"digest" : "",
		"tags" : "",
		"style" : "",
		"subpatcher_template" : "",
		"assistshowspatchername" : 0,
		"boxes" : [ 			{
				"box" : 				{
					"bgcolor" : [ 0.682352941176471, 0.796078431372549, 0.513725490196078, 1.0 ],
					"id" : "obj-6",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 370.0, 525.0, 150.0, 20.0 ],
					"text" : "mlmat.lstm"
				}

			}
, 			{
				"box" : 				{
					"bgcolor" : [ 0.682352941176471, 0.796078431372549, 0.513725490196078, 1.0 ],
					"id" : "obj-5",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 370.0, 497.0, 150.0, 20.0 ],
					"text" : "mlmat.rnn"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-31",
					"linecount" : 2,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 29.0, 883.0, 150.0, 33.0 ],
					"text" : "GAN - DCGAN, WGAN, etc"
				}

			}
, 			{
				"box" : 				{
					"bgcolor" : [ 0.682352941176471, 0.796078431372549, 0.513725490196078, 1.0 ],
					"id" : "obj-28",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 29.0, 835.0, 150.0, 20.0 ],
					"text" : "mlmat.rbm"
				}

			}
, 			{
				"box" : 				{
					"bgcolor" : [ 0.682352941176471, 0.796078431372549, 0.513725490196078, 1.0 ],
					"id" : "obj-27",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 29.0, 799.0, 150.0, 20.0 ],
					"text" : "mlmat.rnn"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-10",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 3,
					"outlettype" : [ "jit_matrix", "jit_matrix", "" ],
					"patching_rect" : [ 29.0, 733.0, 173.0, 22.0 ],
					"text" : "mlmat.variational_autoencoder"
				}

			}
, 			{
				"box" : 				{
					"bgcolor" : [ 0.682352941176471, 0.796078431372549, 0.513725490196078, 1.0 ],
					"id" : "obj-7",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 188.0, 355.0, 150.0, 20.0 ],
					"text" : "mlmat.logistic_regression"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "jit_matrix", "jit_matrix", "" ],
					"patching_rect" : [ 130.0, 122.0, 103.0, 22.0 ],
					"text" : "mlmat.mean_shift"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-3",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 5,
					"outlettype" : [ "jit_matrix", "jit_matrix", "jit_matrix", "jit_matrix", "" ],
					"patching_rect" : [ 130.0, 56.0, 66.0, 22.0 ],
					"text" : "mlmat.split"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-2",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "jit_matrix" ],
					"patching_rect" : [ 307.0, 56.0, 80.0, 22.0 ],
					"text" : "mlmat.concat",
					"varname" : "mlmat.concat"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-1",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "jit_matrix", "" ],
					"patching_rect" : [ 29.0, 56.0, 84.0, 22.0 ],
					"text" : "mlmat.convert"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-26",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 3,
					"outlettype" : [ "jit_matrix", "jit_matrix", "" ],
					"patching_rect" : [ 29.0, 673.0, 154.0, 22.0 ],
					"text" : "mlmat.sparse_autoencoder"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-25",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "jit_matrix", "" ],
					"patching_rect" : [ 29.0, 274.0, 67.0, 22.0 ],
					"text" : "mlmat.som"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-24",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 3,
					"outlettype" : [ "jit_matrix", "jit_matrix", "" ],
					"patching_rect" : [ 213.0, 56.0, 82.0, 22.0 ],
					"text" : "mlmat.scaling"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-23",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 4,
					"outlettype" : [ "jit_matrix", "jit_matrix", "jit_matrix", "" ],
					"patching_rect" : [ 130.0, 274.0, 64.0, 22.0 ],
					"text" : "mlmat.pca"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-22",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 2,
					"outlettype" : [ "jit_matrix", "" ],
					"patching_rect" : [ 29.0, 497.0, 121.0, 22.0 ],
					"text" : "mlmat.mlp_regressor"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-21",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 3,
					"outlettype" : [ "jit_matrix", "jit_matrix", "" ],
					"patching_rect" : [ 178.0, 497.0, 117.0, 22.0 ],
					"text" : "mlmat.mlp_classifier"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-20",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 2,
					"outlettype" : [ "jit_matrix", "" ],
					"patching_rect" : [ 418.0, 56.0, 80.0, 22.0 ],
					"text" : "mlmat.lookup"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-19",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "jit_matrix", "" ],
					"patching_rect" : [ 524.0, 56.0, 67.0, 22.0 ],
					"text" : "mlmat.load"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-18",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 3,
					"outlettype" : [ "jit_matrix", "jit_matrix", "" ],
					"patching_rect" : [ 29.0, 418.0, 103.0, 22.0 ],
					"text" : "mlmat.linear_svm"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-17",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 2,
					"outlettype" : [ "jit_matrix", "" ],
					"patching_rect" : [ 29.0, 351.0, 137.0, 22.0 ],
					"text" : "mlmat.linear_regression"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-16",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 3,
					"outlettype" : [ "jit_matrix", "jit_matrix", "" ],
					"patching_rect" : [ 29.0, 166.0, 64.0, 22.0 ],
					"text" : "mlmat.knn"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-15",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 3,
					"outlettype" : [ "jit_matrix", "jit_matrix", "" ],
					"patching_rect" : [ 29.0, 122.0, 87.0, 22.0 ],
					"text" : "mlmat.kmeans"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-14",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 3,
					"outlettype" : [ "jit_matrix", "jit_matrix", "" ],
					"patching_rect" : [ 130.0, 166.0, 61.0, 22.0 ],
					"text" : "mlmat.kfn"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-13",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 3,
					"outlettype" : [ "jit_matrix", "jit_matrix", "" ],
					"patching_rect" : [ 29.0, 221.0, 88.0, 22.0 ],
					"text" : "mlmat.id3_tree"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-12",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 3,
					"outlettype" : [ "jit_matrix", "jit_matrix", "" ],
					"patching_rect" : [ 130.0, 221.0, 121.0, 22.0 ],
					"text" : "mlmat.hoeffding_tree"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-11",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 3,
					"outlettype" : [ "jit_matrix", "jit_matrix", "" ],
					"patching_rect" : [ 29.0, 617.0, 71.0, 22.0 ],
					"text" : "mlmat.hmm"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-8",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 5,
					"outlettype" : [ "jit_matrix", "jit_matrix", "jit_matrix", "jit_matrix", "" ],
					"patching_rect" : [ 29.0, 560.0, 71.0, 22.0 ],
					"text" : "mlmat.gmm"
				}

			}
 ],
		"lines" : [  ],
		"dependency_cache" : [ 			{
				"name" : "mlmat.concat.maxpat",
				"bootpath" : "~/Dropbox/Documents/Max 8/Packages/mlmat/patchers",
				"patcherrelativepath" : "../patchers",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "mlmat.convert.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "mlmat.gmm.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "mlmat.hmm.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "mlmat.hoeffding_tree.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "mlmat.id3_tree.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "mlmat.kfn.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "mlmat.kmeans.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "mlmat.knn.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "mlmat.linear_regression.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "mlmat.linear_svm.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "mlmat.load.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "mlmat.lookup.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "mlmat.mean_shift.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "mlmat.mlp_classifier.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "mlmat.mlp_regressor.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "mlmat.pca.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "mlmat.scaling.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "mlmat.som.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "mlmat.sparse_autoencoder.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "mlmat.split.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "mlmat.variational_autoencoder.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "mlmat_concat.js",
				"bootpath" : "~/Dropbox/Documents/Max 8/Packages/mlmat/jsextensions",
				"patcherrelativepath" : "../jsextensions",
				"type" : "TEXT",
				"implicit" : 1
			}
 ],
		"autosave" : 0
	}

}
