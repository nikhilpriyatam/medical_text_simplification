=======================================================
Leveraging Social Media for Medical Text Simplification
=======================================================

This repository contains the code required for training and evaluating a Denoising autoencoder based model which is used to simplify medical text. It also contains the datasets used for evaluating the models. For more details please refer to the SIGIR 2020 paper "Leveraging social media for medical text simplification". If you find this code or dataset useful in your research, please consider citing:

.. code-block:: none
   
   @inproceedings{pattisapu2020leveraging,
      title={Leveraging Social Media for Medical Text Simplification},
      author={Pattisapu, Nikhil and Prabhu, Nishant and Bhati, Smriti and Varma, Vasudeva},
      booktitle={Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
      pages={1141--1144},
      year={2020}}

In case you need access to the training data, kindly drop us an email!


DAE for Medical Text Simplification 
===================================

.. image:: https://github.com/nikhilpriyatam/medical_text_simplification/blob/master/images/MTS_architecture.jpg?raw=true
   :height: 100px
   :width: 200 px
   :scale: 50 %

* The above Figure shows our overall system architecture. 
* The simple medical sentences needed for training our Denoising Autoencoder (DAE) model are obtained from medical social media blogs. 
* We discover the medical entity mentions from these sentences using MetaMap and replace them with their corresponding UMLS concept names. 
* Finally, we tokenize the noisy sentences using an existing sub-word tokenizer and the resulting tokens are used as inputs (or source tokens) to the encoder.
* The original unperturbed medical blog sentences are also tokenized using the same sub-word tokenizer which results in the target tokens.


Training DAE
============

* In order to train the model, follow the below mentioned steps. We use `fairseq` for training and inference.

* Create three files for train, development and test which contains medical sentences.For now, we assume that the extension for these files is .sen which denotes simple english.

* Generate aligned train, development and test files between noisy English (.nen) and simple English (.sen) using :code:`preprocess.py` which generates the source-target pairs. In the source sentences, medical concept mentions are replaced by their UMLS concept names. 
   Usage: :code:`python preprocess.py <inp_file> <op_src_file> <op_tgt_file>`

* Tokenize the source and target files using scibert tokens.
   Usage: :code:`python tokenize.py <src_file_to_be_tokenized> <vocab_file_path> <op_file_path>`

* Preprocess the tokenized data using fairseq-preprocess

   .. code-block:: bash
      fairseq-preprocess \
      --task translation \
      -s nen -t sen \
      --trainpref <prefix_to_train_files>
      --validpref <prefix_to_dev_files>
      --testpref <comma_separated_prefixes_to_test_files>
      --destdir <path_to_bin_dir>
      --workers 16
      --srcdict <path_to_vocab_fairseq_style>
      --tgtdict <path_to_vocab_fairseq_style>
  
* Train the autoencoder model using fairseq-transformer

   .. code-block:: bash
      CUDA_VISIBLE_DEVICES=2 fairseq-train <path_to_bin_dir> \
      --arch transformer \
      --optimizer adam \
      --adam-betas '(0.9, 0.98)' \
      --clip-norm 0.0 \
      --lr 5e-4 \
      --lr-scheduler inverse_sqrt \
      --warmup-updates 4000 \
      --weight-decay 0.0001 \
      --criterion label_smoothed_cross_entropy \
      --label-smoothing 0.1 \
      --max-tokens 2048 \
      --max-epoch 10 \
      --num-workers 16 \
      --save-dir <path_where_checkpoints_are_to_be_stored>


==============
Evaluating DAE
==============

* Get the predictions using `fairseq-interactive`

   .. code-block:: bash
      CUDA_VISIBLE_DEVICES=2 fairseq-interactive \
      --beam 5 \
      -s nen -t sen \
      --path <path_to_trained_model> \
      --input <inp_file_path> \
      --max-tokens 4096 \
      --num-workers 32 \
      <path_to_bin_dir> > <path_to_prediction_file>

* Postprocess the output

   :code:`python postprocess.py <path_to_ip_pred_file> <path_to_processed_pred_file>`
