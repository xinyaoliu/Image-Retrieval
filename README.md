# Image Question Answering (VQA)

Keyword: VQA CNN RNN LSTM GRU TORCH 

![AI](https://cloud.githubusercontent.com/assets/10870023/15724955/5c194da2-27fe-11e6-8d85-2607a3acce28.jpg)

This work explores image question answering (VQA) with different neural network models and reproduces the experimental conclusions presented by previous researchers in this area. 

The code is developed based on Abhishek Das's code https://github.com/abhshkdz/neural-vqa, and the paper
[Exploring Models and Data for Image Question Answering][2] 
by Mengye Ren, Ryan Kiros & Richard Zemel.

Visual question and answering (VQA) is still a new topic has not been fully studied. Even through there are many strong evidences suggest that long short term memory (LSTM) is much better than regular RNN, RNN may still achieve a good result since the question is normally very short. Gated Recurrent Units (GRUs), a simpler variant of LSTMs, were first proposed in 2014. It mainly serves the same purpose with LSTMs. Therefore, an empirical evaluation of VQA with different neural network models mentioned above is studied in this paper. 

![Model architecture](https://cloud.githubusercontent.com/assets/10870023/15724892/0c758608-27fe-11e6-9e77-cb9c0ce6a265.png)


The LSTM structure labeled above can be replaced by RNN or GRU simply by replacing '-model rnn' '-model gru'. 

Here is what a typical RNN looks like:

![](https://cloud.githubusercontent.com/assets/10870023/15727063/2ca9a228-2809-11e6-98f8-1be925e1f853.jpg)

Here is what a typical LSTM looks like:

![](https://cloud.githubusercontent.com/assets/10870023/15727062/2ca75a68-2809-11e6-908e-a2ff48a3614c.jpg)

Here is what a typical GRU looks like:

![](https://cloud.githubusercontent.com/assets/10870023/15727061/2ca6424a-2809-11e6-8d30-8dff3119f48e.jpg)

Here is what a typical CNN looks like:

![](https://cloud.githubusercontent.com/assets/10870023/15726571/704d28e0-2806-11e6-92c8-bcbeb385671f.jpg)

## Setup

Requirements:

- [Torch][10]
- [loadcaffe][9]

Download the [MSCOCO][11] train+val images and [VQA][1] data using `sh data/download_data.sh`. Extract all the downloaded zip files inside the `data` folder.

```
unzip Annotations_Train_mscoco.zip
unzip Questions_Train_mscoco.zip
unzip train2014.zip

unzip Annotations_Val_mscoco.zip
unzip Questions_Val_mscoco.zip
unzip val2014.zip
```

Download the [VGG-19][7] Caffe model and prototxt using `sh models/download_models.sh`.

### Known issues

- When you install torch/lua, LuaJIT is the default download version. However, LuaJIT limits the usage of the system memory up to 2G. 
  To avoid memory issues with LuaJIT, install Torch with Lua 5.1 (`TORCH_LUA_VERSION=LUA51 ./install.sh`).
  More instructions [here][4].
- If working with plain Lua, [luaffifb][8] may be needed for [loadcaffe][9],unless using pre-extracted fc7 features.
- The default of all the source file is set to GPU mode. 
  CPU mode is supported, but i don't think it is a good idea to train such large network in CPU.
  [Embedding Ads](http://www.geforce.com/hardware/10series/geforce-gtx-1080)

## Pretrained Model and Data

Since the original dataset is quite large (~21G),[You can click here to download Pretained Model and Data](https://www.dropbox.com/sh/fu7c9bojbl7w4o4/AAAHgZzMqRsG8Afj0bIBx30da?dl=0) 

## Usage

### Extract image features

```
th extract_fc7.lua -split train
th extract_fc7.lua -split val
```

#### Options

- 'For more information, please check the comment inside the file'


### Training

```
th train.lua -gpuid 0 -model gru -`add more options`
```

#### Options
- `model`: Model used for training (rnn | lstm | gru). Default is lstm.
- 'For more information please check the comment inside the file'

### Testing

```
th predict.lua -checkpoint_file checkpoints/vqa_epoch23.26_0.4610.t7 -input_image_path data/train2014/COCO_train2014_000000405541.jpg -question 'What is the cat on?' -`add more options`
```

#### Options

- 'For more information please check the comment inside the source code file'

## Sample predictions

Sample questions and answers predicted by the VIS+RNN/LSTM/GRU model.

![](https://cloud.githubusercontent.com/assets/10870023/15725763/3aff866e-2802-11e6-97ce-7788b9cd7844.png)

![](https://cloud.githubusercontent.com/assets/10870023/15725789/58d71300-2802-11e6-94c6-18bfbf555553.png)


## References

- [Exploring Models and Data for Image Question Answering][2], Ren et al., NIPS15
- [VQA: Visual Question Answering][3], Antol et al., ICCV15
- [Mscoco Dataset][11]


## License

[MIT][12]

[1]: http://visualqa.org/
[2]: http://arxiv.org/abs/1505.02074
[3]: http://arxiv.org/abs/1505.00468
[4]: https://github.com/torch/distro
[5]: http://nlp.stanford.edu/projects/glove/
[6]: http://arxiv.org/abs/1409.1556
[7]: https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md
[8]: https://github.com/facebook/luaffifb
[9]: https://github.com/szagoruyko/loadcaffe
[10]: http://torch.ch/
[11]: http://mscoco.org/
[12]: https://abhshkdz.mit-license.org/
