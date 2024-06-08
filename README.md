<a id='1'></a>
# BETR Implementation for Trajectory Prediction

In this repository I have Implemented VectorNet: [Encoding HD Maps and Agent Dynamics from Vectorized Representation](https://arxiv.org/abs/2005.04259) paper.The implementation includes modifications to both the vector representation and the model architecture, reflecting my own vision for enhancing the original approach.

I also have trained this architecture on [**Argoverse2** Dataset](https://www.argoverse.org/av2.html) and tested it using suitable metrics (Mean Displacement Error `MDE`, Final Displacement Error `FDE`, Miss Rate `MR`).

---
<a id='2'></a>
# Table of Contents: 
* Vectornet Custom Implementation.
  * [Description.](#1)
  * [Table of Contents.](#2)
  * [Customization.](#3)
  * [Installation.](#4)
  * [Usage.](#5)
    * [Preprocess Argoverse Data.](#6)
    * [Train Model.](#7)


<a id='3'></a>
# Customization: 

## __First__
I have extracted my vector representation from data such that:


### Agent Vectors: 
$$V_i = [\begin{matrix} x_s & y_s & x_e & y_e & ts_{avg} & Candidate Density & vx_{avg} & vy_{avg} & heading_{avg} & P_{id}\end{matrix}]$$

### Object Vectors: 
$$V_i = [\begin{matrix} x_s & y_s & x_e & y_e & ts_{avg} & D_{a_{avg}} & \theta_{a_{avg}} & vx_{avg} & vy_{avg} & heading_{avg} & objtype & P_{id} \end{matrix}]$$

### Lane Vectors: 
$$V_i = [\begin{matrix} x_s & y_s & z_s & x_e & y_e & z_e & I(Intersection)& dir_{avg} & type & line_{id} \end{matrix}]$$

## **Second**
I have replaced `GNN based Graph Local Encoders` with other vector encoder for each type of vector using `transformers and attention mechanism`, I have also used **Positional Encoding** based on time step of each vector to hold time information.

I also replaced normal `Global GNN layer` with Attentional GNN to better encode vectors of each type.

I Used a simple **Decoder** consists of `three MLP layers` to compare it with other model regardeless of decoder architecture.

<p align="center">
  <img src="https://i.ibb.co/x8BXPB9/Screenshot-from-2023-12-11-18-19-33.png" alt="Screenshot from 2023-12-11 18-19-33" border="0">
</p>
<p align="center">
  <em>fig.1 Global Architecture</em>
</p>


<p align="center">
  <img src="https://i.ibb.co/VCYDLxH/Screenshot-from-2023-12-11-18-23-17.png" alt="Screenshot from 2023-12-11 18-23-17" border="0" width="45%">
  <img src="https://i.ibb.co/ZJmW2H8/Screenshot-from-2023-12-11-18-24-09.png" alt="Screenshot from 2023-12-11 18-24-09" border="0" width="45%">
</p>
<p align="center">
  <em>Transformer inside design</em> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <em>Dense Layer inside design</em> 
</p>

----
<a id='4'></a>
# Installation:

Required Python Libraries: 
<pre>
    torch-geometric
    kornia
    colorama
    universal-pathlib
    plotly
</pre>

<pre>bash
> pip3 install -r requirements.txt
</pre>


<a id='5'></a>
# Usage:

<a id='6'></a>
## Preprocess Argoverse Data: 
Build **Docker** image to preprocess data in multithreads

<pre>bash
> docker build -t preprocess .
</pre>

Run script to run processing image: 
<pre>run.sh
  #!/bin/bash
  echo passwd | sudo -S rm /usr/bin/logs
  cd /usr/bin
  echo passwd | sudo -S touch /usr/bin/logs
  echo passwd | sudo -S chmod +777 /usr/bin/logs
  echo "Starting..." | cat >> logs
  while ! ping -c 1 google.com; do
  	sleep 5
  done
  
  echo "Internet Available..." | cat >> logs
  echo passwd | sudo -S systemctl start docker 
  sleep 10s
  echo passwd | sudo -S docker ps -a >> logs

  docker run --rm -it -v "/home/mahmoud":/main --gpus all preprocess
</pre>

<pre>bash
> sh run.sh
</pre>


<a id='7'></a>
## Train model: 
Build **Docker** image to train model using processed data.
Edit in `config.py` file to be compatible with your own configurations

<pre>bash
> docker build -t train .
</pre>

Run script to run training image: 
<pre>run.sh
  #!/bin/bash
  echo passwd | sudo -S rm /usr/bin/logs
  cd /usr/bin
  echo passwd | sudo -S touch /usr/bin/logs
  echo passwd | sudo -S chmod +777 /usr/bin/logs
  echo "Starting..." | cat >> logs
  while ! ping -c 1 google.com; do
  	sleep 5
  done
  
  echo "Internet Available..." | cat >> logs
  echo passwd | sudo -S systemctl start docker 
  sleep 10s
  echo passwd | sudo -S docker ps -a >> logs

  docker run --rm -it -v "/home/mahmoud":/main --gpus all train
</pre>

<pre>bash
> sh run.sh
</pre>


