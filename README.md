<strong>VHAGaR:</strong><br />
In this repository, we provide an implementation for the paper "Verification of Neural Networks' Global Robustness" <a href="https://arxiv.org/abs/2402.19322">VHAGaR's paper</a>. The repository owner is anan.kabaha@campus.technion.ac.il. 

<strong>Prerequisites:</strong><br />
<div style="background-color: #f2f2f2; padding: 1px;">
  <pre style="font-family: 'Courier New', monospace; font-size: 14px;">
  Julia Version 1.11.3
  Gurobi 10.0 or 11.0 or 12.0 
  Python 3.8.10 
  Torch 2.4.1
</pre>
</div>

<strong>Clone VHAGaR:</strong><br />
<div style="background-color: #f2f2f2; padding: 1px;">
  <pre style="font-family: 'Courier New', monospace; font-size: 14px;">
  git clone https://github.com/ananmkabaha/VHAGaR.git
  cd to VHAGaR
  run julia run.jl
</pre>
</div>

<strong>VHAGaR parameters:</strong><br />
--dataset: the dataset, can be either mnist, fmnist, cifar10<br />
--model_name: the model name, can be either FC0, FC1, CNN0, CNN1, or CNN2<br />
--model_path: the path of the model<br />
--perturbation: the peturbation can be either occ, patch, brightness, linf, contrast, translation, rotation, or max <br />
--perturbation_size: the size of the pertubation in this format: occ: i,j,width , patch: eps,i,j,width, brightness: eps, linf: eps, contrast: eps, translation: tx,ty, rotation: angle<br />
--ctag: the source class<br />
--ct: the target classes<br />
--timeout: the MIP timeout<br />
--output_dir: the output directoy<br />

<strong>Examples:</strong><br />
julia run.jl --dataset mnist --model_name CNN1 -m ./models/mnist_CNN1/model.p --perturbation occ --perturbation_size 1,1,5 --output_dir ./results/ --ctag 1 --ct 2,3,4,5,6,7,8,9,10 --timeout 10800<br />
