Flow of the code:

1. install all the dependencies. May be you can create a requirements.txt file before jumping directly to the notebook. Because it doesnt require extra ordinary dependencies, however, before running cell 1, please install geoopt, clone the official git repo.


add like this:

!pip install geoopt

!git clone https://github.com/hchau630/LSC
%cd LSC


2. import all the dependencies.

3. setup your cuda (change runtime on Google Colab to GPU)

4. Define functions: 

plot_psi, plot_w, plot_operator, plot_reconstruction, save_loss, plot_loss, get_MNIST_dataloader, initialize_omega

4.1 plot_psi() - plotting the initialisation as image for psi that is part of theta = W,psi

4.2 plot_w() - Also a random initialisation, Given an image I, our goal is to infer the transformation parameters s and sparse code α according to their posterior distribution. Given a large ensemble of images, our goal is to learn the parameters θ = {W, Φ} by maximising their log-likelihood

4.3 plot_operator() - explicitly models the transformations, so that images are now represented as transformations of patterns generated from the sparse coding model

4.4 plot_reconstruction() - Reconstructs the input using MAP estimate of s.

4.5 save_loss() - logs the losses for each iteration.

4.6 plot_loss() 

4.7 get_MNIST_dataloader() - importing and pre-processing the dataset.



5. Hyperparameters
B (batch size) 100
K (number of dictionary templates) 10
N (number of samples along each dimension of the integral R¯ = Rs Pθ(s|I, α)R(s)) 50
L (number of irreducible representations) 128
T (number of gradient update steps for α) 20
n (dimensionality of transformation parameter s) 2
σ2 (variance of the Gaussian noise  in the generative model) 0.01
λ (sparse penalty) 10
ηΦ (learning rate for Φ) 0.05
ηW (learning rate for W) 0.3
α0 (initialization of α) 0.01
multiplicity of ω 1
parameters for geoopt Riemannian ADAM optimizer (excluding learning rate) default

6. Visualize the synthetic dataset to check whether your transformations and translations has worked or not. With the help of previously defined function get_custom_dataloader you will get the custom rotations and transformations. It includes Rotation and Scaling, Horizontal and Vertical Translation.

7. Train the netwrok for 10 epochs
# Tunable parameters
eta_psi = 0.05
eta_w = 0.3
warmup_epochs = 3
adjust_psi_learning_rate = create_psi_learning_rate_adjuster(0.0, eta_psi, warmup_epochs*num_train_batches)
adjust_w_learning_rate = create_w_learning_rate_adjuster(0.0, eta_w, warmup_epochs*num_train_batches)
w_optimizer = geoopt.optim.RiemannianAdam([w], lr=eta_w)
unique_omega = torch.unique(omega, dim=0) # (L,n)
s = 0.1
lamb = 1.0 # sparsity cost
lamb2 = 0.0 # slow feature cost 
lamb3 = 0.0 # psi regularization
epochs = 10 # 20000
save_skip = 100
steps = 20
N_samples = 100

# These options can be changed as needed.
test = True
save = True
annealing = False
shuffle = True
warmup = True

# The options below should not need to be changed.
adaptive = True
adaptive_steps = False
modified = False
video = False
map_est = False
rgb = False
plot = False



8. Now plot Loss-Epoch curve.

9. With the help of previously defined functions plot different charts.

10. Get the visualisations showing how your tun-able parameters adjust your outputs in different iterations of the training process.

11. With reconstruction images, you will observe that the model has learned how each digit is being drawn and have decided the atoms from the dictionary for the same MNIST digit. For each dataset, LSC is able to learn the 10 digits as well as the two operators that generated it. each of the learned dictionary template Φi corresponds to one of the digits.

12. even though the rotation + scaling dataset contains only rotations between −75◦ and 75◦, the model learns the full 360◦ rotation. This ability to generalize and extrapolate correctly the transformation present in the dataset is a feature of the Lie group structure that is built into LSC.

13. Even with latent traversals of the transformation parameters s1 and s2, obtained by applying T (s) with varying values of s to five images from each test set, model is able to learn the digits.

Rest of the explanation is in report.