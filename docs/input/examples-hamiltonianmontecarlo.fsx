﻿(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#r "../../src/DiffSharp/bin/Debug/FsAlg.dll"
#load "../../packages/FSharp.Charting.0.90.9/FSharp.Charting.fsx"

(**
Hamiltonian Monte Carlo
=======================

[Hamiltonian Monte Carlo](http://en.wikipedia.org/wiki/Hybrid_Monte_Carlo) (HMC) is a type of [Markov chain Monte Carlo](http://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) (MCMC) algorithm for obtaining random samples from probability distributions for which direct sampling is difficult. HMC makes use of [Hamiltonian mechanics](http://en.wikipedia.org/wiki/Hamiltonian_mechanics) for efficiently exploring target distributions and provides better convergence characteristics that avoid the slow exploration of random sampling (in alternatives such as the [Metropolis-Hastings algorithm](http://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm)).

The advantages of HMC come at the cost of evaluating gradients of distribution functions, which need to be worked out and supplied by the user. Using the DiffSharp library, we can design an HMC algorithm that only needs the target distribution as its input, computing the needed gradients efficiently through AD.

Let us demonstrate how an AD-HMC can be implemented.

First we need a scheme for integrating Hamiltonian dynamics with discretized time. The [Leapfrog algorithm]() is the common choice, due to its [symplectic](http://en.wikipedia.org/wiki/Symplectic_integrator) property and straightforward implementation.

Hamiltonian mechanics is a formulation of classical mechanics, describing the time evolution of a system by the equations

$$$
  \begin{eqnarray*}
    \frac{\textrm{d}\mathbf{p}}{\textrm{d}t} &=& -\frac{\partial H}{\partial\mathbf{x}}\\
    \frac{\textrm{d}\mathbf{x}}{\textrm{d}t} &=& +\frac{\partial H}{\partial\mathbf{p}}\;,\\
  \end{eqnarray*}

where $\mathbf{p}$ and $\mathbf{x}$ are the momentum and the position of an object.

$H(\mathbf{x},\mathbf{p}) = U(\mathbf{x}) + K(\mathbf{p})$ is called the _Hamiltonian_, corresponding to the sum of the potential energy $U$ and the kinetic energy $K$ of the system.

The leapfrog algorithm integrates the time evolution of the system via the updates

$$$
  \begin{eqnarray*}
    \mathbf{p}(t+\frac{\delta}{2}) &=& \mathbf{p}(t) - \frac{\delta}{2}\, \nabla_{\mathbf{x}(t)}\, U(\mathbf{x})\\
    \mathbf{x}(t+\delta) &=& \mathbf{x}(t) + \delta\, \nabla_{\mathbf{p}(t+\frac{\delta}{2})}\, K(\mathbf{p})\\
    \mathbf{p}(t+\delta) &=& \mathbf{p}(t+\frac{\delta}{2}) - \frac{\delta}{2}\, \nabla_{\mathbf{x}(t+\delta)}\, U(\mathbf{x})\;,\\
  \end{eqnarray*}

where $\delta$ is the integration step size.

*)

open DiffSharp.AD.Specialized.Reverse1
open DiffSharp.AD.Specialized.Reverse1.Vector
open FsAlg.Generic

// Leapfrog integrator
// u: potential energy function
// k: kinetic energy function
// d: integration step size
// steps: number of integration steps
// (x0, p0): initial position and momentum vectors
let leapFrog u k d steps (x0, p0) =
    let hd = d / 2.
    [1..steps] 
    |> List.fold (fun (x, p) _ ->
        let p' = p - hd * grad u x
        let x' = x + d * grad k p'
        x', p' - hd * grad u x') (x0, p0)

(**

We define simple functions for generating [uniform](http://en.wikipedia.org/wiki/Uniform_distribution_(continuous)) and [standard normal](http://en.wikipedia.org/wiki/Normal_distribution) random numbers.
    
*)

let Rnd = new System.Random()

// Uniform random number ~U(0, 1)
let rnd() = Rnd.NextDouble()

// Standard normal random number ~N(0, 1), via Box-Muller transform
let rec rndn() =
    let x, y = rnd() * 2.0 - 1.0, rnd() * 2.0 - 1.0
    let s = x * x + y *y
    if s > 1.0 then rndn() else x * sqrt (-2.0 * (log s) / s)

(**

Now we have everything ready for our HMC implementation.

Briefly, the the essence of HMC is to sample the state space with Hamiltonian dynamics via using the potential energy term

$$$
  U(\mathbf{x}) = - \textrm{log} \, f(\mathbf{x})\;,

where $f(\mathbf{x})$ is the target density. The kinetic energy term is commonly taken as

$$$
  K(\mathbf{p}) = \frac{\mathbf{p}^T \mathbf{p}}{2}\; .

Starting from a given value of $\mathbf{x}$, the algorithm proceeds via the steps of sampling a random momentum $\mathbf{p}$, running the Hamiltonian dynamics for a set number of steps to arrive at $\mathbf{x}^*$ and $\mathbf{p}^*$, and testing a Metropolis acceptance criterion for updating $\mathbf{x} \leftarrow \mathbf{x}^*$.

*)

// Hamiltonian Monte Carlo
// n: number of samples wanted
// hdelta: step size for Hamiltonian dynamics
// hsteps: number of steps for Hamiltonian dynamics
// x0: initial state
// f: target distribution function
let hmc n hdelta hsteps (x0:Vector<_>) (f:Vector<Adj>->Adj) =
    let u x = -log (f x) // potential energy
    let k p = 0.5 * Vector.fold (fun acc a -> acc + a * a) (adj 0.) p // kinetic energy
    let hamilton x p = u (Vector.map adj x) + k (Vector.map adj p) |> float
    let x = ref x0
    [|for i in 1..n do
        let p = Vector.init x0.Length (fun _ -> rndn())
        let x', p' = leapFrog u k hdelta hsteps (!x, p)
        if rnd() < exp ((hamilton !x p) - (hamilton x' p')) then x := x'
        yield !x|]

(**

Whereas the classical HMC requires the user to supply the log-density and also its gradient, in our implementation we need to supply only the target density function. The rest is taken care of by the reverse mode AD module **DiffSharp.AD.Reverse**. This has two main advantages: AD computes the exact gradient efficiently and it is also applicable to complex density functions where closed-form expressions for the gradient cannot be formulated.

Let us now test this HMC algorithm with a [multivariate normal distribution](http://en.wikipedia.org/wiki/Multivariate_normal_distribution), which has the density

$$$
  f_{\mathbf{x}}(x_1,\dots,x_k) = \frac{1}{\sqrt{(2\pi)^k \left|\mathbf{\Sigma}\right|}} \textrm{exp} \left( -\frac{1}{2} (\mathbf{x} - \mathbf{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \mathbf{\mu}) \right)\;,

where $\mathbf{\mu}$ is the mean vector and $\mathbf{\Sigma}$ is the covariance matrix.

*)

// Multivariate normal distribution (any dimension)
// mu: mean vector
// sigma: covariance matrix
// x: variable vector
let multiNormal mu sigma (x:Vector<Adj>) =
    let s = sigma |> Matrix.inverse |> Matrix.map adj
    let m = mu |> Vector.map adj
    exp (-((x - m) * s * (x - m)) / 2.)


(**

Here we plot 10000 samples from the bivariate case with $\mathbf{\mu} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$ and $\mathbf{\Sigma} = \begin{bmatrix} 1 & 0.8 \\ 0.8 & 1 \end{bmatrix}\,$.

*)

// Take 10000 samples from a bivariate normal distribution
// mu1 = 0, mu2 = 0, correlation = 0.8
let samples = 
    multiNormal (vector [0.; 0.]) (matrix [[1.; 0.8]; [0.8; 1.]])
    |> hmc 10000 0.1 10 (vector [0.; 0.])


open FSharp.Charting

Chart.Point(samples |> Array.map (fun v -> v.[0], v.[1]), MarkerSize = 2)


(**

<div class="row">
    <div class="span6 offset1">
        <img src="img/examples-hamiltonianmontecarlo-chart1.png" alt="Chart" style="width:550px"/>
    </div>
</div>

*)