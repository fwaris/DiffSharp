﻿//
// This file is part of
// DiffSharp -- F# Automatic Differentiation Library
//
// Copyright (C) 2014, National University of Ireland Maynooth.
//
//   DiffSharp is free software: you can redistribute it and/or modify
//   it under the terms of the GNU General Public License as published by
//   the Free Software Foundation, either version 3 of the License, or
//   (at your option) any later version.
//
//   DiffSharp is distributed in the hope that it will be useful,
//   but WITHOUT ANY WARRANTY; without even the implied warranty of
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//   GNU General Public License for more details.
//
//   You should have received a copy of the GNU General Public License
//   along with DiffSharp. If not, see <http://www.gnu.org/licenses/>.
//
// Written by:
//
//   Atilim Gunes Baydin
//   atilimgunes.baydin@nuim.ie
//
//   Barak A. Pearlmutter
//   barak@cs.nuim.ie
//
//   Hamilton Institute & Department of Computer Science
//   National University of Ireland Maynooth
//   Maynooth, Co. Kildare
//   Ireland
//
//   www.bcl.hamilton.ie
//

#light

/// Forward AD module, lazy higher-order
module DiffSharp.AD.ForwardN

open DiffSharp.Util.LinearAlgebra
open DiffSharp.Util.General

/// DualN numeric type, where the tangent value is another DualN, forming a lazy chain of higher-order derivatives
// UNOPTIMIZED
type DualN =
    // Primal, tangent
    | DualN of float * Lazy<DualN>
    static member Create(p) = DualN(p, Lazy<DualN>(fun () -> DualN.Zero))
    static member Create(p, t) = DualN(p, Lazy<DualN>(fun () -> DualN.Create(t)))
    static member op_Explicit(p) = DualN.Create(p)
    static member op_Explicit(DualN(p, _)) = p
    override d.ToString() = let (DualN(p, t)) = d in sprintf "DualN (%f, %f)" p (float t.Value)
    member d.P = let (DualN(p, _)) = d in p
    member d.T = let (DualN(_, t)) = d in t.Value
    static member Zero = DualN(0., Lazy<DualN>(fun () -> DualN.Zero))
    static member One = DualN(1., Lazy<DualN>(fun () -> DualN.Zero))
    // DualN - DualN binary operations
    static member (+) (a:DualN, b:DualN) = DualN(a.P + b.P, Lazy<DualN>(fun () -> a.T + b.T))
    static member (-) (a:DualN, b:DualN) = DualN(a.P - b.P, Lazy<DualN>(fun () -> a.T - b.T))
    static member (*) (a:DualN, b:DualN) = DualN(a.P * b.P, Lazy<DualN>(fun () -> a.T * b + a * b.T))
    static member (/) (a:DualN, b:DualN) = DualN(a.P / b.P, Lazy<DualN>(fun () -> (a.T * b - a * b.T) / (b * b)))
    static member Pow (a:DualN, b:DualN) = DualN(a.P ** b.P, Lazy<DualN>(fun () -> (a ** b) * ((b * a.T / a) + ((log a) * b.T))))
    // DualN - float binary operations
    static member (+) (a:DualN, b) = DualN(a.P + b, Lazy<DualN>(fun () -> a.T))
    static member (-) (a:DualN, b) = DualN(a.P - b, Lazy<DualN>(fun () -> a.T))
    static member (*) (a:DualN, b) = DualN(a.P * b, Lazy<DualN>(fun () -> a.T * b))
    static member (/) (a:DualN, b) = DualN(a.P / b, Lazy<DualN>(fun () -> a.T / b))
    static member Pow (a:DualN, b) = DualN(a.P ** b, Lazy<DualN>(fun () -> b * (a ** (b - 1.)) * a.T))
    // float - DualN binary operations
    static member (+) (a, b:DualN) = DualN(b.P + a, Lazy<DualN>(fun () -> b.T))
    static member (-) (a, b:DualN) = DualN(a - b.P, Lazy<DualN>(fun () -> -b.T))
    static member (*) (a, b:DualN) = DualN(b.P * a, Lazy<DualN>(fun () -> b.T * a))
    static member (/) (a, b:DualN) = DualN(a / b.P, Lazy<DualN>(fun () -> -a * b.T / (b * b)))
    static member Pow (a, b:DualN) = DualN(a ** b.P, Lazy<DualN>(fun () -> (DualN.Create(a) ** b) * (log a) * b.T))
    // DualN - int binary operations
    static member (+) (a:DualN, b:int) = a + float b
    static member (-) (a:DualN, b:int) = a - float b
    static member (*) (a:DualN, b:int) = a * float b
    static member (/) (a:DualN, b:int) = a / float b
    static member Pow (a:DualN, b:int) = DualN.Pow(a, float b)
    // int - DualN binary operations
    static member (+) (a:int, b:DualN) = (float a) + b
    static member (-) (a:int, b:DualN) = (float a) - b
    static member (*) (a:int, b:DualN) = (float a) * b
    static member (/) (a:int, b:DualN) = (float a) / b
    static member Pow (a:int, b:DualN) = DualN.Pow(float a, b)
    // DualN unary operations
    static member Log (a:DualN) = DualN(log a.P, Lazy<DualN>(fun () -> a.T / a))
    static member Exp (a:DualN) = DualN(exp a.P, Lazy<DualN>(fun () -> a.T * exp a))
    static member Sin (a:DualN) = DualN(sin a.P, Lazy<DualN>(fun () -> a.T * cos a))
    static member Cos (a:DualN) = DualN(cos a.P, Lazy<DualN>(fun () -> -a.T * sin a))
    static member Tan (a:DualN) = DualN(tan a.P, Lazy<DualN>(fun () -> a.T / ((cos a) * (cos a))))
    static member (~-) (a:DualN) = DualN(-a.P, Lazy<DualN>(fun () -> -a.T))
    static member Sqrt (a:DualN) = DualN(sqrt a.P, Lazy<DualN>(fun () -> a.T / (2. * sqrt a)))
    static member Sinh (a:DualN) = DualN(sinh a.P, Lazy<DualN>(fun () -> a.T * cosh a))
    static member Cosh (a:DualN) = DualN(cosh a.P, Lazy<DualN>(fun () -> a.T * sinh a))
    static member Tanh (a:DualN) = DualN(tanh a.P, Lazy<DualN>(fun () -> a.T / ((cosh a) * (cosh a))))
    static member Asin (a:DualN) = DualN(asin a.P, Lazy<DualN>(fun () -> a.T / sqrt (1. - a * a)))
    static member Acos (a:DualN) = DualN(acos a.P, Lazy<DualN>(fun () -> -a.T / sqrt (1. - a * a)))
    static member Atan (a:DualN) = DualN(atan a.P, Lazy<DualN>(fun () -> a.T / (1. + a * a)))

/// DualN operations module (automatically opened)
[<AutoOpen>]
module DualNOps =
    /// Make DualN, with primal value `p` and tangent 0
    let inline dualN p = DualN.Create(p, 0.)
    /// Make DualN, with primal value `p` and tangent value `t`
    let inline dualNSet (p, t) = DualN.Create(p, t)
    /// Make active DualN (i.e. variable of differentiation), with primal value `p` and tangent 1
    let inline dualNAct p = DualN.Create(p, 1.)
    /// Make an array of arrays of DualN, with primal values given in array `x`. The tangent values along the diagonal are 1, the rest are 0.
    let inline dualNActArrayArray (x:float[]) = Array.init x.Length (fun i -> (Array.init x.Length (fun j -> if i = j then dualNAct x.[j] else dualN x.[j])))
    /// Get the primal value of a DualN
    let inline primal (DualN(p, _)) = p
    /// Get the tangent value of a DualN
    let inline tangent (DualN(_, t)) = float t.Value
    /// Get the tangent-of-tangent value of a DualN
    let inline tangent2 (d:DualN) = tangent d.T
    /// Get the primal and tangent value of a DualN, as a tuple
    let inline tuple (DualN(p, t)) = (p, float t.Value)

    /// Compute the `n`-th derivative of a DualN
    let rec diffLazy n =
        match n with
        | a when a < 0 -> invalidArg "n" "Order of derivative cannot be negative."
        | 0 -> fun (x:DualN) -> x
        | 1 -> fun x -> x.T
        | _ -> fun x -> diffLazy (n - 1) x.T

    /// Custom operator (/^) for differentiation. Usage: `x` /^ `n`, value of the `n`-th order derivative of `x`.
    let ( /^ ) x n =
        diffLazy n x |> primal


/// ForwardN differentiation operations module (automatically opened)
[<AutoOpen>]
module ForwardNOps =
    /// Original value and first derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff' f x =
        dualNAct x |> f |> tuple
    
    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff f x =
        dualNAct x |> f |> tangent

    /// Original value and second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2' f x =
        dualNAct x |> f |> fun a -> (primal a, tangent2 a)
        
    /// Second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2 f x =
        dualNAct x |> f |> tangent2

    /// Original value, first derivative, and second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2'' f x =
        dualNAct x |> f |> fun a -> (primal a, tangent a, tangent2 a)

    /// `n`-th derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diffn n f x =
        dualNAct x |> f |> diffLazy n |> primal

    /// Original value and the `n`-th derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diffn' n f x =
        let orig = x |> dualNAct |> f
        let d = orig |> diffLazy n
        (primal orig, primal d)

    /// Original value and gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline gradv' f x v =
        Array.zip x v |> Array.map dualNSet |> f |> tuple

    /// Gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline gradv f x v =
        gradv' f x v |> snd

    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad' f x =
        let a = Array.map f (dualNActArrayArray x)
        (primal a.[0], Array.map tangent a)

    /// Gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad f x =
        grad' f x |> snd
            
    /// Original value and Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian' f x =
        let a = Array.map f (dualNActArrayArray x)
        (let (DualN(p, _)) = a.[0] in p, Array.sumBy tangent2 a)

    /// Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian f x =
        laplacian' f x |> snd

    /// Original value and transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT' f x =
        let a = Array.map f (dualNActArrayArray x)
        (Array.map primal a.[0], Array2D.map tangent (array2D a))

    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT f x =
        jacobianT' f x |> snd

    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian' f x =
        jacobianT' f x |> fun (v, j) -> (v, transpose j)

    /// Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian f x =
        jacobian' f x |> snd

    /// Original value and Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianv' f x v = 
        Array.zip x v |> Array.map dualNSet |> f |> Array.map tuple |> Array.unzip

    /// Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianv f x v = 
        jacobianv' f x v |> snd

/// Module with differentiation operators using Vector and Matrix input and output, instead of float[] and float[,]
module Vector =
    /// Original value and first derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff' f x = ForwardNOps.diff' f x
    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff f x = ForwardNOps.diff f x
    /// Original value and second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2' f x = ForwardNOps.diff2' f x
    /// Second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2 f x = ForwardNOps.diff2 f x
    /// Original value, first derivative, and second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2'' f x = ForwardNOps.diff2'' f x
    /// Original value and the `n`-th derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diffn' n f x = ForwardNOps.diffn' n f x 
    /// `n`-th derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diffn n f x = ForwardNOps.diffn n f x
    /// Original value and directional derivative of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline gradv' f x v = ForwardNOps.gradv' f (array x) (array v)
    /// Directional derivative of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline gradv f x v = ForwardNOps.gradv f (array x) (array v)
    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad' f x = ForwardNOps.grad' f (array x) |> fun (a, b) -> (a, vector b)
    /// Gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad f x = ForwardNOps.grad f (array x) |> vector
    /// Original value and Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian' f x = ForwardNOps.laplacian' f (array x)
    /// Laplacian of a vector-to-scalar function `f`, at point x
    let inline laplacian f x = ForwardNOps.laplacian f (array x)
    /// Original value and transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT' f x = ForwardNOps.jacobianT' f (array x) |> fun (a, b) -> (vector a, matrix b)
    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT f x = ForwardNOps.jacobianT f (array x) |> matrix
    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian' f x = ForwardNOps.jacobian' f (array x) |> fun (a, b) -> (vector a, matrix b)
    /// Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian f x = ForwardNOps.jacobian f (array x) |> matrix
    /// Original value and Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianv' f x v = ForwardNOps.jacobianv' f (array x) (array v) |> fun (a, b) -> (vector a, vector b)
    /// Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianv f x v = ForwardNOps.jacobianv f (array x) (array v) |> vector