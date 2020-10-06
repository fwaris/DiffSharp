﻿namespace DiffSharp

open DiffSharp.ShapeChecking
open DiffSharp.Util

#if SYMBOLIC_SHAPES

/// <summary>
///  Represents an integer that may be symbolic, e.g. the size of one dimension of a tensor,
///  or an index into a tensor.
/// </summary>
///
/// <remarks>
///  Note that symbolic integers only appear when using Backend.ShapeChecking.  Otherwise
///  it can always be assumed that the symbol is empty.
/// </remarks>
[<Struct; CustomEquality; CustomComparison>]
type Int internal (n: int, sym: Symbol) = 

    static member inline internal unop (x: Int) f1 f2 =
        match x.TryGetValue() with 
        | ValueSome xv -> f1 xv
        | ValueNone -> f2 x.SymbolRaw

    static member inline internal binop (x: Int) (y: Int) f1 f2 =
        match x.TryGetValue(), y.TryGetValue() with 
        | ValueSome xv, ValueSome yv -> f1 xv yv
        | ValueNone, ValueNone -> f2 x.SymbolRaw y.SymbolRaw
        | ValueSome _, ValueNone ->
            let symX = x.AsSymbol(y.SymbolRaw.SymbolScope)
            f2 symX y.SymbolRaw
        | ValueNone, ValueSome _ ->
            let symY = y.AsSymbol(x.SymbolRaw.SymbolScope)
            f2 x.SymbolRaw symY

    new (n: int) = Int(n, Unchecked.defaultof<_>)

    static member FromSymbol (sym: Symbol) = Int(0, sym)

    member internal x.SymbolRaw : Symbol = sym

    member x.AsSymbol(syms: SymbolScope) =
        match box sym with 
        | null  -> syms.CreateConst(x)
        | _ -> sym

    member x.TryGetName() =
        match box sym with 
        | null -> ValueNone
        | _ -> ValueSome (sym.ToString())

    member x.TryGetValue() =
        match box sym with 
        | null -> ValueSome n
        | _ -> Int.TryEvaluate(sym)

    static member TryEvaluate(sym: Symbol) =
        match sym.TryEvaluate() with 
        | ValueSome (:? int as n) -> ValueSome n
        | _ -> ValueNone

    /// Return the value, exception if symbolic
    member x.Value =
        match x.TryGetValue() with 
        | ValueNone -> failwithf "can't simulate value of symbolic integer expressoin %s" (sym.ToString())
        | ValueSome v -> v

    /// Return the value, or '1' if this has no definite solution, normally to get a representative value
    member x.ValueOrOne =
        match x.TryGetValue() with 
        | ValueNone -> 1
        | ValueSome v -> v

    static member (+) (a:Int, b:Int) : Int =
        Int.binop a b (fun a b -> Int (a+b)) (fun a b -> Int.FromSymbol (Symbol.binop "add" a b))

    static member (+) (a:Int, b:int) : Int = a + Int b

    static member (+) (a:int, b:Int) : Int = Int a + b

    static member (-) (a:Int, b:Int) : Int =
        Int.binop a b (fun a b -> Int (a-b)) (fun a b -> Int.FromSymbol (Symbol.binop "sub" a b))

    static member (-) (a:Int, b:int) : Int = a - Int b

    static member (-) (a:int, b:Int) : Int = Int a - b

    static member (%) (a:Int,b:Int) : Int =
        Int.binop a b (fun a b -> Int (a%b)) (fun a b -> Int.FromSymbol (Symbol.binop "mod" a b))

    static member (%) (a:Int, b:int) : Int = a % Int b

    static member (%) (a:int, b:Int) : Int = Int a % b

    static member (*) (a:Int,b:Int) : Int = 
        Int.binop a b (fun a b -> Int (a*b)) (fun a b -> Int.FromSymbol (Symbol.binop "mul" a b))

    static member (*) (a:Int, b:int) : Int = a * Int b

    static member (*) (a:int, b:Int) : Int = Int a * b

    static member (/) (a:Int,b:Int) : Int = 
        Int.binop a b (fun a b -> Int (a*b)) (fun a b -> Int.FromSymbol (Symbol.binop "div" a b))

    static member (/) (a:Int, b:int) : Int = a / Int b

    static member (/) (a:int, b:Int) : Int = Int a / b

    /// Negation operator
    static member (~-) (a:Int) : Int = 
        Int.unop a (fun a -> Int (-a)) (fun a -> Int.FromSymbol (Symbol.unop "neg" a))

    /// Constraint equality
    static member (=~=) (a:Int,b:Int) : bool = 
        Int.binop a b (fun a b -> a = b) (fun a b -> a.Solve(b))

    /// Constraint less-than-or-equal. Returns true if no contradiciton was detected when the constraint was asserted.
    static member (<=~) (a:Int,b:Int) : bool = 
        Int.binop a b (fun a b -> a <= b) (fun a b -> a.SymbolScope.Constrain("<=", [a;b]))

    static member Zero = Int 0

    static member Abs(dim: Int) = Int (abs dim.Value)

    member _.IsRequest = (n = -1)

    member _.IsInvalid = (n < -1)

    /// Allow injection of symbol variables
    static member Symbolic (sym: Symbol) = Int.FromSymbol sym

    override x.GetHashCode() =
        match x.TryGetValue() with 
        | ValueNone -> 0
        | ValueSome v -> v

    override x.Equals(y:obj) =
          match y with 
          | :? Int as y -> Int.binop x y (=) (fun xsym ysym -> xsym.KnownToBeEqual(ysym))
          | _ -> false

    interface System.IComparable with 
       member x.CompareTo(y:obj) = 
          match y with 
          | :? Int as y -> compare x.Value y.Value // TODO - symbols
          | _ -> failwith "wrong type"

    override x.ToString() =
        match x.TryGetValue() with 
        | ValueNone -> string sym
        | ValueSome v -> string v

/// Represents the shape of a tensor.  Each dimension may be symbolic.
[<Struct; CustomEquality; NoComparison>]
type Shape internal (values: int[], dims: Int[]) = 

    static member inline internal unop (x: Shape) f1 f2 =
        match x.TryGetValues() with 
        | ValueSome xv -> f1 xv
        | ValueNone -> f2 x.Dims

    static member inline internal binop (x: Shape) (y: Shape) f1 f2 =
        match x.TryGetValues(), y.TryGetValues() with 
        | ValueSome xv, ValueSome yv -> f1 xv yv
        | _, _ -> f2 x.Dims y.Dims

    /// Creates a constant shape from an array of integers
    new (values: int[]) = Shape(values, null)

    /// Creates a possibly-symbolic shape from an array of possibly-symbolic integers
    new (arr: Int[]) = Shape(null, arr)

    /// Get the number of dimensions in the shape
    member _.Length =
        match values with 
        | null -> dims.Length
        | _ -> values.Length

    /// Get the possibly-symbolic dimensions of the shape
    member _.Dims =
        match values with 
        | null -> dims
        | _ -> values |> Array.map Int

    /// <summary>Get the values of the shape. Raises an exception if any of the dimensions are symbolic.</summary>
    /// <remarks>Symbolic dimensions will only appear when Backend.ShapeChecking is used.</remarks>
    member _.TryGetValues() =
        match values with 
        | null ->
            let vs = dims |> Array.map (fun dim -> dim.TryGetValue())
            if vs |> Array.forall (fun v -> v.IsSome) then
                ValueSome (vs |> Array.map (fun v -> v.Value))
            else ValueNone
        | _ -> ValueSome values

    /// <summary>Get the values of the shape. Raises an exception if any of the dimensions are symbolic.</summary>
    /// <remarks>Symbolic dimensions will only appear when Backend.ShapeChecking is used.</remarks>
    member shape.Values =
        match shape.TryGetValues() with 
        | ValueSome values -> values
        | ValueNone -> failwithf "the shape '%A' is symbolic" shape

    /// <summary>Get a length of a particular dimension of the shape.</summary>
    /// <remarks>If the shape is symbolic then the length may be symbolic.</remarks>
    member _.Item with get i = 
        match values with 
        | null -> dims.[i]
        | _ -> Int values.[i]

    /// <summary>Gets the total number of elements in the shape.</summary>
    /// <remarks>
    ///   Raises an exception if any of the dimensions are symbolic. 
    ///   Symbolic dimensions will only appear when Backend.ShapeChecking is used.
    /// </remarks>
    member shape.nelement =
        if shape.Length = 0 then 1
        else Array.reduce (*) shape.Values

    /// Gets the total number of elements in a possibly-symbolic shape
    member shape.nelementx =
        if shape.Length = 0 then Int 1
        else Array.reduce (*) shape.Dims

    /// Gets the total number of elements in a possibly-symbolic shape
    member shape.flatten() = 
        match values with 
        | null -> Shape [| shape.nelementx |]
        | _ -> Shape [| shape.nelement |]

    override x.Equals(y:obj) =
        match y with 
        | :? Shape as y ->
            match values, y.ValuesRaw with 
            | _, null | null, _ -> x.Dims = y.Dims
            | xvalues, yvalues -> xvalues = yvalues
        | _ -> false

    override shape.GetHashCode() = hash shape.Dims

    /// Constraint equality
    static member (=~=) (a:Shape,b:Shape) : bool = 
        Shape.binop a b (fun a b -> a = b) (fun a b -> a.Length = b.Length && (a,b) ||> Array.forall2(=~=))

    member _.GetSlice(low:int option,high:int option) =
        match values with 
        | null -> Shape (FSharp.Core.Operators.OperatorIntrinsics.GetArraySlice dims low high)
        | _ -> Shape (FSharp.Core.Operators.OperatorIntrinsics.GetArraySlice values low high)

    override x.ToString() = "[" + String.concat "," (x.Dims |> Array.map string) + "]"

    member internal _.ValuesRaw = values

    member internal _.DimsRaw = dims

#else
/// Represents an integer used in as a dimension, e.g. the size of one dimension of a tensor,
/// or an index into a tensor.
[<Struct; CustomEquality; CustomComparison>]
type Int (n: int) = 

    /// Return the value
    member x.Value = n

    /// Return the value
    member x.ValueOrOne = n

    static member (+) (a:Int, b:Int) : Int = Int (a.Value + b.Value)

    static member (+) (a:Int, b:int) : Int = Int (a.Value + b)

    static member (-) (a:Int, b:Int) : Int = Int (a.Value - b.Value)

    static member (-) (a:Int, b:int) : Int = Int (a.Value - b)

    static member (%) (a:Int,b:Int) : Int = Int (a.Value % b.Value)

    static member (*) (a:Int,b:Int) : Int = Int (a.Value * b.Value)

    static member (*) (a:int,b:Int) : Int = Int (a * b.Value)

    static member (*) (a:Int,b:int) : Int = Int (a.Value * b)

    static member (/) (a:Int,b:Int) : Int = Int (a.Value / b.Value)

    static member (/) (a:Int,b:int) : Int = Int (a.Value / b)

    static member Zero = Int 0

    static member Abs(dim: Int) = Int (abs dim.Value)

    member _.IsRequest = (n = -1)

    member _.IsInvalid = (n < -1)

    override x.GetHashCode() = n

    override x.Equals(y:obj) =
          match y with 
          | :? Int as y ->  n = y.Value
          | _ -> false

    interface System.IComparable with 
       member x.CompareTo(y:obj) = 
          match y with 
          | :? Int as y -> compare x.Value y.Value
          | _ -> failwith "wrong type"

    override x.ToString() = string x.Value

    /// Constraint equality
    static member (=~=) (a:Int,b:Int) : bool = (a.Value = b.Value)

    /// Constraint less-than-or-equal. 
    static member (<=~) (a:Int,b:Int) : bool = (a.Value <= b.Value)

/// Represents the shape of a tensor.
[<Struct; CustomEquality; NoComparison>]
type Shape (values: int[]) = 

    new (values: Int[]) = 
        let valuesi : int[] = values |> Array.map (fun v -> v.Value)
        Shape (valuesi)

    /// Get the number of dimensions in the shape
    member _.Length = values.Length

    member internal _.ValuesRaw = values

    /// Get the possibly-symbolic dimensions of the shape
    member _.Dims = Array.map Int values

    member _.Values = values

    member _.Item with get i = Int values.[i]

    /// <summary>Gets the total number of elements in the shape.</summary>
    member shape.nelement =
        if shape.Length = 0 then 1
        else Array.reduce (*) shape.Values

    /// Gets the total number of elements in a possibly-symbolic shape
    member shape.nelementx = Int shape.nelement 

    /// Gets the total number of elements in a possibly-symbolic shape
    member shape.flatten() = Shape [| shape.nelement |]

    override x.Equals(y:obj) =
        match y with 
        | :? Shape as y -> values = y.ValuesRaw
        | _ -> false

    override x.GetHashCode() = hash x.Dims

    member _.GetSlice(x:int option,y:int option) =
        Shape (FSharp.Core.Operators.OperatorIntrinsics.GetArraySlice values x y)

    override x.ToString() = "[" + String.concat "," (x.Values |> Array.map string) + "]"

    /// Constraint equality
    static member (=~=) (a:Shape,b:Shape) : bool =  (a.Values = b.Values)

#endif
