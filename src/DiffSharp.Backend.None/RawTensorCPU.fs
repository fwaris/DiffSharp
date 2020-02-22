namespace DiffSharp.Backend.None

open DiffSharp
open DiffSharp.Backend
open DiffSharp.Util

type RawTensorFloat32CPU(values: float32[], shape:int[]) =
    inherit RawTensor(shape, Float32, CPU, DiffSharp.Backend.Backend.None)

    member __.Values = values

    member private t.IndexToFlatIndex(index:int[]) =
        let mutable flatIndex = 0
        for i=0 to index.Length - 1 do
            let v = if i = index.Length - 1 then 1 else (Array.reduce (*) t.Shape.[i+1..])
            flatIndex <- flatIndex + index.[i] * v
        flatIndex
    
    member private t.FlatIndexToIndex(flatIndex:int) =
        let index = Array.create t.Dim 0
        let mutable mul = t.Nelement
        let mutable fi = flatIndex
        for i=t.Dim downto 1 do
            mul <- mul / t.Shape.[t.Dim-i]
            index.[i-1] <- fi / mul
            fi <- fi - index.[i-1] * mul
        index |> Array.rev

    member t.Item
        with get ([<System.ParamArray>] index:int[]) =
            if index.Length <> t.Dim then failwithf "Expecting a %id index" t.Dim
            t.Values.[t.IndexToFlatIndex(index)]
        and set ([<System.ParamArray>] index:int[]) v =
            if index.Length <> t.Dim then failwithf "Expecting a %id index" t.Dim
            t.Values.[t.IndexToFlatIndex(index)] <- v

    override t.GetItem(index:int[]) = upcast RawTensorFloat32CPU.Create(t.[index])
    
    override t.GetSlice(bounds:int[,]) =
        // if bounds.GetLength(0) <> t.Dim then failwithf "Expecting %i-by-2 bounds" t.Dim
        // printfn "%A" bounds
        let shape = Array.init (bounds.GetLength(0)) (fun i -> bounds.[i,1] - bounds.[i,0] + 1) |> shapeSqueeze -1
        // printfn "%A" shape
        let array = Array.create (shapeLength shape) 0.f
        let mutable arrayi = 0
        let rec slice (bounds:int[,]) externalCoords =
            if bounds.GetLength(0) = 1 then
                for i=bounds.[0,0] to bounds.[0,1] do
                    // printfn "inner %A" i
                    let globalCoords = Array.append externalCoords [|i|]
                    array.[arrayi] <- t.[globalCoords]
                    arrayi <- arrayi + 1
            else
                for i=bounds.[0,0] to bounds.[0,1] do
                    // printfn "outer %A" i
                    slice bounds.[1..,*] (Array.append externalCoords [|i|])
        slice bounds [||]
        upcast RawTensorFloat32CPU(array, shape)

    override t1.CompareTo(t2) =
        compare (t1.ToValue():?>float32) (t2.ToValue():?>float32)
    
    override t.Clone() = upcast RawTensorFloat32CPU(Array.copy t.Values, Array.copy t.Shape)
    override t.CreateFromScalar(value, shape) =
        let value = value:?>float32
        match shape.Length with
        | 0 -> upcast RawTensorFloat32CPU([|value|], [||])
        | _ -> upcast RawTensorFloat32CPU(Array.create (shapeLength shape) value, shape)

    override t.Create(values) = upcast RawTensorFloat32CPU.Create(values)
    override t.Zero() = upcast RawTensorFloat32CPU.Zero()
    override t.Zeros(shape) = upcast RawTensorFloat32CPU.Zeros(shape)
    override t.One() = upcast RawTensorFloat32CPU([|1.f|], [||])
    override t.Ones(shape) = upcast RawTensorFloat32CPU.Ones(shape)
    override t.Random(shape) = upcast RawTensorFloat32CPU.Random(shape)
    override t.RandomNormal(shape) = upcast RawTensorFloat32CPU.RandomNormal(shape)

    override probs.RandomMultinomial(numSamples) =
        if probs.Dim < 1 || probs.Dim > 2 then failwithf "Expecting 1d or 2d probs, received shape %A" probs.Shape
        if probs.Dim = 1 then
            let p = probs.Values |> Array.map float
            let result = [|for i=0 to numSamples-1 do yield float32 (Random.ChoiceIndex(p))|]
            upcast RawTensorFloat32CPU(result, [|numSamples|])
        else
            let p = probs.ToArray() :?> float32[,] |> Array2D.map float
            let result = Array2D.init (p.GetLength(0)) numSamples (fun i _ -> Random.ChoiceIndex(p.[i,*]))
            upcast RawTensorFloat32CPU.Create(result)

    override t.GetString() =
        // sprintf "RawTensor(Value=%A, Shape=%A, Dim=%A, Length=%A)" t.Value t.Shape t.Dim t.Length
        match t.Dim with
        | 0 -> sprintf "%A" t.Values.[0]
        | _ ->
            let sb = System.Text.StringBuilder()
            let rec print (shape:int[]) externalCoords = 
                if shape.Length = 1 then
                    sb.Append("[") |> ignore
                    let mutable prefix = ""
                    for i=0 to shape.[0]-1 do
                        let globalCoords = Array.append externalCoords [|i|]
                        sb.Append(prefix) |> ignore
                        sb.Append(sprintf "%A" (t.[globalCoords])) |> ignore
                        prefix <- "; "
                    sb.Append("]") |> ignore
                else
                    sb.Append("[") |> ignore
                    let mutable prefix = ""
                    for i=0 to shape.[0]-1 do
                        sb.Append(prefix) |> ignore
                        print shape.[1..] (Array.append externalCoords [|i|])
                        prefix <- "; "
                    sb.Append("]") |> ignore
            print t.Shape [||]
            sb.ToString()

    override t.ToValue() =
        match t.Dim with
        | 0 -> upcast t.Values.[0]
        | _ -> failwithf "Cannot convert %Ad Tensor to scalar" t.Dim

    override t.ToArray() =
        match t.Dim with
        | 0 -> failwith "Cannot convert 0d Tensor to array"
        | 1 -> upcast Array.init t.Shape.[0] (fun i -> t.[i])
        | 2 -> upcast Array2D.init t.Shape.[0] t.Shape.[1] (fun i j -> t.[i, j])
        | 3 -> upcast Array3D.init t.Shape.[0] t.Shape.[1] t.Shape.[2] (fun i j k -> t.[i, j, k])
        | 4 -> upcast Array4D.init t.Shape.[0] t.Shape.[1] t.Shape.[2] t.Shape.[3] (fun i j k l -> t.[i, j, k, l])
        | _ -> failwithf "Cannot get array for Tensor dimensions > 4. Consider slicing the Tensor. Shape: %A" t.Shape

    override t1.Equals(t2:RawTensor) = 
        match t2 with
        | :? RawTensorFloat32CPU as t2 -> t1.Shape = t2.Shape && t1.Values = t2.Values
        | _ -> failwithf "Cannot compare RawTensors of different types. t1:%A, t2:%A" t1 t2

    override t1.ApproximatelyEquals(t2:RawTensor, tolerance) =
        let tolerance = float32 <| tolerance
        match t2 with
        | :? RawTensorFloat32CPU as t2 -> t1.Shape = t2.Shape && arraysApproximatelyEqual tolerance t1.Values t2.Values
        | _ -> failwithf "Cannot compare RawTensors of different types. t1:%A, t2:%A" t1 t2

    override __.StackTs(tensors) =
        let tensors = tensors |> Seq.toList
        let values, shapes = tensors |> List.map (fun t -> (t :?> RawTensorFloat32CPU).Values, t.Shape) |> List.unzip
        if not (allEqual shapes) then failwith "Expecting Tensors with same shape"
        let n = tensors |> List.length
        let m = shapeLength shapes.[0]
        let result = Array.create (n * m) 0.f
        for i=0 to n-1 do
            for j=0 to m-1 do
                result.[i*m+j] <-values.[i].[j]
        upcast RawTensorFloat32CPU(result, Array.append [|n|] shapes.[0])

    override t.UnstackT() =
        if t.Dim < 1 then failwith "Cannot unstack scalar Tensor (dim < 1)"
        let n = t.Shape.[0]
        let unstackedShape = if t.Dim = 1 then [||] else t.Shape |> Array.skip 1
        let unstackedLength = shapeLength unstackedShape
        Array.init n (fun i -> Array.init unstackedLength (fun j -> t.Values.[i*unstackedLength+j]))
        |> Array.map (fun v -> upcast RawTensorFloat32CPU(v, unstackedShape))

    override t.TransposeT2() =
        if t.Dim <> 2 then failwith "Expecting a 2d Tensor"
        let tcols = t.Shape.[1]
        let result = Array2D.init t.Shape.[1] t.Shape.[0] (fun i j -> t.Values.[j*tcols + i])
        upcast RawTensorFloat32CPU.Create(result)

    override t.SqueezeT(dim) =
        let result = Array.copy t.Values
        upcast RawTensorFloat32CPU(result, shapeSqueeze dim t.Shape)

    override t.UnsqueezeT(dim) =
        let result = Array.copy t.Values
        upcast RawTensorFloat32CPU(result, shapeUnsqueeze dim t.Shape)

    override t.FlipT(dims:int[]) =
        if dims.Length > t.Dim then failwithf "Expecting dims (list of dimension indices to flip) of length less than Tensor's dimensions, received %A, %A" dims.Length t.Dim
        if hasDuplicates dims then failwithf "Expecting dims (list of dimension indices to flip) without repetition, received %A" dims
        if (Array.max dims) >= t.Dim then failwithf "Expecting dims (list of dimension indices to flip) where all indices are less than the tensor dimension, received %A, %A" dims t.Dim
        match t.Dim with
        | 0 -> t.Clone()
        | _ ->
            let result = RawTensorFloat32CPU.Zeros(t.Shape)
            let rec flip (shape:int[]) externalCoords = 
                if shape.Length = 1 then
                    for i=0 to shape.[0]-1 do
                        let globalCoords = Array.append externalCoords [|i|]
                        result.[mirrorCoordinates globalCoords t.Shape dims] <- t.[globalCoords]
                else
                    for i=0 to shape.[0]-1 do
                        flip shape.[1..] (Array.append externalCoords [|i|])
            flip t.Shape [||]        
            upcast result

    override t.DilateT(dilations:int[]) =
        if dilations.Length <> t.Dim then failwithf "Expecting dilations (dilation to use in each dimension) of same length with Tensor's dimensions, received %A, %A" dilations.Length t.Dim
        if (Array.min dilations) < 1 then failwithf "Expecting dilations (dilation to use in each dimension) >= 1 where 1 represents no dilation, received %A" dilations
        match t.Dim with
        | 0 -> t.Clone()
        | _ ->
            let result = RawTensorFloat32CPU.Zeros(dilatedShape t.Shape dilations)
            let rec dilate (shape:int[]) externalCoords = 
                if shape.Length = 1 then
                    for i=0 to shape.[0]-1 do
                        let globalCoords = Array.append externalCoords [|i|]
                        result.[dilatedCoordinates globalCoords dilations] <- t.[globalCoords]
                else
                    for i=0 to shape.[0]-1 do
                        dilate shape.[1..] (Array.append externalCoords [|i|])
            dilate t.Shape [||]        
            upcast result        

    override t.UndilateT(dilations:int[]) =
        match t.Dim with
        | 0 -> t.Clone()
        | _ ->
            let result = RawTensorFloat32CPU.Zeros(undilatedShape t.Shape dilations)
            let rec dilate (shape:int[]) externalCoords = 
                if shape.Length = 1 then
                    for i=0 to shape.[0]-1 do
                        let globalCoords = Array.append externalCoords [|i|]
                        result.[globalCoords] <- t.[dilatedCoordinates globalCoords dilations]
                else
                    for i=0 to shape.[0]-1 do
                        dilate shape.[1..] (Array.append externalCoords [|i|])
            dilate result.Shape [||]        
            upcast result        

    override t.ViewT(shape:int[]) =
        if shapeLength t.Shape <> shapeLength shape then failwithf "Cannot view Tensor of shape %A as shape %A" t.Shape shape
        let result = Array.copy t.Values
        upcast RawTensorFloat32CPU(result, shape)

    static member Zero() =
        let values = [|0.f|]
        RawTensorFloat32CPU(values, [||])

    static member One() =
        let values = [|1.f|]
        RawTensorFloat32CPU(values, [||])
    
    static member Zeros(shape:int[]) : RawTensorFloat32CPU =
        let values = Array.create (shapeLength shape) 0.f
        RawTensorFloat32CPU(values, shape)

    static member Ones(shape:int[]) =
        let values = Array.create (shapeLength shape) 1.f
        RawTensorFloat32CPU(values, shape)

    static member Random(shape:int[])  =
        let values = Array.init (shapeLength shape) (fun _ -> float32 (Random.Uniform()))
        RawTensorFloat32CPU(values, shape)

    static member RandomNormal(shape:int[]) =
        let values = Array.init (shapeLength shape) (fun _ -> float32 (Random.Normal()))
        RawTensorFloat32CPU(values, shape)

    static member Create(value:obj) = 
        let array, shape = value |> flatArrayAndShape<float32>
        if notNull array then 
            RawTensorFloat32CPU(array, shape)
        else 
            let array, shape = value |> flatArrayAndShape<double>
            if notNull array then 
                RawTensorFloat32CPU(array |> Array.map float32, shape)
            else
                let array, shape = value |> flatArrayAndShape<int>
                if notNull array then 
                    RawTensorFloat32CPU(array |> Array.map float32, shape)
                else
                    failwithf "Cannot convert value of type %A to RawTensorFloat32CPU" (value.GetType())

    override t1.LtTT(t2) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorFloat32CPU).Values
        let result = Array.map2 (fun t1 t2 -> if t1 < t2 then 1.f else 0.f) t1value t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.GtTT(t2) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorFloat32CPU).Values
        let result = Array.map2 (fun t1 t2 -> if t1 > t2 then 1.f else 0.f) t1value t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.LeTT(t2) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorFloat32CPU).Values
        let result = Array.map2 (fun t1 t2 -> if t1 <= t2 then 1.f else 0.f) t1value t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.GeTT(t2) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorFloat32CPU).Values
        let result = Array.map2 (fun t1 t2 -> if t1 >= t2 then 1.f else 0.f) t1value t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t.MaxIndexT() =
        t.FlatIndexToIndex(maxIndex t.Values)

    override t.MinIndexT() =
        t.FlatIndexToIndex(minIndex t.Values)

    override t1.AddTT(t2) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorFloat32CPU).Values
        let result = Array.map2 (+) t1value t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.AddTT0(t2) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorFloat32CPU).Values.[0]
        let result = Array.map ((+) t2value) t1value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.AddT2T1(t2) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorFloat32CPU).Values
        let result = Array.copy t1value
        for i=0 to t1.Shape.[0]-1 do
            for j=0 to t1.Shape.[1]-1 do
                let flatindex = i*t1.Shape.[1] + j
                result.[flatindex] <- result.[flatindex] + t2value.[j]
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.AddTTSlice(location:int[], t2) =
        if not (shapeContains t1.Shape t2.Shape) then failwithf "Expecting t1.Shape to contain t2.Shape, received %A, %A" t1.Shape t2.Shape
        let t1value = t1.Values
        let t2 = t2 :?> RawTensorFloat32CPU
        let result = Array.copy t1value
        let shape2 = shapeUnsqueezeAs t2.Shape t1.Shape
        let rec add (shape2:int[]) externalCoords =
            if shape2.Length = 1 then
                for i=0 to shape2.[0]-1 do
                    let globalCoords = Array.append externalCoords [|i|]
                    let t1Coords = Array.map2 (+) globalCoords location
                    let t1FlatIndex = t1.IndexToFlatIndex(t1Coords)
                    result.[t1FlatIndex] <- result.[t1FlatIndex] + t2.[globalCoords]
            else
                for i=0 to shape2.[0]-1 do
                    add (shape2.[1..]) (Array.append externalCoords [|i|])
        add shape2 [||]
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.SubTT(t2) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorFloat32CPU).Values
        let result = Array.map2 (-) t1value t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.SubT0T(t2) =
        let t1value = t1.Values.[0]
        let t2value = (t2 :?> RawTensorFloat32CPU).Values
        let result = Array.map ((-) t1value) t2value
        upcast RawTensorFloat32CPU(result, t2.Shape)

    override t1.SubTT0(t2) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorFloat32CPU).Values.[0]
        let result = Array.map (fun t -> t - t2value) t1value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.MulTT(t2) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorFloat32CPU).Values
        let result = Array.map2 (*) t1value t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.MulTT0(t2) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorFloat32CPU).Values.[0]
        let result = Array.map ((*) t2value) t1value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.DivTT(t2) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorFloat32CPU).Values
        let result = Array.map2 (/) t1value t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.DivT0T(t2) =
        let t1value = t1.Values.[0]
        let t2value = (t2 :?> RawTensorFloat32CPU).Values
        let result = Array.map ((/) t1value) t2value
        upcast RawTensorFloat32CPU(result, t2.Shape)

    override t1.DivTT0(t2) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorFloat32CPU).Values.[0]
        let result = Array.map (fun t -> t / t2value) t1value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.PowTT(t2) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorFloat32CPU).Values
        let result = Array.map2 ( ** ) t1value t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.PowT0T(t2) =
        let t1value = t1.Values.[0]
        let t2value = (t2 :?> RawTensorFloat32CPU).Values
        let result = Array.map (fun t -> t1value ** t) t2value
        upcast RawTensorFloat32CPU(result, t2.Shape)

    override t1.PowTT0(t2) =
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorFloat32CPU).Values.[0]
        let result = Array.map (fun t -> t ** t2value) t1value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.MatMulT2T2(t2) =
        if t1.Dim <> 2 || t2.Dim <> 2 then failwithf "Expecting two 2d Tensors, received Tensors with shapes %A, %A" t1.Shape t2.Shape
        let t1rows, t1cols = t1.Shape.[0], t1.Shape.[1]
        let t2rows, t2cols = t2.Shape.[0], t2.Shape.[1]
        if t1cols <> t2rows then failwithf "Cannot multiply Tensors with shapes %A, %A" t1.Shape t2.Shape
        let t1value = t1.Values
        let t2value = (t2 :?> RawTensorFloat32CPU).Values        
        let result = Array2D.init t1rows t2cols (fun i j -> Array.sumBy (fun k -> t1value.[i*t1cols + k] * t2value.[k*t2cols + j]) [|0..(t2rows-1)|] )
        upcast RawTensorFloat32CPU.Create(result)
    
    override t1.Conv1D(t2, stride, padding) =
        // t1: input, NxCxI (batchSize x inputChannels x inputLength)
        // t2: filters, KxCxF (outputChannels x inputChannels x kernelLength)
        if t1.Dim <> 3 || t2.Dim <> 3 then failwithf "Expecting two 3d Tensors t1, t2 where t1 is input (NxCxI: batchSize x inputChannels x inputLength) and t2 is filters (KxCxF: outputChannels x inputChannels x kernelLength), received Tensors with shapes %A, %A" t1.Shape t2.Shape
        let t1 =
            if padding = 0 then
                t1
            elif padding > 0 then
                let tshape = Array.copy t1.Shape
                tshape.[2] <- t1.Shape.[2] + padding * 2
                let t = RawTensorFloat32CPU.Zeros(tshape)
                t.AddTTSlice([|0; 0; padding|], t1) :?> RawTensorFloat32CPU
            else
                failwithf "Expecting padding >= 0, received %A" padding
        let batchSize = t1.Shape.[0]
        let inputChannels = t1.Shape.[1]
        let inputLength = t1.Shape.[2]
        let outputChannels = t2.Shape.[0]
        if t2.Shape.[1] <> inputChannels then failwithf "Input and filters have different number of channels: %A, %A" inputChannels t2.Shape.[1]
        let kernelLength = t2.Shape.[2]
        if kernelLength > inputLength then failwithf "Expecting kernelLength <= inputLength, received %A, %A" kernelLength inputLength
        let outputLength = inputLength - kernelLength + 1
        let outputShape = [|batchSize; outputChannels; outputLength|]
        let result = RawTensorFloat32CPU.Zeros(outputShape)
        let t2 = t2 :?> RawTensorFloat32CPU
        for n=0 to batchSize-1 do
            for k=0 to outputChannels-1 do
                for v=0 to outputLength-1 do
                    let mutable value = 0.f
                    for c=0 to inputChannels-1 do
                        for u=0 to kernelLength-1 do
                            value <- value + t2.[k, c, u] * t1.[n, c, v + u]
                    result.[[|n; k; v|]] <- value
        if stride = 1 then
            result :> RawTensor
        elif stride > 1 then
            let outputLength = (float outputLength) / (float stride) |> ceil |> int
            let outputShape = [|batchSize; outputChannels; outputLength|]
            let mutable sresult = RawTensorFloat32CPU.Zeros(outputShape)
            for v=0 to outputLength-1 do
                let sliceBounds = array2D [[0; batchSize-1]; [0; outputChannels-1]; [v * stride; v * stride]]
                let slice = result.GetSlice(sliceBounds).ViewT([|batchSize; outputChannels; 1|])
                sresult <- sresult.AddTTSlice([|0; 0; v|], slice) :?> RawTensorFloat32CPU
            sresult :> RawTensor
        else
            failwithf "Expecting stride >= 1, received %A" stride

    override t1.Conv2D(t2, stride, padding) =
        // t1: input, NxCxHxW (batchSize x inputChannels x inputHeight x inputWidth)
        // t2: filters, KxCxFxG (outputChannels x inputChannels x kernelHeight x kernelWidth)
        if t1.Dim <> 4 || t2.Dim <> 4 then failwithf "Expecting two 4d Tensors t1, t2 where t1 is input, NxCxHxW (batchSize x inputChannels x inputHeight x inputWidth) and t2 is filters, KxCxFxG (outputChannels x inputChannels x kernelHeight x kernelWidth), received Tensors with shapes %A, %A" t1.Shape t2.Shape
        if stride.Length <> 2 then failwithf "Expecting stride to be a length-two array, received %A" stride
        if padding.Length <> 2 then failwithf "Expecting padding to be a length-two array, received %A" padding
        let t1 =
            if padding.[0] = 0 && padding.[1] = 0 then
                t1
            elif padding.[0] >= 0 && padding.[1] >= 0 then
                let tshape = Array.copy t1.Shape
                tshape.[2] <- t1.Shape.[2] + padding.[0] * 2
                tshape.[3] <- t1.Shape.[3] + padding.[1] * 2
                let t = RawTensorFloat32CPU.Zeros(tshape)
                t.AddTTSlice([|0; 0; padding.[0]; padding.[1]|], t1) :?> RawTensorFloat32CPU
            else
                failwithf "Expecting all paddings >= 0, received %A" padding
        let batchSize = t1.Shape.[0]
        let inputChannels = t1.Shape.[1]
        let inputHeight = t1.Shape.[2]
        let inputWidth = t1.Shape.[3]
        let outputChannels = t2.Shape.[0]
        if t2.Shape.[1] <> inputChannels then failwithf "Input and filters have different number of channels: %A, %A" inputChannels t2.Shape.[1]
        let kernelHeight = t2.Shape.[2]
        let kernelWidth = t2.Shape.[3]
        if kernelHeight > inputHeight then failwithf "Expecting kernelHeight <= inputHeight, received %A, %A" kernelHeight inputHeight
        if kernelWidth > inputWidth then failwithf "Expecting kernelWidth <= inputWidth, received %A, %A" kernelWidth inputWidth
        let outputHeight = inputHeight - kernelHeight + 1
        let outputWidth = inputWidth - kernelWidth + 1
        let outputShape = [|batchSize; outputChannels; outputHeight; outputWidth|]
        let result = RawTensorFloat32CPU.Zeros(outputShape)
        let t2 = t2 :?> RawTensorFloat32CPU
        for n=0 to batchSize-1 do
            for k=0 to outputChannels-1 do
                for v0=0 to outputHeight-1 do
                    for v1=0 to outputWidth-1 do
                        let mutable value = 0.f
                        for c=0 to inputChannels-1 do
                            for u0=0 to kernelHeight-1 do
                                for u1=0 to kernelWidth-1 do
                                    value <- value + t2.[k, c, u0, u1] * t1.[n, c, v0+u0, v1+u1]
                        result.[[|n; k; v0; v1|]] <- value
        if stride.[0] = 1 && stride.[1] = 1 then
            result :> RawTensor
        elif stride.[0] >= 1 && stride.[1] >= 1 then
            let outputHeight = (float outputHeight) / (float stride.[0]) |> ceil |> int
            let outputWidth = (float outputWidth) / (float stride.[1]) |> ceil |> int
            let outputShape = [|batchSize; outputChannels; outputHeight; outputWidth|]
            let mutable sresult = RawTensorFloat32CPU.Zeros(outputShape)
            for v0=0 to outputHeight-1 do
                for v1=0 to outputWidth-1 do
                    let sliceBounds = array2D [[0; batchSize-1]; [0; outputChannels-1]; [v0 * stride.[0]; v0 * stride.[0]]; [v1 * stride.[1]; v1 * stride.[1]]]
                    let slice = result.GetSlice(sliceBounds).ViewT([|batchSize; outputChannels; 1; 1|])
                    sresult <- sresult.AddTTSlice([|0; 0; v0; v1|], slice) :?> RawTensorFloat32CPU
            sresult :> RawTensor
        else
            failwithf "Expecting all strides >= 1, received %A" stride


    override t.NegT() =
        let result = Array.map (~-) t.Values
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.SumT() =
        let result = Array.reduce (+) t.Values
        upcast RawTensorFloat32CPU([|result|], [||])
    
    override t.SumT2Dim0() =
        if t.Dim <> 2 then failwith "Expecting a 2d Tensor"
        let result = Array.init t.Shape.[1] (fun j -> Array.init t.Shape.[0] (fun i -> t.Values.[i * t.Shape.[1] + j]) |> Array.reduce (+))
        let resultShape = [|t.Shape.[1]|]
        upcast RawTensorFloat32CPU(result, resultShape)

    override t.SignT() =
        let result = t.Values |> Array.map (sign >> float32)
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.FloorT() =
        let result = t.Values |> Array.map floor
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.CeilT() =
        let result = t.Values |> Array.map ceil
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.RoundT() =
        let result = t.Values |> Array.map round
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.AbsT() =
        let result = t.Values |> Array.map abs
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.ReluT() =
        let result = t.Values |> Array.map (max 0.f) 
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.SigmoidT() =
        let result = t.Values |> Array.map (fun v -> 1.f / (1.f + exp -v))
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.ExpT() =
        let result = t.Values |> Array.map exp
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.LogT() =
        let result = t.Values |> Array.map log
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.Log10T() =
        let result = t.Values |> Array.map log10
        upcast RawTensorFloat32CPU(result, t.Shape)
        
    override t.SqrtT() =
        let result = t.Values |> Array.map sqrt
        upcast RawTensorFloat32CPU(result, t.Shape)
        
    override t.SinT() =
        let result = t.Values |> Array.map sin
        upcast RawTensorFloat32CPU(result, t.Shape)
        
    override t.CosT() =
        let result = t.Values |> Array.map cos
        upcast RawTensorFloat32CPU(result, t.Shape)                
        
    override t.TanT() =
        let result = t.Values |> Array.map tan
        upcast RawTensorFloat32CPU(result, t.Shape)
        
    override t.SinhT() =
        let result = t.Values |> Array.map sinh
        upcast RawTensorFloat32CPU(result, t.Shape)
        
    override t.CoshT() =
        let result = t.Values |> Array.map cosh
        upcast RawTensorFloat32CPU(result, t.Shape)                
        
    override t.TanhT() =
        let result = t.Values |> Array.map tanh
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.AsinT() =
        let result = t.Values |> Array.map asin
        upcast RawTensorFloat32CPU(result, t.Shape)
        
    override t.AcosT() =
        let result = t.Values |> Array.map acos
        upcast RawTensorFloat32CPU(result, t.Shape)                
        
    override t.AtanT() =
        let result = t.Values |> Array.map atan
        upcast RawTensorFloat32CPU(result, t.Shape)
        
and RawTensorFloat32CPUStatics() = 

    inherit RawTensorStatics()

    override __.Zero = upcast RawTensorFloat32CPU.Zero()
    override __.One = upcast RawTensorFloat32CPU.One()
    override __.Zeros(shape:int[]) = upcast RawTensorFloat32CPU.Zeros(shape)
    override __.Ones(shape:int[]) = upcast RawTensorFloat32CPU.Ones(shape)
    override __.Random(shape:int[]) = upcast RawTensorFloat32CPU.Random(shape)
    override __.RandomNormal(shape:int[]) = upcast RawTensorFloat32CPU.RandomNormal(shape)
    override __.Create(values:obj) : RawTensor = upcast RawTensorFloat32CPU.Create(values)

 