namespace DiffSharp

[<AutoOpen>]
module AvgPoolExtensions =

    type Tensor with
        /// <summary>Applies a 1D average pooling over an input signal composed of several input planes, returning the max indices along with the outputs.</summary>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="ceil_mode">TBD.</param>
        /// <param name="count_include_pad">TBD.</param>
        member a.avgpool1d(kernelSize:int, ?stride:int, ?padding:int, ?ceil_mode: bool, ?count_include_pad: bool) =
            let stride = defaultArg stride kernelSize
            let padding = defaultArg padding 0
            let ceil_mode = defaultArg ceil_mode false
            let count_include_pad= defaultArg count_include_pad true
            Tensor.Op
                { new UnaryOp() with 
                    member _.ComputeRaw(a) = a.AvgPool1D(kernelSize, stride, padding, ceil_mode, count_include_pad)
                    member _.Forward(fab, a, ad) = ad.avgpool1d(kernelSize, stride, padding, ceil_mode, count_include_pad)
                    member _.Reverse(a, td) = td.avgpoolReverse1d(a, kernelSize, stride, padding, ceil_mode, count_include_pad)
                 }
                 a

        /// <summary>Computes a partial inverse of avgpool1d</summary>
        /// <param name="originalInput">The original input to avgpool1d, used for size information.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="ceil_mode">TBD.</param>
        /// <param name="count_include_pad">TBD.</param>
        member a.avgpoolReverse1d(originalInput:Tensor, kernelSize:int, stride:int, padding:int, ceil_mode: bool, count_include_pad: bool) =
            Tensor.Op
                { new UnaryOp() with 
                    member _.ComputeRaw(a) = a.AvgPoolReverse1D(originalInput.primalRaw, kernelSize, stride, padding, ceil_mode, count_include_pad)
                    member _.Forward(fab, a, ad) = ad.avgpoolReverse1d(originalInput, kernelSize, stride, padding, ceil_mode, count_include_pad)
                    member _.Reverse(a, td) = td.avgpool1d(kernelSize, stride, padding, ceil_mode, count_include_pad)
                 }
                 a

    type Tensor with
        /// <summary>Applies a 1D average pooling over an input signal composed of several input planes, returning the max indices along with the outputs.</summary>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="ceil_mode">TBD.</param>
        /// <param name="count_include_pad">TBD.</param>
        member a.avgpool2d(?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>, ?kernelSize:int, ?stride:int, ?padding:int, ?ceil_mode: bool, ?count_include_pad: bool) =
            let stride = defaultArg stride kernelSize
            let padding = defaultArg padding 0
            let ceil_mode = defaultArg ceil_mode false
            let count_include_pad= defaultArg count_include_pad true
            Tensor.Op
                { new UnaryOp() with 
                    member _.ComputeRaw(a) = a.AvgPool2D(kernelSize, stride, padding, ceil_mode, count_include_pad)
                    member _.Forward(fab, a, ad) = ad.avgpool2d(kernelSize, stride, padding, ceil_mode, count_include_pad)
                    member _.Reverse(a, td) = td.avgpoolReverse2d(a, kernelSize, stride, padding, ceil_mode, count_include_pad)
                 }
                 a

        /// <summary>Computes a partial inverse of avgpool1d</summary>
        /// <param name="originalInput">The original input to avgpool1d, used for size information.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="ceil_mode">TBD.</param>
        /// <param name="count_include_pad">TBD.</param>
        member a.avgpoolReverse2d(originalInput:Tensor, kernelSize:int, stride:int, padding:int, ceil_mode: bool, count_include_pad: bool) =
            Tensor.Op
                { new UnaryOp() with 
                    member _.ComputeRaw(a) = a.AvgPoolReverse2D(originalInput.primalRaw, kernelSize, stride, padding, ceil_mode, count_include_pad)
                    member _.Forward(fab, a, ad) = ad.avgpoolReverse2d(originalInput, kernelSize, stride, padding, ceil_mode, count_include_pad)
                    member _.Reverse(a, td) = td.avgpool2d(kernelSize, stride, padding, ceil_mode, count_include_pad)
                 }
                 a
