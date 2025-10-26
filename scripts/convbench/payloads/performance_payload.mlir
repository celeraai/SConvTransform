!input_tensor_t = tensor<{{NI}}x{{CI}}x{{PHI}}x{{PWI}}xf32>
!weight_tensor_t = tensor<{{DO}}x{{CI}}x{{HK}}x{{WK}}xf32>
!output_tensor_t = tensor<{{NO}}x{{DO}}x{{HO}}x{{WO}}xf32>

module {
  func.func @conv_2d_nchw_fchw(%in: !input_tensor_t, %wei: !weight_tensor_t,
                              %out: !output_tensor_t) -> !output_tensor_t {
    %res = linalg.conv_2d_nchw_fchw
      {dilations = dense<[{{HD}},{{WD}}]> : tensor<2xi64>, strides = dense<[{{HS}},{{WS}}]> : tensor<2xi64> }
      ins(%in, %wei: !input_tensor_t, !weight_tensor_t)
      outs(%out: !output_tensor_t) -> !output_tensor_t
    return %res : !output_tensor_t
  }

  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Create tensors
    // TODO: Eliminate repeated values 
    %inp = tensor.generate {
    ^bb0(%n : index, %c : index, %h : index, %w : index):
      %m1 = arith.addi %w, %h : index
      %m2 = arith.addi %m1, %c : index
      %m3 = arith.addi %m2, %n : index
      %i64 = arith.index_cast %m3 : index to i64
      %f32 = arith.uitofp %i64 : i64 to f32
      tensor.yield %f32 : f32
    } : !input_tensor_t

    %wei = tensor.generate {
    ^bb0(%f : index, %c : index, %h : index, %w : index):
      %m1 = arith.addi %w, %h : index
      %m2 = arith.addi %m1, %c : index
      %m3 = arith.addi %m2, %f : index
      %i64 = arith.index_cast %m3 : index to i64
      %f32 = arith.uitofp %i64 : i64 to f32
      tensor.yield %f32 : f32
    } : !weight_tensor_t

    %out = tensor.generate {
    ^bb0(%n : index, %f : index, %h : index, %w : index):
      %m1 = arith.addi %w, %h : index
      %m2 = arith.addi %m1, %f : index
      %m3 = arith.addi %m2, %n : index
      %i64 = arith.index_cast %m3 : index to i64
      %f32 = arith.uitofp %i64 : i64 to f32
      tensor.yield %f32 : f32
    } : !output_tensor_t

    // Number of iterations
    %f0 = arith.constant 0. : f64
    %num_reps = arith.constant {{RUNS}} : index

    // Run convolution and time it
    %t_start = func.call @rtclock() : () -> f64
    %final_res = scf.for %arg0 = %c0 to %num_reps step %c1
      iter_args(%out_loop = %out) -> (!output_tensor_t) {
      %zero = arith.constant 0.0 : f32
      %zeroed_out = linalg.fill ins(%zero : f32) outs(%out_loop : !output_tensor_t) -> !output_tensor_t
      %res = func.call @conv_2d_nchw_fchw(%inp, %wei, %zeroed_out)
        : (!input_tensor_t, !weight_tensor_t, !output_tensor_t)
        -> !output_tensor_t
      scf.yield %res : !output_tensor_t
    }
    %t_end = func.call @rtclock() : () -> f64
    %t = arith.subf %t_end, %t_start : f64

    // Print the result
    %un_res = tensor.cast %final_res : !output_tensor_t to tensor<*xf32>
    func.call @printMemrefF32(%un_res) : (tensor<*xf32>) -> ()

    // num_flops_per_iter = 2 * OH * OW * F * C * KH * KW
    %num_flops_per_iter = arith.constant {{FLOPS}} : index

    // num_flops_total = num_flops_per_iter * num_reps
    %num_flops_total = arith.muli %num_flops_per_iter, %num_reps: index

    // Print the number of flops per second
    %num_flops_total_i = arith.index_cast %num_flops_total : index to i64
    %num_flops_total_f = arith.uitofp %num_flops_total_i : i64 to f64
    %flops_per_s = arith.divf %num_flops_total_f, %t : f64
    call @printFlops(%flops_per_s) : (f64) -> ()

    return
  }

  func.func private @printMemrefF32(tensor<*xf32>) attributes { llvm.emit_c_interface }
  func.func private @printFlops(f64)
  func.func private @rtclock() -> f64
}
