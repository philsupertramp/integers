mod lib;

use lib::{Tensor, kernels, XorShift64};


fn main() {
    let t1: Tensor<i8> = Tensor::new([1, 2].to_vec());
    let t2: Tensor<i8> = Tensor::from_vec([1,2].to_vec(), [2, 1].to_vec());
    println!("Hello, world!");
    println!("{:?}", &t1);
    println!("{:?}", &t2);
    println!("{} == {}", t2.len(), t1.len());

}
