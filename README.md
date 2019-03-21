## References
[Swizzle Inventor: Data Movement Synthesis for GPU Kernels, ASPLOS 2019](https://mangpo.net/papers/swizzle-inventor-asplos19.pdf)

## License
Refer to [LICENSE](LICENSE) for the license and copyright information for this project.

## Software Prerequisites
* [Racket](https://racket-lang.org/download/)
* [Rosette 2.x](https://github.com/emina/rosette/releases/tag/2.2). Note: Swizzle Inventor has not been tested with Rosette 3.x.

## Running Synthesizer

#### 1D stencil
ex2-stencil.rkt

#### 1D conv
ex2-conv1d.rkt

#### 2D conv
ex2-stencil2d.rkt

#### Finite field multiplication
ex3-poly-mult.rkt

#### AoS-load-sum
ex4-aos-sum.rkt

#### AoS-load-store
ex5-aos-pure-load.rkt

## Racket to CUDA Code Generator
codegen-test.rkt
