#|
 | Copyright (c) 2018-2019, University of California, Berkeley.
 | Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
 |
 | Redistribution and use in source and binary forms, with or without 
 | modification, are permitted provided that the following conditions are met:
 |
 | 1. Redistributions of source code must retain the above copyright notice, 
 | this list of conditions and the following disclaimer.
 |
 | 2. Redistributions in binary form must reproduce the above copyright notice, 
 | this list of conditions and the following disclaimer in the documentation 
 | and/or other materials provided with the distribution.
 |
 | THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 | AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 | IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 | ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
 | LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 | CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 | SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 | INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 | CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 | ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 | POSSIBILITY OF SUCH DAMAGE.
 |#

#lang rosette

(require rosette/lib/synthax)
(require "util.rkt" "cuda.rkt" "cuda-synth.rkt")

(define n-block 1)
(define block-dim-y 1)

;; Create input and output matrices.
(define (create-IO warpSize n)
  (set-warpSize warpSize)
  (define A (create-matrix (x-y-z n n-block) gen-uid))
  (define B (create-matrix (x-y-z n n-block) gen-uid))
  (define C (create-matrix (x-y-z (* 2 n) n-block)))
  (define C* (create-matrix (x-y-z (* 2 n) n-block)))
  (values A B C C*))

;; Run sequential program spec and GPU kernel kernel, and compare their outputs.
;; n is the degree of polynomial multiplication.
(define (run-with-warp-size spec kernel w n)
  (define-values (A B C C*)
    (create-IO w n))

  (spec A B C n n-block)
  (run-kernel kernel (x-y-z w block-dim-y) (x-y-z 1 n-block) A B C* n)
  ;(pretty-display ">>> C")
  ;(acc-print C)
  ;(pretty-display ">>> C*")
  ;(acc-print C*)
  (acc-equal? C C*))

;; Sequential program spec
(define (mult-spec A B C n rows)
  (for ([row rows])
    (for ([index n])
      (let ([c (create-accumulator (list bvand bvxor) identity)])
        (for ([i (add1 index)])
          (let ([a (get A i row)]
                [b (get B (- index i) row)])
            (accumulate c (list a b))))
        (set C index row c))
      (let ([d (create-accumulator (list bvand bvxor) identity)])
        (for ([i (range (add1 index) n)])
          (let ([a (get A i row)]
                [b (get B (- (+ index n) i) row)])
            (accumulate d (list a b))))
        (set C (+ n index) row d)))))

;; Complete kernel for poly-mult degree 32
(define (mult threadId blockID blockDim A B C n)
  (define block-offset (* blockID blockDim))
  (define globalID (+ threadId block-offset))
  (define a-cached 0)
  (define b-cached 0)
  (global-to-reg A a-cached globalID)
  (global-to-reg B b-cached globalID)
  
  (define tidx (get-x threadId))
  (define acc1 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc2 (create-accumulator (list bvand bvxor) identity blockDim))

  (for ([i n])
    (let ([a (shfl a-cached i)]
          [b (shfl b-cached (- tidx i))])
      (accumulate acc1 (list a b) #:pred (<= i tidx))))
  
  (for ([i n])
    (let ([a (shfl a-cached i)]
          [b (shfl b-cached (- tidx i))])
      (accumulate acc2 (list a b) #:pred (> i tidx))))

  (reg-to-global acc1 C (+ block-offset threadId))
  (reg-to-global acc2 C (+ block-offset (@dup (x-y-z n 0)) threadId))
  )

;; Complete kernel for poly-mult degree 32 using sw-xform function 
(define (mult32 threadId blockID blockDim A B C n)
  (define globalID (+ threadId (* blockID blockDim)))
  (define a-cached 0)
  (define b-cached 0)
  (global-to-reg A a-cached globalID #:size (x-y-z n))
  (global-to-reg B b-cached globalID #:size (x-y-z n))
  
  (define tidx (modulo (get-x threadId) 32))
  (define acc1 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc2 (create-accumulator (list bvand bvxor) identity blockDim))

  (for ([i n])
    (let* ([lane-a (sw-xform tidx warpSize 0 warpSize warpSize 1
                        i warpSize 1 warpSize)]
           [lane-b (sw-xform tidx warpSize 1 warpSize warpSize 1
                        i warpSize (- warpSize 1) warpSize)]
           [a (shfl a-cached lane-a)]
           [b (shfl b-cached lane-b)]
          )
      (accumulate acc1 (list a b) #:pred (<= (@dup i) tidx))
      (accumulate acc2 (list a b) #:pred (> (@dup i) tidx))
      ))

  (reg-to-global acc1 C globalID #:size (x-y-z (* 2 n)))
  (reg-to-global acc2 C (+ globalID (@dup (x-y-z n 0))) #:size (x-y-z (* 2 n)))
  )

;; Kernel sketch for degree 32 using registers
(define (mult32-sketch threadId blockID blockDim A B C n)
  ;; For 2D kernel like this, threadId, blockID, and blockDim contain two values: .x and .y.
  ;; (* blockID blockDim) = (x-y-z (* blockID.x blockDim.x) (* blockID.y blockDim.y))
  ;; x-y-z is for creating a tuple of values
  (define globalID (+ threadId (* blockID blockDim)))
  (define a-cached 0)
  (define b-cached 0)
  (global-to-reg A a-cached globalID #:size (x-y-z n))
  (global-to-reg B b-cached globalID #:size (x-y-z n))
  
  (define tidx (modulo (get-x threadId) 32)) ;; threadId.x % 32
  (define acc1 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc2 (create-accumulator (list bvand bvxor) identity blockDim))

  (for ([i n])
    (let* ([lane-a (?sw-xform-easy tidx warpSize i warpSize [])]
           [lane-b (?sw-xform-easy tidx warpSize i warpSize [])]
           ;[lane-a (?lane-mod tidx (@dup i) 2 n [warpSize])]
           ;[lane-b (?lane-mod tidx (@dup i) 2 n [warpSize])]
           [a (shfl a-cached lane-a)]
           [b (shfl b-cached lane-b)]
          )
      (accumulate acc1 (list a b) #:pred (?cond-easy tidx (@dup i)))
      (accumulate acc2 (list a b) #:pred (?cond-easy tidx (@dup i)))
      ))

  (reg-to-global acc1 C globalID #:size (x-y-z (* 2 n)))
  (reg-to-global acc2 C (+ globalID (@dup (x-y-z n 0))) #:size (x-y-z (* 2 n)))
  )

;; Kernel sketch for degree 32 using shared memory
(define (mult32-shared-sketch threadId blockID blockDim A B C n)
  ;; For 2D kernel like this, threadId, blockID, and blockDim contain two values: .x and .y.
  ;; (* blockID blockDim) = (x-y-z (* blockID.x blockDim.x) (* blockID.y blockDim.y))
  ;; x-y-z is for creating a tuple of values
  (define globalID (+ threadId (* blockID blockDim)))
  (define-shared a-cached (create-matrix blockDim))
  (define-shared b-cached (create-matrix blockDim))
  (global-to-shared A a-cached
                    (x-y-z 1 1) ;; stride
                    (* blockDim blockID)
                    blockDim
                    #f #:round (x-y-z 1 1) #:size (x-y-z n 1))
  (global-to-shared B b-cached
                    (x-y-z 1 1) ;; stride
                    (* blockDim blockID)
                    blockDim
                    #f #:round (x-y-z 1 1) #:size (x-y-z n 1))
  
  (define tidx (modulo (get-x threadId) 32)) ;; threadId.x % 32
  (define tidy (get-y threadId))
  (define acc1 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc2 (create-accumulator (list bvand bvxor) identity blockDim))

  (for ([i n])
    (let* ([lane-a (?sw-xform-easy tidx warpSize i warpSize [])]
           [lane-b (?sw-xform-easy tidx warpSize i warpSize [])]
           ;[lane-a (?lane-mod tidx (@dup i) 2 n [warpSize])]
           ;[lane-b (?lane-mod tidx (@dup i) 2 n [warpSize])]
           [a (get a-cached lane-a tidy)]
           [b (get b-cached lane-b tidy)]
          )
      (accumulate acc1 (list a b) #:pred (?cond-easy tidx (@dup i)))
      (accumulate acc2 (list a b) #:pred (?cond-easy tidx (@dup i)))
      ))

  (reg-to-global acc1 C globalID #:size (x-y-z (* 2 n)))
  (reg-to-global acc2 C (+ globalID (@dup (x-y-z n 0))) #:size (x-y-z (* 2 n)))
  )

;; Complete kernel for degree 64 using registers
(define (mult64 threadId blockID blockDim A B C n)
  (define globalID (+ threadId (* blockID blockDim)))
  (define a-cached (create-matrix-local (x-y-z 2 1)))
  (define b-cached (create-matrix-local (x-y-z 2 1)))
  (global-to-local A a-cached
                   (x-y-z 1 1) ;; stride
                   (* (quotient globalID (x-y-z warpSize 1))
                      (x-y-z n 1))
                   (x-y-z n 1)
                   #f #:warp-shape (x-y-z warpSize 1) #:round (x-y-z 2 1))
  (global-to-local B b-cached
                   (x-y-z 1 1) ;; stride
                   (* (quotient globalID (x-y-z warpSize 1))
                      (x-y-z n 1))
                   (x-y-z n 1)
                   #f #:warp-shape (x-y-z warpSize 1) #:round (x-y-z 2 1))
  
  (define tidx (get-idInWarp threadId))
  (define acc1 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc2 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc3 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc4 (create-accumulator (list bvand bvxor) identity blockDim))

  (for ([i warpSize])
    (let* ([lane-a1 (sw-xform tidx warpSize 0 warpSize warpSize 1
                        i warpSize 1 warpSize)]
           [lane-a2 (sw-xform tidx warpSize 0 warpSize warpSize 1
                        i warpSize 1 warpSize)]
           [lane-b1 (sw-xform tidx warpSize 1 warpSize warpSize 1
                        i warpSize (- warpSize 1) warpSize)]
           [lane-b2 (sw-xform tidx warpSize 1 warpSize warpSize 1
                        i warpSize (- warpSize 1) warpSize)]
           [a1 (shfl (get a-cached (@dup 0) (@dup 0)) lane-a1)]
           [a2 (shfl (get a-cached (@dup 1) (@dup 0)) lane-a2)]
           [b1 (shfl (get b-cached (@dup 0) (@dup 0)) lane-b1)]
           [b2 (shfl (get b-cached (@dup 1) (@dup 0)) lane-b2)]
          )
      (accumulate acc1 (list a1 b1) #:pred (<= i tidx))
      
      (accumulate acc2 (list a1 b1) #:pred (> i tidx))
      (accumulate acc2 (list a1 b2) #:pred (<= i tidx))
      (accumulate acc2 (list a2 b1) #:pred (<= i tidx))
      
      (accumulate acc3 (list a1 b2) #:pred (> i tidx))
      (accumulate acc3 (list a2 b1) #:pred (> i tidx))
      (accumulate acc3 (list a2 b2) #:pred (<= i tidx))
      
      (accumulate acc4 (list a2 b2) #:pred (> i tidx))
      ))

  (reg-to-global acc1 C globalID)
  (reg-to-global acc2 C (+ globalID (@dup (x-y-z warpSize 0))))
  (reg-to-global acc3 C (+ globalID (@dup (x-y-z (* 2 warpSize) 0))))
  (reg-to-global acc4 C (+ globalID (@dup (x-y-z (* 3 warpSize) 0))))
  )

;; Kernel sketch for degree 64 using registers
(define (mult64-sketch threadId blockID blockDim A B C n)
  (define globalID (+ threadId (* blockID blockDim)))
  (define a-cached (create-matrix-local (x-y-z 2 1)))
  (define b-cached (create-matrix-local (x-y-z 2 1)))
  (global-to-local A a-cached
                   (x-y-z 1 1) ;; stride
                   (* (quotient globalID (x-y-z warpSize 1))
                      (x-y-z n 1))
                   (x-y-z n 1)
                   #f #:warp-shape (x-y-z warpSize 1) #:round (x-y-z 2 1))
  (global-to-local B b-cached
                   (x-y-z 1 1) ;; stride
                   (* (quotient globalID (x-y-z warpSize 1))
                      (x-y-z n 1))
                   (x-y-z n 1)
                   #f #:warp-shape (x-y-z warpSize 1) #:round (x-y-z 2 1))
  
  (define tidx (get-idInWarp threadId))
  (define acc1 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc2 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc3 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc4 (create-accumulator (list bvand bvxor) identity blockDim))

  (for ([i warpSize])
    (let* ([lane-a1 (?sw-xform-easy tidx warpSize
                          i warpSize [])]
           [lane-a2 (?sw-xform-easy tidx warpSize
                          i warpSize [])]
           [lane-b1 (?sw-xform-easy tidx warpSize
                          i warpSize [])]
           [lane-b2 (?sw-xform-easy tidx warpSize
                          i warpSize [])]
           [idx-a1 (ite (?cond tidx (@dup i)) (@dup 0) (@dup 1))]
           [idx-a2 (ite (?cond tidx (@dup i)) (@dup 0) (@dup 1))]
           [idx-b1 (ite (?cond tidx (@dup i)) (@dup 0) (@dup 1))]
           [idx-b2 (ite (?cond tidx (@dup i)) (@dup 0) (@dup 1))]
           [a1 (shfl (get a-cached idx-a1 (@dup 0)) lane-a1)]
           [a2 (shfl (get a-cached idx-a2 (@dup 0)) lane-a2)]
           [b1 (shfl (get b-cached idx-b1 (@dup 0)) lane-b1)]
           [b2 (shfl (get b-cached idx-b2 (@dup 0)) lane-b2)]
          )
      (accumulate acc1 (list a1 b1) #:pred (?cond tidx (@dup i)))
      (accumulate acc2 (list a1 b1) #:pred (?cond tidx (@dup i)))
      (accumulate acc3 (list a1 b1) #:pred (?cond tidx (@dup i)))
      (accumulate acc4 (list a1 b1) #:pred (?cond tidx (@dup i)))
      
      (accumulate acc1 (list a1 b2) #:pred (?cond tidx (@dup i)))
      (accumulate acc2 (list a1 b2) #:pred (?cond tidx (@dup i)))
      (accumulate acc3 (list a1 b2) #:pred (?cond tidx (@dup i)))
      (accumulate acc4 (list a1 b2) #:pred (?cond tidx (@dup i)))
      
      (accumulate acc1 (list a2 b1) #:pred (?cond tidx (@dup i)))
      (accumulate acc2 (list a2 b1) #:pred (?cond tidx (@dup i)))
      (accumulate acc3 (list a2 b1) #:pred (?cond tidx (@dup i)))
      (accumulate acc4 (list a2 b1) #:pred (?cond tidx (@dup i)))
      
      (accumulate acc1 (list a2 b2) #:pred (?cond tidx (@dup i)))
      (accumulate acc2 (list a2 b2) #:pred (?cond tidx (@dup i)))
      (accumulate acc3 (list a2 b2) #:pred (?cond tidx (@dup i)))
      (accumulate acc4 (list a2 b2) #:pred (?cond tidx (@dup i)))
      ))

  (reg-to-global acc1 C globalID)
  (reg-to-global acc2 C (+ globalID (@dup (x-y-z warpSize 0))))
  (reg-to-global acc3 C (+ globalID (@dup (x-y-z (* 2 warpSize) 0))))
  (reg-to-global acc4 C (+ globalID (@dup (x-y-z (* 3 warpSize) 0))))
  )

;; Complete sketch for degree 32 using shared memory
(define (mult32-shared threadId blockID blockDim A B C n)
  (define warpId (get-warpId threadId))
  (define-shared a-cached (create-matrix (x-y-z warpSize block-dim-y)))
  (define-shared b-cached (create-matrix (x-y-z warpSize block-dim-y)))
  (define block-offset (* blockID blockDim))
  (global-to-shared A a-cached
                    (x-y-z 1 1) ;; stride
                    block-offset
                    blockDim #:size warpSize)
  (global-to-shared B b-cached
                    (x-y-z 1 1) ;; stride
                    block-offset
                    blockDim #:size warpSize)
  
  (define tidx (get-x threadId))
  (define tidy (get-y threadId))
  (define acc1 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc2 (create-accumulator (list bvand bvxor) identity blockDim))

  (for ([i n])
    (let* ([lane-a (sw-xform tidx warpSize 0 warpSize warpSize 1
                        i warpSize 1 warpSize)]
           [lane-b (sw-xform tidx warpSize 1 warpSize warpSize 1
                        i warpSize -1 warpSize)]
           #;[lane-a (?sw-xform tidx warpSize
                         i warpSize #:fw 1)]
           #;[lane-b (?sw-xform tidx warpSize
                         i warpSize #:fw 1)]
           [a (get a-cached lane-a tidy)]
           [b (get b-cached lane-b tidy)]
          )
      (accumulate acc1 (list a b) #:pred (<= i tidx) #;(?cond tidx (@dup i)))
      (accumulate acc2 (list a b) #:pred (> i tidx) #;(?cond tidx (@dup i)))
      ))

  (reg-to-global acc1 C (+ block-offset threadId))
  (reg-to-global acc2 C (+ block-offset (@dup (x-y-z n 0)) threadId))
  )

;; Complete sketch for degree 64 using shared memory
(define (mult64-shared threadId blockID blockDim A B C n)
  (define warpId (get-warpId threadId))
  (define-shared a-cached (create-matrix (* (x-y-z 2 1) blockDim)))
  (define-shared b-cached (create-matrix (* (x-y-z 2 1) blockDim)))
  (define block-offset (* (x-y-z 2 1) blockID blockDim))
  (global-to-shared A a-cached
                    (x-y-z 1 1) ;; stride
                    block-offset
                    (* (x-y-z 2 1) blockDim))
  (global-to-shared B b-cached
                    (x-y-z 1 1) ;; stride
                    block-offset
                    (* (x-y-z 2 1) blockDim))

  (pretty-display `(a ,a-cached))
  
  (define tidx (modulo (get-x threadId) 32))
  (define tidy (get-y threadId))
  (define acc1 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc2 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc3 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc4 (create-accumulator (list bvand bvxor) identity blockDim))

  (for ([i warpSize])
    (let* ([lane-a1 (sw-xform tidx n 0 n n 1
                         i warpSize 1 warpSize 0)]
           [lane-a2 (sw-xform tidx n 0 n n 1
                         i warpSize 1 warpSize warpSize)]
           [lane-b1 (sw-xform tidx n 1 n n 1
                         i warpSize -1 warpSize 0)]
           [lane-b2 (sw-xform tidx n 1 n n 1
                         i warpSize -1 warpSize warpSize)]
           [a1 (get a-cached lane-a1 tidy)]
           [a2 (get a-cached lane-a2 tidy)]
           [b1 (get b-cached lane-b1 tidy)]
           [b2 (get b-cached lane-b2 tidy)]
          )
      (accumulate acc1 (list a1 b1) #:pred (<= i tidx))
      (accumulate acc3 (list a1 b1) #:pred (> i tidx))
      
      (accumulate acc2 (list a1 b2) #:pred #t)
      
      (accumulate acc2 (list a2 b1) #:pred (<= i tidx))
      (accumulate acc4 (list a2 b1) #:pred (> i tidx))
      
      (accumulate acc3 (list a2 b2) #:pred #t)
      ))

  (reg-to-global acc1 C (+ block-offset threadId))
  (reg-to-global acc2 C (+ block-offset (x-y-z warpSize 0) threadId))
  (reg-to-global acc3 C (+ block-offset (x-y-z (* 2 warpSize) 0) threadId))
  (reg-to-global acc4 C (+ block-offset (x-y-z (* 3 warpSize) 0) threadId))
  )

;; Kernel sketch for degree 64 using shared memory
(define (mult64-shared-sketch threadId blockID blockDim A B C n)
  (define warpId (get-warpId threadId))
  (define-shared a-cached (create-matrix (* (x-y-z 2 1) blockDim)))
  (define-shared b-cached (create-matrix (* (x-y-z 2 1) blockDim)))
  (define block-offset (* (x-y-z 2 1) blockID blockDim))
  (global-to-shared A a-cached
                    (x-y-z 1 1) ;; stride
                    block-offset
                    (* (x-y-z 2 1) blockDim) #:round (x-y-z 2 1) #:size (x-y-z n))
  (global-to-shared B b-cached
                    (x-y-z 1 1) ;; stride
                    block-offset
                    (* (x-y-z 2 1) blockDim) #:round (x-y-z 2 1) #:size (x-y-z n))

  (pretty-display `(a ,a-cached))
  
  (define tidx (get-idInWarp threadId))
  (define tidy (get-y threadId))
  (define acc1 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc2 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc3 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc4 (create-accumulator (list bvand bvxor) identity blockDim))

  (for ([i warpSize])
    (let* ([lane-a1 (?sw-xform-easy tidx n
                          i warpSize [])]
           [lane-a2 (?sw-xform-easy tidx n
                          i warpSize [])]
           [lane-b1 (?sw-xform-easy tidx n
                          i warpSize [])]
           [lane-b2 (?sw-xform-easy tidx n
                          i warpSize [])]
           [a1 (get a-cached lane-a1 tidy)]
           [a2 (get a-cached lane-a2 tidy)]
           [b1 (get b-cached lane-b1 tidy)]
           [b2 (get b-cached lane-b2 tidy)]
          )
      (accumulate acc1 (list a1 b1) #:pred (?cond tidx (@dup i)))
      (accumulate acc2 (list a1 b1) #:pred (?cond tidx (@dup i)))
      (accumulate acc3 (list a1 b1) #:pred (?cond tidx (@dup i)))
      (accumulate acc4 (list a1 b1) #:pred (?cond tidx (@dup i)))
      
      (accumulate acc1 (list a1 b2) #:pred (?cond tidx (@dup i)))
      (accumulate acc2 (list a1 b2) #:pred (?cond tidx (@dup i)))
      (accumulate acc3 (list a1 b2) #:pred (?cond tidx (@dup i)))
      (accumulate acc4 (list a1 b2) #:pred (?cond tidx (@dup i)))
      
      (accumulate acc1 (list a2 b1) #:pred (?cond tidx (@dup i)))
      (accumulate acc2 (list a2 b1) #:pred (?cond tidx (@dup i)))
      (accumulate acc3 (list a2 b1) #:pred (?cond tidx (@dup i)))
      (accumulate acc4 (list a2 b1) #:pred (?cond tidx (@dup i)))
      
      (accumulate acc1 (list a2 b2) #:pred (?cond tidx (@dup i)))
      (accumulate acc2 (list a2 b2) #:pred (?cond tidx (@dup i)))
      (accumulate acc3 (list a2 b2) #:pred (?cond tidx (@dup i)))
      (accumulate acc4 (list a2 b2) #:pred (?cond tidx (@dup i)))
      ))

  (reg-to-global acc1 C (+ block-offset threadId))
  (reg-to-global acc2 C (+ block-offset (x-y-z warpSize 0) threadId))
  (reg-to-global acc3 C (+ block-offset (x-y-z (* 2 warpSize) 0) threadId))
  (reg-to-global acc4 C (+ block-offset (x-y-z (* 3 warpSize) 0) threadId))
  )

;; Check correctness of a complete kernel against a spec.
(define (test)
  (for ([w (list 32)])
    (let ([ret (run-with-warp-size mult-spec mult32-shared-sketch w (* 1 w))])
      (pretty-display `(test ,w ,ret))
      (pretty-display `(cost ,(get-cost)))
      ))
  )
;(test)

;; Synthesize a kernel sketch given a spec.
(define (synthesis)
  (pretty-display "solving...")
  (assert (andmap
           ;(lambda (w) (run-with-warp-size mult-spec mult32-sketch w (* 1 w)))
           (lambda (w) (run-with-warp-size mult-spec mult64-sketch w (* 2 w)))
           (list 4)))
  (define cost (get-cost))
  (define sol (time (optimize #:minimize (list cost) #:guarantee (assert #t))))

  (define this-cost (evaluate cost sol))
  (print-forms sol)
  (pretty-display `(cost ,this-cost))
  )
(define t0 (current-seconds))
(synthesis)
(define t1 (current-seconds))
(- t1 t0)
