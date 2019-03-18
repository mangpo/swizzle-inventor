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

(require "util.rkt" "cuda.rkt" "cuda-synth.rkt")

(define struct-size 2)
(define n-block 1)

(define (create-IO warpSize)
  (set-warpSize warpSize)
  (define block-size (* 1 warpSize))
  (define array-size (* n-block block-size))
  (define I-sizes (x-y-z (* array-size struct-size)))
  (define O-sizes (x-y-z array-size))
  (define I (create-matrix I-sizes gen-uid))
  (define O (create-matrix O-sizes))
  (define O* (create-matrix O-sizes))
  (values block-size I-sizes O-sizes I O O*))

(define (run-with-warp-size spec kernel w)
  (define-values (block-size I-sizes O-sizes I O O*)
    (create-IO w))
  
  (define c (gcd struct-size warpSize))
  (define a (/ struct-size c))
  (define b (/ warpSize c))

  (reset-cost)
  (spec I O O-sizes)
  (pretty-display `(spec-cost ,(get-cost)))
  (run-kernel kernel (x-y-z block-size) (x-y-z n-block) I O* I-sizes O-sizes a b c)
  ;(acc-print O)
  ;(acc-print O*)
  (acc-equal? O O*))

(define (AOS-sum-spec I O O-sizes)
  (for ([i (get-x O-sizes)])
    (let ([o (create-accumulator (list +) identity)])
      (for ([j struct-size])
        (accumulate o (get I (+ (* i struct-size) j))))
      (set O i o)))
  )

(define (AOS-sum-slow threadId blockID blockDim I O I-sizes O-sizes a b c)
  (define I-cached (create-matrix-local (x-y-z struct-size)))
  (define warpID (get-warpId threadId))
  (define offset (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))  ;; warpID = (threadIdy * blockDimx + threadIdx)/warpSize
  (define gid (get-global-threadId threadId blockID))
  (global-to-local I I-cached
                 (x-y-z struct-size)
                 offset (x-y-z (* warpSize struct-size)) #f)

  (define localId (get-idInWarp threadId))
  (define o (create-accumulator (list +) identity blockDim))
  (for ([i struct-size])
    (let* ([index (@dup i)]
           [lane localId]
           [x (shfl (get I-cached index) lane)])
      (accumulate o x)
      ))
  (reg-to-global o O gid)
  )

(define (AOS-sum-sketch threadId blockID blockDim I O I-sizes O-sizes a b c)
  
  (define I-cached (create-matrix-local (x-y-z struct-size)))
  ;(define warpID (get-warpId threadId))
  ;(define gid (get-global-threadId threadId blockID))
  (define gid (+ (* blockID blockDim) threadId))
  (define localId (get-idInWarp threadId))
  (global-to-local I I-cached
                        (x-y-z 1) ;; stride
                        ;(+ (* struct-size blockID blockDim) (* struct-size warpID warpSize))
                        (* struct-size (- gid localId))
                        (x-y-z (* warpSize struct-size))
                        #f #:round struct-size)

  (define o (create-accumulator (list +) identity blockDim))

  ;; column shuffle
  (define I-cached2 (permute-vector I-cached struct-size
                                    (lambda (i) (?lane-mod localId (@dup i) 2 struct-size)
                                      #;(?fan i struct-size localId warpSize))))
  (pretty-display "finish permute-vector")

  ;; row shuffle
  (for ([i struct-size])
    (let* (;[lane (?fan localId warpSize i struct-size)]
           [lane (?lane-mod localId (@dup i) 2 warpSize)]
           [x (shfl (get I-cached2 (@dup i)) lane)]
           )
      (accumulate o x #:pred #t)
    ))
  
  (reg-to-global o O gid)
  (pretty-display "finish kernel")
  )

(define (test)
  (for ([w (list 32)])
    (let ([ret (run-with-warp-size AOS-sum-spec AOS-sum-sketch w)])
      (pretty-display `(test ,w ,ret ,(get-cost)))))
  )
;(test)

(define (synthesis)
  (pretty-display "solving...")
  (assert
   (andmap (lambda (w) (run-with-warp-size AOS-sum-spec AOS-sum-sketch w))
           (list 32)))
  (define cost (get-cost))
  (define sol (time (optimize #:minimize (list cost) #:guarantee (assert #t))))

  (define this-cost (evaluate cost sol))
  (print-forms sol)
  (pretty-display `(cost ,this-cost))
  
  ;(define sol2 (solve (assert (< cost this-cost))))
  ;(pretty-display `(cost2 ,(evaluate cost sol2)))
  )
(define t0 (current-seconds))
(synthesis)
(define t1 (current-seconds))
(- t1 t0)
