#|
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

(define struct-size 3)
(define n-block 1)

(define syms #f)

(define (create-IO warpSize)
  (set-warpSize warpSize)
  (define block-size (* 1 warpSize))
  (define array-size (* n-block block-size))
  (define I-sizes (x-y-z (* array-size struct-size)))
  (define O-sizes (x-y-z array-size))
  (define I (create-matrix I-sizes gen-sym))
  (set! syms (symbolics I))
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
  ;(pretty-display O)
  ;(pretty-display O*)
  (equal? O O*)
  )

(define (AOS-sum-spec I O O-sizes)
  (for ([i (get-x O-sizes)])
    (let ([o 0])
      (for ([j struct-size])
        (set! o (+ o (get I (+ (* i struct-size) j)))))
      (set O i o)))
  )

(define (AOS-sum3 threadId blockID blockDim I O I-sizes O-sizes a b c)
   (define I-cached (create-matrix-local (x-y-z struct-size)))
   (define gid (+ (* blockID blockDim) threadId))
   (define localId (get-idInWarp threadId))
   (global-to-local
    I
    I-cached
    (x-y-z 1)
    (* struct-size (- gid localId))
    (x-y-z (* warpSize struct-size))
    #f
    #:round
    struct-size)
   (define o 0)
   (define I-cached2
     (permute-vector
      I-cached
      struct-size
      (lambda (i) (sw-xform i struct-size 0 1 1 -1 localId warpSize -1 8 0))))
   (for
    ((i struct-size))
    (let* ((lane (sw-xform localId warpSize 2 -32 32 -1 i struct-size 16 3 10))
           (x (shfl (get I-cached2 (@dup i)) lane)))
      (set! o (+ o x))))
   (reg-to-global o O gid))

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

  (define o 0)

  ;; column shuffle
  (define I-cached2 (permute-vector I-cached struct-size
                                    (lambda (i) (?sw-xform-easy i struct-size localId warpSize))))

  ;; row shuffle
  (for ([i struct-size])
    (let* ([lane (?sw-xform-easy localId warpSize i struct-size)]
           [x (shfl (get I-cached2 (@dup i)) lane)]
           )
      (set! o (+ o x))
    ))
  
  (reg-to-global o O gid)
  )

(define (test)
  (for ([w (list 32)])
    (let ([ret (run-with-warp-size AOS-sum-spec AOS-sum3 w)])
      (pretty-display `(test ,w ,ret ,(get-cost)))))
  )
;(test)

(define (synthesis)
  (pretty-display "solving...")
  (define t
   (andmap (lambda (w) (run-with-warp-size AOS-sum-spec AOS-sum-sketch w))
           (list 32)))
  (define sol (time (synthesize #:forall syms
                                #:guarantee (assert t))))
  (print-forms sol)
  )
(define t0 (current-seconds))
(synthesis)
(define t1 (current-seconds))
(- t1 t0)
