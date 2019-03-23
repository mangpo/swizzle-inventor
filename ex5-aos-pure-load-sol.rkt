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

(define struct-size 6)
(define n-block 1)

(define (create-IO warpSize)
  (set-warpSize warpSize)
  (define block-size (* 2 warpSize))
  (define array-size (* n-block block-size))
  (define I-sizes (x-y-z (* array-size struct-size)))
  (define I (create-matrix I-sizes gen-uid))
  (define O (create-matrix I-sizes))
  (define O* (create-matrix I-sizes))
  (values block-size I-sizes I O O*))

(define (run-with-warp-size spec kernel w)
  (define-values (block-size I-sizes I O O*)
  (create-IO w))

  (define c (gcd struct-size warpSize))
  (define a (/ struct-size c))
  (define b (/ warpSize c))

  (run-kernel spec (x-y-z block-size) (x-y-z n-block) I O a b c)
  (run-kernel kernel (x-y-z block-size) (x-y-z n-block) I O* a b c)
  (define ret (equal? O O*))
  (pretty-display `(O ,(print-vec O)))
  (pretty-display `(O* ,(print-vec O*)))
  ret)

(define (AOS-load-spec threadId blockID blockDim I O a b c)
  (define I-cached (create-matrix-local (x-y-z struct-size)))
  (define warpID (get-warpId threadId))
  (define offset (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))  ;; warpID = (threadIdy * blockDimx + threadIdx)/warpSize
  (define gid (get-global-threadId threadId blockID))
  (global-to-local I I-cached
                 (x-y-z struct-size)
                 offset (x-y-z (* warpSize struct-size)) #f)
  (local-to-global I-cached O
                      (x-y-z 1) offset (x-y-z (* warpSize struct-size)) #f #:round struct-size)
  )

(define (print-vec x)
  (format "#(~a)" (string-join (for/list ([xi x]) (format "~a" xi)))))

(define (AOS-load2 threadId blockID blockDim I O a b c)
   (define I-cached (create-matrix-local (x-y-z struct-size)))
   (define O-cached (create-matrix-local (x-y-z struct-size)))
   (define warpID (get-warpId threadId))
   (define offset
     (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))
   (define gid (get-global-threadId threadId blockID))
   (global-to-local
    I
    I-cached
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size) #:round struct-size)
    #f)
   (define indices (make-vector struct-size))
   (define indices-o (make-vector struct-size))
   (define localId (get-idInWarp threadId))
   (for
    ((i struct-size))
    (let* ((index (sw-xform i struct-size 0 1 2 1 localId warpSize 0 1))
           (lane (sw-xform localId warpSize 2 16 32 -1 i struct-size 0 1))
           (x (shfl (get I-cached index) lane))
           (index-o (sw-xform i struct-size 0 1 2 1 localId warpSize 0 16)))
      (unique-warp (modulo lane warpSize))
      (vector-set! indices i index)
      (vector-set! indices-o i index-o)
      (set O-cached index-o x)))
   (for
    ((t blockSize))
    (let ((l
           (for/list ((i struct-size)) (vector-ref (vector-ref indices i) t)))
          (lo
           (for/list
            ((i struct-size))
            (vector-ref (vector-ref indices-o i) t))))
      (unique-list l)
      (unique-list lo)))
   (local-to-global
    O-cached
    O
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f #:round struct-size))

(define (AOS-load-rcr-2 threadId blockID blockDim I O a b c)
   (define I-cached (create-matrix-local (x-y-z struct-size)))
   (define temp (create-matrix-local (x-y-z struct-size)))
   (define O-cached (create-matrix-local (x-y-z struct-size)))
   (define warpID (get-warpId threadId))
   (define offset
     (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))
   (define gid (get-global-threadId threadId blockID))
   (global-to-local
    I
    I-cached
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f #:round struct-size)
   (define indices (make-vector struct-size))
   (define indices-o (make-vector struct-size))
   (define localId (get-idInWarp threadId))
   (for
    ((i struct-size))
    (let* ((lane1 (sw-xform localId warpSize 0 1 2 1 i struct-size 0 1))
           (x (shfl (get I-cached (@dup i)) lane1)))
      (set temp (@dup i) x)))
   (for
    ((i struct-size))
    (let* ((index (sw-xform i struct-size 0 1 2 1 localId warpSize 0 1))
           (lane2 (sw-xform localId warpSize 16 2 32 1 i struct-size 15 1))
           (x (shfl-send (get temp index) lane2)))
      (set O-cached (@dup i) x)))
   (local-to-global
    O-cached
    O
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f #:round struct-size))

(define (AOS-load3 threadId blockID blockDim I O a b c)
  (define I-cached (create-matrix-local (x-y-z struct-size)))
  (define O-cached (create-matrix-local (x-y-z struct-size)))
  (define localId (modulo (get-x threadId) 32))
  (define offset (* struct-size (- (+ (* blockID blockDim) (get-x threadId)) localId)))
   (global-to-local
    I
    I-cached
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f #:round struct-size)
   (for
    ((i struct-size))
    (let* ((index (sw-xform i struct-size 2 3 3 1 localId warpSize 0 1))
           (lane (sw-xform localId warpSize 3 32 32 1 i struct-size 0 1))
           (x (shfl (get I-cached index) lane))
           (index-o (sw-xform i struct-size 1 3 3 1 localId warpSize 0 warpSize)))
      (unique-warp (modulo lane warpSize))
      (set O-cached index-o x)))
   (local-to-global
    O-cached
    O
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f #:round struct-size))

(define (AOS-loadhsh3 threadId blockID blockDim I O a b c)
   (define I-cached (create-matrix-local (x-y-z struct-size)))
   (define temp (create-matrix-local (x-y-z struct-size)))
   (define O-cached (create-matrix-local (x-y-z struct-size)))
   (define warpID (get-warpId threadId))
   (define offset
     (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))
   (define gid (get-global-threadId threadId blockID))
   (global-to-local
    I
    I-cached
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f #:round struct-size)
   (define indices (make-vector struct-size))
   (define indices-o (make-vector struct-size))
   (define localId (get-idInWarp threadId))
   (for
    ((i struct-size))
    (let* ((lane1 (sw-xform localId warpSize 0 1 32 1 i struct-size 31 1))
           (x (shfl (get I-cached (@dup i)) lane1)))
      (set temp (@dup i) x)))
   (for
    ((i struct-size))
    (let* ((index (sw-xform i struct-size 2 3 3 1 localId warpSize 0 1))
           (lane2 (sw-xform localId warpSize 11 32 32 1 i struct-size 20 1))
           (x (shfl-send (get temp index) lane2)))
      (set O-cached (@dup i) x)))
   (local-to-global
    O-cached
    O
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f #:round struct-size))

(define (AOS-loadhsh3* threadId blockID blockDim I O a b c)
  (define I-cached (create-matrix-local (x-y-z struct-size)))
  (define warpID (get-warpId threadId))
  (define offset
    (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))
  (define gid (get-global-threadId threadId blockID))
  (global-to-local
   I
   I-cached
   (x-y-z 1)
   offset
   (x-y-z (* warpSize struct-size))
   #f #:round struct-size
   #:shfl (lambda (localId i) (sw-xform localId warpSize 0 1 32 1 i struct-size 31 1)))
  (define localId (get-idInWarp threadId))
  (define O-cached (permute-vector I-cached struct-size
                                   (lambda (i) (sw-xform i struct-size 2 3 3 1 localId warpSize 0 1))))
  (local-to-global
   O-cached
   O
   (x-y-z 1)
   offset
   (x-y-z (* warpSize struct-size))
   #f #:round struct-size
   #:shfl (lambda (localId i)
            (sw-xform localId warpSize 11 32 32 1 i struct-size 20 1)))
  )

(define (AOS-load4 threadId blockID blockDim I O a b c)
   (define I-cached (create-matrix-local (x-y-z struct-size)))
   (define O-cached (create-matrix-local (x-y-z struct-size)))
   (define warpID (get-warpId threadId))
   (define offset
     (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))
   (define gid (get-global-threadId threadId blockID))
   (global-to-local
    I
    I-cached
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f #:round struct-size)
   (define indices (make-vector struct-size))
   (define indices-o (make-vector struct-size))
   (define localId (get-idInWarp threadId))
   (for
    ((i struct-size))
    (let* ((index (sw-xform i struct-size 3 4 4 1 localId warpSize 0 1))
           (lane (sw-xform localId warpSize 4 8 32 -1
                      i struct-size 0 1))
           (x (shfl (get I-cached index) lane))
           (index-o (sw-xform i struct-size 0 1 4 1 localId warpSize 0 8)))
      (pretty-display `(lane ,lane))
      (unique-warp (modulo lane warpSize))
      (vector-set! indices i index)
      (vector-set! indices-o i index-o)
      (set O-cached index-o x)))
   (for
    ((t blockSize))
    (let ((l
           (for/list ((i struct-size)) (vector-ref (vector-ref indices i) t)))
          (lo
           (for/list
            ((i struct-size))
            (vector-ref (vector-ref indices-o i) t))))
      (unique-list l)
      (unique-list lo)))
   (local-to-global
    O-cached
    O
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f #:round struct-size))

(define (AOS-load-rcr-4 threadId blockID blockDim I O a b c)
   (define I-cached (create-matrix-local (x-y-z struct-size)))
   (define temp (create-matrix-local (x-y-z struct-size)))
   (define O-cached (create-matrix-local (x-y-z struct-size)))
   (define warpID (get-warpId threadId))
   (define offset
     (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))
   (define gid (get-global-threadId threadId blockID))
   (global-to-local
    I
    I-cached
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f #:round struct-size)
   (define indices (make-vector struct-size))
   (define indices-o (make-vector struct-size))
   (define localId (get-idInWarp threadId))
   (for
    ((i struct-size))
    (let* ((lane1 (sw-xform localId warpSize 0 1 4 1 i struct-size 0 1))
           (x (shfl (get I-cached (@dup i)) lane1)))
      (set temp (@dup i) x)))
   (for
    ((i struct-size))
    (let* ((index (sw-xform i struct-size 0 1 4 1 localId warpSize 0 -1))
           (lane2 (sw-xform localId warpSize 24 4 32 1 i struct-size 7 1))
           (x (shfl-send (get temp index) lane2)))
      (set O-cached (@dup i) x)))
   (local-to-global
    O-cached
    O
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f #:round struct-size))

(define (AOS-load5 threadId blockID blockDim I O a b c)
   (define I-cached (create-matrix-local (x-y-z struct-size)))
   (define O-cached (create-matrix-local (x-y-z struct-size)))
   (define warpID (get-warpId threadId))
   (define offset
     (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))
   (define gid (get-global-threadId threadId blockID))
   (global-to-local
    I
    I-cached
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f #:round struct-size)
   (define indices (make-vector struct-size))
   (define indices-o (make-vector struct-size))
   (define localId (get-idInWarp threadId))
   (for
    ((i struct-size))
    (let* ((index (sw-xform i struct-size 3 5 5 1 localId warpSize 1 1))
           (lane (sw-xform localId warpSize 5 32 32 1 i struct-size 0 1))
           (x (shfl (get I-cached index) lane))
           (index-o (sw-xform i struct-size 1 5 5 -1 localId warpSize 0 1)))
      (unique-warp (modulo lane warpSize))
      (vector-set! indices i index)
      (vector-set! indices-o i index-o)
      (set O-cached index-o x)))
   (for
    ((t blockSize))
    (let ((l
           (for/list ((i struct-size)) (vector-ref (vector-ref indices i) t)))
          (lo
           (for/list
            ((i struct-size))
            (vector-ref (vector-ref indices-o i) t))))
      (unique-list l)
      (unique-list lo)))
   (local-to-global
    O-cached
    O
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f #:round struct-size))

(define (AOS-load-rcr-5 threadId blockID blockDim I O a b c)
   (define I-cached (create-matrix-local (x-y-z struct-size)))
   (define temp (create-matrix-local (x-y-z struct-size)))
   (define O-cached (create-matrix-local (x-y-z struct-size)))
   (define warpID (get-warpId threadId))
   (define offset
     (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))
   (define gid (get-global-threadId threadId blockID))
   (global-to-local
    I
    I-cached
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f #:round struct-size)
   (define indices (make-vector struct-size))
   (define indices-o (make-vector struct-size))
   (define localId (get-idInWarp threadId))
   (for
    ((i struct-size))
    (let* ((x (get I-cached (@dup i))))
      (set temp (@dup i) x)))
   (for
    ((i struct-size))
    (let* ((index (sw-xform i 5 3 5 5 1
                       localId warpSize 2 warpSize))
           #;(index
            (modulo (+ (* 3 i) (* localId 2)) 5))
           (lane2 (sw-xform localId 32 13 32 32 1
                       i 5 19 5))
           #;(lane2
            (modulo
             (- (* 13 localId) (* 13 i))
             32))
           (x (shfl-send (get temp index) lane2)))
      ;(pretty-display `(lane ,(print-vec (modulo lane2 32))))
      (set O-cached (@dup i) x)))
   (local-to-global
    O-cached
    O
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f #:round struct-size))

(define (AOS-load6 threadId blockID blockDim I O a b c)
  (define I-cached (create-matrix-local (x-y-z struct-size)))
  (define O-cached (create-matrix-local (x-y-z struct-size)))
  
  (define localId (modulo (get-x threadId) 32))
  (define offset (* struct-size (- (+ (* blockID blockDim) (get-x threadId)) localId)))
  
  (global-to-local I I-cached
                 (x-y-z 1)
                 offset
                 (x-y-z (* warpSize struct-size)) #f #:round struct-size)

  ;; column shuffle
  (define I-cached2 (permute-vector I-cached struct-size
                                    (lambda (i)
                                      #;(+ (modulo (quotient (+ (modulo (- i localId) struct-size) 1) 2) 3)
                                           (* 3 (modulo (- i localId) 2)))
                                      (sw-xform i struct-size 3 2 struct-size 1
                                           localId warpSize 0 warpSize #;offset 3
                                           #:ecr 5 #:ec 1)
                                      )))

  ;; row shuffle
  (for ([i struct-size])
    (let* ([lane
            #;(modulo
             (+ (* 6 localId) (modulo (+ i (quotient localId 16)) 6))
             warpSize)
            (sw-xform localId warpSize 6 16 warpSize -1
                      i struct-size 1 struct-size  #;offset 0
                      #:gcd 6)]
           [x (shfl (get I-cached2 (@dup i)) lane)]
           )
      (set O-cached (@dup i) x))
    )
  
  ;; column shuffle
  (define O-cached2 (permute-vector O-cached struct-size
                                    (lambda (i)
                                      #;(modulo (- i (quotient localId 16)) struct-size)
                                      (sw-xform i struct-size 1 struct-size struct-size 1
                                                     localId warpSize 0 -16  #;offset 0))))
  
  (local-to-global O-cached2 O
                      (x-y-z 1)
                      offset
                      (x-y-z (* warpSize struct-size)) #f #:round struct-size)
  )

(define (test)
  (for ([w (list 32)])
    (let ([ret (run-with-warp-size AOS-load-spec AOS-load6 w)])
      (pretty-display `(test ,w ,ret))))
  )
(test)

