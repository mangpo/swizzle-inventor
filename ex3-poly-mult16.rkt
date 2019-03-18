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

(require rosette/lib/synthax)
(require "util.rkt" "cuda.rkt" "cuda-synth.rkt")

(define n-block 1)
(define block-dim-y 1)

(define (create-IO warpSize n)
  (set-warpSize warpSize)
  (define A (create-matrix (x-y-z n n-block) gen-uid))
  (define B (create-matrix (x-y-z n n-block) gen-uid))
  (define C (create-matrix (x-y-z (* 2 n) n-block)))
  (define C* (create-matrix (x-y-z (* 2 n) n-block)))
  (values A B C C*))

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

(define (mult16 threadId blockID blockDim A B C n)
  (define globalID (+ threadId (* blockID blockDim)))
  (define a-cached (create-matrix-local (x-y-z 1 1)))
  (define b-cached (create-matrix-local (x-y-z 1 1)))
  (global-to-local A a-cached
                   (x-y-z 1 1) ;; stride
                   (* (quotient globalID (x-y-z warpSize 1))
                      (x-y-z n 1))
                   (x-y-z n 1)
                   #f #:warp-shape (x-y-z warpSize 1) #:round (x-y-z 1 1) #:size (x-y-z n 1))
  (global-to-local B b-cached
                   (x-y-z 1 1) ;; stride
                   (* (quotient globalID (x-y-z warpSize 1))
                      (x-y-z n 1))
                   (x-y-z n 1)
                   #f #:warp-shape (x-y-z warpSize 1) #:round (x-y-z 1 1) #:size (x-y-z n 1))
  
  (define tidx (get-idInWarp threadId))
  (define acc1 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc2 (create-accumulator (list bvand bvxor) identity blockDim))

  (define quad (/ n 2))
  (define ai (quotient tidx n))
  (define bi (modulo (quotient tidx quad) 2))
  (define id (modulo tidx quad))

  (for ([i quad])
    (let* ([lane-a1 (@dup i) #;(?fan-easy id quad
                          i quad [])]
           [lane-b1 (modulo (- id i) quad) #;(?fan-easy id 8
                          i quad [])]
           [a1 (shfl (get a-cached (@dup 0) (@dup 0)) (+ (* quad ai) lane-a1))]
           [b1 (shfl (get b-cached (@dup 0) (@dup 0)) (+ (* quad bi) lane-b1))]
          )
      (accumulate acc1 (list a1 b1) #:pred (<= i id) #;(?cond id (@dup i)))
      (accumulate acc2 (list a1 b1) #:pred (> i id) #;(?cond id (@dup i)))
      ))

  (reg-to-global (ite (= ai 0) acc1 acc2) C globalID)
  (reg-to-global-update accumulate-merge acc2 C (+ globalID (@dup (x-y-z quad 0))) #:pred (= ai 0))
  (reg-to-global-update accumulate-merge acc1 C (- globalID (@dup (x-y-z quad 0))) #:pred (= ai 1))
  )

(define (mult16-sketch threadId blockID blockDim A B C n)
  (define globalID (+ threadId (* blockID blockDim)))
  (define a-cached (create-matrix-local (x-y-z 1 1)))
  (define b-cached (create-matrix-local (x-y-z 1 1)))
  (global-to-local A a-cached
                   (x-y-z 1 1) ;; stride
                   (* (quotient globalID (x-y-z warpSize 1))
                      (x-y-z n 1))
                   (x-y-z n 1)
                   #f #:warp-shape (x-y-z warpSize 1) #:round (x-y-z 1 1) #:size (x-y-z n 1))
  (global-to-local B b-cached
                   (x-y-z 1 1) ;; stride
                   (* (quotient globalID (x-y-z warpSize 1))
                      (x-y-z n 1))
                   (x-y-z n 1)
                   #f #:warp-shape (x-y-z warpSize 1) #:round (x-y-z 1 1) #:size (x-y-z n 1))
  
  (define tidx (get-idInWarp threadId))
  (define acc1 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc2 (create-accumulator (list bvand bvxor) identity blockDim))

  (define quad (/ n 2))
  (define ai (quotient tidx n))
  (define bi (modulo (quotient tidx quad) 2))
  (define id (modulo tidx quad))

  (for ([i quad])
    (let* ([lane-a1 (?fan id quad
                          i quad [])]
           [lane-b1 (?fan id quad
                          i quad [])]
           [a1 (shfl (get a-cached (@dup 0) (@dup 0)) (+ (* quad ai) lane-a1))]
           [b1 (shfl (get b-cached (@dup 0) (@dup 0)) (+ (* quad bi) lane-b1))]
          )
      (accumulate acc1 (list a1 b1) #:pred (?cond id (@dup i)))
      (accumulate acc2 (list a1 b1) #:pred (?cond id (@dup i)))
      ))

  (reg-to-global (ite (= ai 0) acc1 acc2) C globalID)
  (reg-to-global-update accumulate-merge acc2 C (+ globalID (@dup (x-y-z quad 0))) #:pred (= ai 0))
  (reg-to-global-update accumulate-merge acc1 C (- globalID (@dup (x-y-z quad 0))) #:pred (= ai 1))
  )


(define (mult16-shared threadId blockID blockDim A B C n)
  (define globalID (+ threadId (* blockID blockDim)))
  (define-shared a-cached (create-matrix blockDim))
  (define-shared b-cached (create-matrix blockDim))
  (define block-offset (* (x-y-z 2 1) blockID blockDim))
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

  (define tidx (get-x threadId))
  (define tidy (get-y threadId))
  (define acc1 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc2 (create-accumulator (list bvand bvxor) identity blockDim))

  (define quad (/ n 2))
  (define ai (quotient tidx n))
  (define bi (modulo (quotient tidx quad) 2))
  (define id (modulo tidx quad))

  (for ([i quad])
    (let* ([lane-a1 (@dup i) #;(?fan-easy id quad
                          i quad [])]
           [lane-b1 (modulo (- id i) quad) #;(?fan-easy id 8
                          i quad [])]
           [a1 (get a-cached (+ (* quad ai) lane-a1) tidy)]
           [b1 (get b-cached (+ (* quad bi) lane-b1) tidy)]
          )
      (accumulate acc1 (list a1 b1) #:pred (<= i id) #;(?cond id (@dup i)))
      (accumulate acc2 (list a1 b1) #:pred (> i id) #;(?cond id (@dup i)))
      ))

  (reg-to-global (ite (= ai 0) acc1 acc2) C globalID)
  (reg-to-global-update accumulate-merge acc2 C (+ globalID (@dup (x-y-z quad 0))) #:pred (= ai 0))
  (reg-to-global-update accumulate-merge acc1 C (- globalID (@dup (x-y-z quad 0))) #:pred (= ai 1))
  )

(define (mult16-shared-sketch threadId blockID blockDim A B C n)
  (define globalID (+ threadId (* blockID blockDim)))
  (define-shared a-cached (create-matrix blockDim))
  (define-shared b-cached (create-matrix blockDim))
  (define block-offset (* (x-y-z 2 1) blockID blockDim))
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

  (define tidx (get-x threadId))
  (define tidy (get-y threadId))
  (define acc1 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc2 (create-accumulator (list bvand bvxor) identity blockDim))

  (define quad (/ n 2))
  (define ai (quotient tidx n))
  (define bi (modulo (quotient tidx quad) 2))
  (define id (modulo tidx quad))

  (for ([i quad])
    (let* ([lane-a1 (?fan id quad
                          i quad [])]
           [lane-b1 (?fan id 8
                          i quad [])]
           [a1 (get a-cached (+ (* quad ai) lane-a1) tidy)]
           [b1 (get b-cached (+ (* quad bi) lane-b1) tidy)]
          )
      (accumulate acc1 (list a1 b1) #:pred (?cond id (@dup i)))
      (accumulate acc2 (list a1 b1) #:pred (?cond id (@dup i)))
      ))

  (reg-to-global (ite (= ai 0) acc1 acc2) C globalID)
  (reg-to-global-update accumulate-merge acc2 C (+ globalID (@dup (x-y-z quad 0))) #:pred (= ai 0))
  (reg-to-global-update accumulate-merge acc1 C (- globalID (@dup (x-y-z quad 0))) #:pred (= ai 1))
  )

(define (mult16-shared-sketch-sol threadId blockID blockDim A B C n)
   (define globalID (+ threadId (* blockID blockDim)))
   (define-shared a-cached (create-matrix blockDim))
   (define-shared b-cached (create-matrix blockDim))
   (define block-offset (* (x-y-z 2 1) blockID blockDim))
   (global-to-shared
    A
    a-cached
    (x-y-z 1 1)
    (* blockDim blockID)
    blockDim
    #f
    #:round
    (x-y-z 1 1)
    #:size
    (x-y-z n 1))
   (global-to-shared
    B
    b-cached
    (x-y-z 1 1)
    (* blockDim blockID)
    blockDim
    #f
    #:round
    (x-y-z 1 1)
    #:size
    (x-y-z n 1))
   (define tidx (get-x threadId))
   (define tidy (get-y threadId))
   (define acc1 (create-accumulator (list bvand bvxor) identity blockDim))
   (define acc2 (create-accumulator (list bvand bvxor) identity blockDim))
   (define quad (/ n 2))
   (define ai (quotient tidx n))
   (define bi (modulo (quotient tidx quad) 2))
   (define id (modulo tidx quad))
   (for
    ((i quad))
    (let* ((lane-a1 (fan id quad 1 quad quad 1 i quad -1 quad 0))
           (lane-b1 (fan id 8 0 8 8 1 i quad 1 quad 0))
           (a1 (get a-cached (+ (* quad ai) lane-a1) tidy))
           (b1 (get b-cached (+ (* quad bi) lane-b1) tidy)))
      (accumulate acc1 (list a1 b1) #:pred (<= (@dup i) id))
      (accumulate acc2 (list a1 b1) #:pred (> (@dup i) id))))
   (reg-to-global (ite (= ai 0) acc1 acc2) C globalID)
   (reg-to-global-update
    accumulate-merge
    acc2
    C
    (+ globalID (@dup (x-y-z quad 0)))
    #:pred
    (= ai 0))
   (reg-to-global-update
    accumulate-merge
    acc1
    C
    (- globalID (@dup (x-y-z quad 0)))
    #:pred
    (= ai 1)))

(define (test)
  (for ([w (list 32)])
    (let ([ret (run-with-warp-size mult-spec mult16-shared-sketch-sol w (/ w 2))])
      (pretty-display `(test ,w ,ret))
      (pretty-display `(cost ,(get-cost)))
      ))
  )
;(test)

;; warp size 4, concrete load: 2 s
;; warp size 4 & 5, concrete load: 7 s
;; warp size 4 & 5, synth load: 5/9 s
;; warp size 32: 44/776 s
(define (synthesis)
  (pretty-display "solving...")
  (assert (andmap
           (lambda (w) (run-with-warp-size mult-spec mult16-shared-sketch w (/ w 2)))
           (list 16)))
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
