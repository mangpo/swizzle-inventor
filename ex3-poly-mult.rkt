#|
 | Copyright (c) 2018-2019, University of California, Berkeley.
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
    (let* ([lane-a (fan tidx warpSize 0 warpSize warpSize 1
                        i warpSize 1 warpSize)]
           [lane-b (fan tidx warpSize 1 warpSize warpSize 1
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
    (let* (;[lane-a (?fan-easy tidx warpSize i warpSize [])]
           ;[lane-b (?fan-easy tidx warpSize i warpSize [])]
           [lane-a (?lane-mod tidx (@dup i) 2 n [warpSize])]
           [lane-b (?lane-mod tidx (@dup i) 2 n [warpSize])]
           [a (shfl a-cached lane-a)]
           [b (shfl b-cached lane-b)]
          )
      (accumulate acc1 (list a b) #:pred (?cond-easy tidx (@dup i)))
      (accumulate acc2 (list a b) #:pred (?cond-easy tidx (@dup i)))
      ))

  (reg-to-global acc1 C globalID #:size (x-y-z (* 2 n)))
  (reg-to-global acc2 C (+ globalID (@dup (x-y-z n 0))) #:size (x-y-z (* 2 n)))
  )


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
    (let* (;[lane-a (?fan-easy tidx warpSize i warpSize [])]
           ;[lane-b (?fan-easy tidx warpSize i warpSize [])]
           [lane-a (?lane-mod tidx (@dup i) 2 n [warpSize])]
           [lane-b (?lane-mod tidx (@dup i) 2 n [warpSize])]
           [a (get a-cached lane-a tidy)]
           [b (get b-cached lane-b tidy)]
          )
      (accumulate acc1 (list a b) #:pred (?cond-easy tidx (@dup i)))
      (accumulate acc2 (list a b) #:pred (?cond-easy tidx (@dup i)))
      ))

  (reg-to-global acc1 C globalID #:size (x-y-z (* 2 n)))
  (reg-to-global acc2 C (+ globalID (@dup (x-y-z n 0))) #:size (x-y-z (* 2 n)))
  )


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
    (let* ([lane-a1 (fan tidx warpSize 0 warpSize warpSize 1
                        i warpSize 1 warpSize)]
           [lane-a2 (fan tidx warpSize 0 warpSize warpSize 1
                        i warpSize 1 warpSize)]
           [lane-b1 (fan tidx warpSize 1 warpSize warpSize 1
                        i warpSize (- warpSize 1) warpSize)]
           [lane-b2 (fan tidx warpSize 1 warpSize warpSize 1
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

(define (mult64-opt threadId blockID blockDim A B C n)
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
    (let* ([lane-a1 (fan tidx warpSize 0 warpSize warpSize 1
                         i warpSize 1 warpSize)]
           [lane-a2 (fan tidx warpSize 0 warpSize warpSize 1
                        i warpSize 1 warpSize)]
           [lane-b1 (fan tidx warpSize 1 warpSize warpSize 1
                        i warpSize (- warpSize 1) warpSize)]
           [lane-b2 (fan tidx warpSize 1 warpSize warpSize 1
                        i warpSize (- warpSize 1) warpSize)]
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
      
      ;(accumulate acc1 (list a1 b2) #:pred (?cond tidx (@dup i)))
      (accumulate acc2 (list a1 b2) #:pred #t #;(?cond tidx (@dup i)))
      ;(accumulate acc3 (list a1 b2) #:pred (?cond tidx (@dup i)))
      ;(accumulate acc4 (list a1 b2) #:pred (?cond tidx (@dup i)))
      
      ;(accumulate acc1 (list a2 b1) #:pred (?cond tidx (@dup i)))
      ;(accumulate acc2 (list a2 b1) #:pred (?cond tidx (@dup i)))
      (accumulate acc3 (list a2 b1) #:pred #t #;(?cond tidx (@dup i)))
      ;(accumulate acc4 (list a2 b1) #:pred (?cond tidx (@dup i)))
      
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

;; 6 statements: 3/7
;; 8 statements: 2/22
;; 16 statements, ?const: 8/2410
;; 16 statements, ?const, ?fan-easy: 8/849
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
  
  (define tidx (modulo (get-x threadId) 32))
  (define acc1 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc2 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc3 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc4 (create-accumulator (list bvand bvxor) identity blockDim))

  (for ([i warpSize])
    (let* ([lane-a1 (?fan tidx warpSize
                          i warpSize [])]
           [lane-a2 (?fan tidx warpSize
                          i warpSize [])]
           [lane-b1 (?fan tidx warpSize
                          i warpSize [])]
           [lane-b2 (?fan tidx warpSize
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
    (let* ([lane-a (fan tidx warpSize 0 warpSize warpSize 1
                        i warpSize 1 warpSize)]
           [lane-b (fan tidx warpSize 1 warpSize warpSize 1
                        i warpSize -1 warpSize)]
           #;[lane-a (?fan tidx warpSize
                         i warpSize #:fw 1)]
           #;[lane-b (?fan tidx warpSize
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
    (let* ([lane-a1 (fan tidx n 0 n n 1
                         i warpSize 1 warpSize 0)]
           [lane-a2 (fan tidx n 0 n n 1
                         i warpSize 1 warpSize warpSize)]
           [lane-b1 (fan tidx n 1 n n 1
                         i warpSize -1 warpSize 0)]
           [lane-b2 (fan tidx n 1 n n 1
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

;; warp-size=4, bw=8: 20/516
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
    (let* ([lane-a1 (?fan-easy tidx n
                          i warpSize [])]
           [lane-a2 (?fan-easy tidx n
                          i warpSize [])]
           [lane-b1 (?fan-easy tidx n
                          i warpSize [])]
           [lane-b2 (?fan-easy tidx n
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

(define (mult64-shared2 threadId blockID blockDim A B C n)
  (define warpId (get-warpId threadId))
  (define-shared a-cached (create-matrix (* (x-y-z 2 1) blockDim)))
  (define-shared b-cached (create-matrix (* (x-y-z 2 1) blockDim)))
  (define block-offset (* (x-y-z 2 1) blockID blockDim))
  (global-to-shared A a-cached
                    (x-y-z 2 1) ;; stride
                    block-offset
                    (* (x-y-z 2 1) blockDim))
  (global-to-shared B b-cached
                    (x-y-z 2 1) ;; stride
                    block-offset
                    (* (x-y-z 2 1) blockDim))

  (pretty-display `(a ,a-cached))
  
  (define tidx (get-idInWarp threadId))
  (define tidy (get-y threadId))
  (define acc1 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc2 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc3 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc4 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc5 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc6 (create-accumulator (list bvand bvxor) identity blockDim))

  (for ([i warpSize])
    (let* ([lane-a1 (fan tidx n 0 n n 1
                         i warpSize 1 warpSize #:offset 0)]
           [lane-a2 (fan tidx n 0 n n 1
                         i warpSize 1 warpSize #:offset warpSize)]
           [lane-b1 (fan tidx n 2 warpSize warpSize 1
                         i warpSize -1 warpSize #:offset 0 #:dg (quotient warpSize 2))]
           [lane-b2 (fan tidx n 2 warpSize warpSize 1
                         i warpSize -1 warpSize #:offset 1 #:dg (quotient warpSize 2))]
           [a1 (get a-cached lane-a1 tidy)]
           [a2 (get a-cached lane-a2 tidy)]
           [b1 (get b-cached lane-b1 tidy)]
           [b2 (get b-cached lane-b2 tidy)]
          )
      (accumulate acc1 (list a1 b1) #:pred (<= i (modulo (* 2 tidx) warpSize)))
      (accumulate acc3 (list a1 b1) #:pred (> i (modulo (* 2 tidx) warpSize)))
      
      (accumulate acc2 (list a1 b2) #:pred (<= i (modulo (+ 1 (* 2 tidx)) warpSize)))
      (accumulate acc4 (list a1 b2) #:pred (> i (modulo (+ 1 (* 2 tidx)) warpSize)))
      
      (accumulate acc3 (list a2 b1) #:pred (<= i (modulo (* 2 tidx) warpSize)))
      (accumulate acc5 (list a2 b1) #:pred (> i (modulo (* 2 tidx) warpSize)))
      
      (accumulate acc4 (list a2 b2) #:pred (<= i (modulo (+ 1 (* 2 tidx)) warpSize)))
      (accumulate acc6 (list a2 b2) #:pred (> i (modulo (+ 1 (* 2 tidx)) warpSize)))
      ))

  ;; for load int2
  (define index1 (lov2vol (x-y-z (modulo (* 2 tidx) warpSize) tidy)))
  (define index2 (+ (x-y-z 1 0) index1))
  (define half-offset (lov2vol (x-y-z (* warpSize (quotient (* 2 tidx) warpSize)) (@dup 0))))

  (pretty-display `(index ,block-offset ,half-offset ,index1 ,(+ block-offset half-offset index1)))
  (reg-to-global-update accumulate-merge acc1 C (+ block-offset half-offset index1))
  (reg-to-global-update accumulate-merge acc2 C (+ block-offset half-offset index2))
  (reg-to-global-update accumulate-merge acc3 C (+ block-offset half-offset (x-y-z (* 1 warpSize) 0) index1))
  (reg-to-global-update accumulate-merge acc4 C (+ block-offset half-offset (x-y-z (* 1 warpSize) 0) index2))
  (reg-to-global-update accumulate-merge acc5 C (+ block-offset half-offset (x-y-z (* 2 warpSize) 0) index1))
  (reg-to-global-update accumulate-merge acc6 C (+ block-offset half-offset (x-y-z (* 2 warpSize) 0) index2))

  )

(define (test)
  (for ([w (list 32)])
    (let ([ret (run-with-warp-size mult-spec mult32-shared-sketch w (* 1 w))])
      (pretty-display `(test ,w ,ret))
      (pretty-display `(cost ,(get-cost)))
      ))
  )
;(test)

(define (synthesis)
  (pretty-display "solving...")
  (assert (andmap
           (lambda (w) (run-with-warp-size mult-spec mult32-shared-sketch w (* 1 w)))
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

(define (load-synth)
  (define-values (block-size sizes A B C D C* D*)
    (create-IO 4))
  
  ;; Store
  (define (mult-store threadId blockId blockDim C D)
    (define warpID (get-warpId threadId))
    (define o
      (for/vector ([w  warpID]
                   [t threadId])
        (ID t w blockId)))
    (reg-to-global o C threadId)
    (reg-to-global o D threadId)
    )

  ;; Run spec -- already ran
  
  ;; Collect IDs
  (define C-IDs (create-matrix sizes))
  (define D-IDs (create-matrix sizes))
  (run-kernel mult-store sizes (x-y-z n-block) C-IDs D-IDs)

  (define-values (C-threads C-warps C-blocks) (get-grid-storage))
  (collect-inputs C C-IDs C-threads C-warps C-blocks)
  (define-values (D-threads D-warps D-blocks) (get-grid-storage))
  (collect-inputs D D-IDs D-threads D-warps D-blocks)

  (define warps (vector-list-append C-warps D-warps))
  (define a-regs (num-regs warps A))
  (pretty-display `(a-regs ,a-regs))
  (define b-regs (num-regs warps B))
  (pretty-display `(b-regs ,b-regs))

  ;; Load
  (define (mult-load threadId blockId blockDim A B C-warp-spec D-warp-spec)
    (define warpId (get-warpId threadId))
    ;; sketch starts
    (define A-cached (create-matrix-local (x-y-z a-regs)))
    (define B-cached (create-matrix-local (x-y-z b-regs)))
    (global-to-local A A-cached
                        (x-y-z 1) ;; stride
                        (x-y-z (?warp-offset [(get-x blockId) (get-x blockDim)] [warpId warpSize])) ;; offset
                        (x-y-z (?warp-size warpSize 1)) ;; load size --> TODO: minimize load size
                        #f)
    (global-to-local B B-cached
                        (x-y-z 1) ;; stride
                        (x-y-z (?warp-offset [(get-x blockId) (get-x blockDim)] [warpId warpSize])) ;; offset
                        (x-y-z (?warp-size warpSize 1)) ;; load size
                        #f)
    ;; sketch ends
    (check-warp-input C-warp-spec A A-cached warpId blockId)
    (check-warp-input D-warp-spec A A-cached warpId blockId)
    (check-warp-input C-warp-spec B B-cached warpId blockId)
    (check-warp-input D-warp-spec B B-cached warpId blockId)
    )

  (run-kernel mult-load sizes (x-y-z n-block) A B C-warps D-warps)
  (define sol
    (time
     (synthesize
      #:forall (append (symbolics A) (symbolics B))
      #:guarantee (assert #t))))
  (print-forms sol)
  )
;(load-synth)
