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

(define (create-IO warpSize)
  (set-warpSize warpSize)
  (define block-size warpSize)
  (define sizes (x-y-z warpSize))
  (define A (create-matrix sizes gen-uid))
  (define B (create-matrix sizes gen-uid))
  (define C (create-matrix sizes))
  (define D (create-matrix sizes))
  (define C* (create-matrix sizes))
  (define D* (create-matrix sizes))
  (values block-size sizes A B C D C* D*))

(define (run-with-warp-size spec kernel w)
  (define-values (block-size sizes A B C D C* D*)
    (create-IO w))

  (spec A B C D sizes)
  (run-kernel kernel (x-y-z block-size) (x-y-z n-block) A B C* D* sizes)
  ;(acc-print O*)
  (and (acc-equal? C C*) (acc-equal? D D*)))

(define (mult-spec A B C D sizes)
  (for ([index (get-x sizes)])
    (let ([c (create-accumulator (list bvand bvxor) identity)])
      (for ([i (add1 index)])
        (let ([a (get A i)]
              [b (get B (- index i))])
          (accumulate c (list a b))))
      (set C index c))
    (let ([d (create-accumulator (list bvand bvxor) identity)])
      (for ([i (range (add1 index) (get-x sizes))])
        (let ([a (get A i)]
              [b (get B (- (+ index (get-x sizes)) (+ i 1)))])
          (accumulate d (list a b))))
      (set D index d))))

(define (mult threadId blockID blockDim A B C D sizes)
  ;(define a-cached #f)
  ;(define b-cached #f)
  ;(global-to-reg A a-cached threadId sizes)
  ;(global-to-reg B b-cached threadId sizes)
  (define a-cached (create-matrix-local (x-y-z 1)))
  (define b-cached (create-matrix-local (x-y-z 1)))
  (global-to-local A a-cached
                        (x-y-z 1) ;; stride
                        (x-y-z (@dup 0))
                        (x-y-z warpSize)
                        #f)
  (global-to-local B b-cached
                      (x-y-z 1) ;; stride
                      (x-y-z (@dup 0))
                      (x-y-z warpSize)
                      #f)
  
  (define tidx (get-x threadId))
  (define acc1 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc2 (create-accumulator (list bvand bvxor) identity blockDim))

  (for ([i (get-x sizes)])
    (let ([a (shfl (get a-cached (@dup 0)) i)]
          [b (shfl (get b-cached (@dup 0)) (- tidx i))])
      (accumulate acc1 (list a b) #:pred (<= i tidx))))
  
  (for ([i (get-x sizes)])
    (let ([a (shfl (get a-cached (@dup 0)) i)]
          [b (shfl (get b-cached (@dup 0)) (- (- tidx (@dup i)) (@dup 1)) #;(- (+ (get-x sizes) tidx) (+ i 1)))])
      (accumulate acc2 (list a b) #:pred (> i tidx))))

  (reg-to-global acc1 C threadId)
  (reg-to-global acc2 D threadId)
  )

(define my-lane-a1 (gen-lane? 2))
(define my-lane-b1 (gen-lane? 2))
(define my-lane-a2 (gen-lane? 2))
(define my-lane-b2 (gen-lane? 2))
(define (mult-sketch threadId blockID blockDim A B C D sizes)
  (define warpId (get-warpId threadId))
  ;(define a-cached #f)
  ;(define b-cached #f)
  ;(global-to-reg A a-cached threadId sizes)
  ;(global-to-reg B b-cached threadId sizes)
  (define a-cached (create-matrix-local (x-y-z 1)))
  (define b-cached (create-matrix-local (x-y-z 1)))
  (global-to-local A a-cached
                        (x-y-z 1) ;; stride
                        (x-y-z (@dup 0))
                        (x-y-z warpSize)
                        #f)
  (global-to-local B b-cached
                      (x-y-z 1) ;; stride
                      (x-y-z (@dup 0))
                      (x-y-z warpSize)
                      #f)
  #;(global-to-local A a-cached
                        (x-y-z (??)) ;; stride
                        (x-y-z (?warp-offset [(get-x blockID) (get-x blockDim)] [warpId warpSize])) ;; offset
                        (x-y-z (?warp-size warpSize 1)) ;; load size --> TODO: minimize load size
                        #f)
  #;(global-to-local B b-cached
                      (x-y-z (??)) ;; stride
                        (x-y-z (?warp-offset [(get-x blockID) (get-x blockDim)] [warpId warpSize])) ;; offset
                        (x-y-z (?warp-size warpSize 1)) ;; load size --> TODO: minimize load size
                      #f)
  ;(define tidx (get-x threadId))
  (define tidx (get-idInWarp threadId))
  (define acc1 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc2 (create-accumulator (list bvand bvxor) identity blockDim))

  (for ([i warpSize #;(choose warpSize (??))])
    (let* (;[lane-a (?lane tidx (@dup i) [warpSize] 2)]
           ;[lane-b (?lane tidx (@dup i) [warpSize] 2)]
           ;[lane-a (interpret-lane my-lane-a1 (vector tidx (@dup i)) (vector warpSize))]
           ;[lane-b (interpret-lane my-lane-b1 (vector tidx (@dup i)) (vector warpSize))]
           [lane-a (?lane-mod2 (@dup i) tidx [warpSize] 0)]
           [lane-b (?lane-mod2 (@dup i) tidx [warpSize] 0)]
           [a (shfl (get a-cached (@dup 0)) lane-a)]
           [b (shfl (get b-cached (@dup 0)) lane-b)]
          )
      (accumulate acc1 (list a b) #:pred (?cond tidx (@dup i)))))
  
  (for ([i warpSize #;(choose warpSize (??))])
    (let* (;[lane-a (?lane tidx (@dup i) [warpSize] 2)]
           ;[lane-b (?lane tidx (@dup i) [warpSize] 2)]
           ;[lane-a (interpret-lane my-lane-a2 (vector tidx (@dup i)) (vector warpSize))]
           ;[lane-b (interpret-lane my-lane-b2 (vector tidx (@dup i)) (vector warpSize))]
           [lane-a (?lane-mod2 (@dup i) tidx [warpSize] 0)]
           [lane-b (?lane-mod2 (@dup i) tidx [warpSize] 0)]
           [a (shfl (get a-cached (@dup 0)) lane-a)]
           [b (shfl (get b-cached (@dup 0)) lane-b)]
          )
      (accumulate acc2 (list a b) #:pred (?cond tidx (@dup i)))))

  (reg-to-global acc1 C threadId)
  (reg-to-global acc2 D threadId)
  )

(define (mult-sketch-clean threadId blockID blockDim A B C D sizes)
  (define warpId (get-warpId threadId))
  (define a-cached (create-matrix-local (x-y-z 1)))
  (define b-cached (create-matrix-local (x-y-z 1)))
  (global-to-local A a-cached
                        (x-y-z 1) ;; stride
                        (x-y-z (@dup 0))
                        (x-y-z warpSize)
                        #f)
  (global-to-local B b-cached
                      (x-y-z 1) ;; stride
                      (x-y-z (@dup 0))
                      (x-y-z warpSize)
                      #f)
  
  (define tidx (get-idInWarp threadId))
  (define acc1 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc2 (create-accumulator (list bvand bvxor) identity blockDim))

  (for ([i warpSize #;(choose warpSize (??))])
    (let* ([lane-a (?lane-mod2 (@dup i) tidx [warpSize] 0)]
           [lane-b (?lane-mod2 (@dup i) tidx [warpSize] 0)]
           [a (shfl (get a-cached (@dup 0)) lane-a)]
           [b (shfl (get b-cached (@dup 0)) lane-b)]
          )
      (accumulate acc1 (list a b) #:pred (?cond tidx (@dup i)))))
  
  (for ([i warpSize #;(choose warpSize (??))])
    (let* ([lane-a (?lane-mod2 (@dup i) tidx [warpSize] 0)]
           [lane-b (?lane-mod2 (@dup i) tidx [warpSize] 0)]
           [a (shfl (get a-cached (@dup 0)) lane-a)]
           [b (shfl (get b-cached (@dup 0)) lane-b)]
          )
      (accumulate acc2 (list a b) #:pred (?cond tidx (@dup i)))))

  (reg-to-global acc1 C threadId)
  (reg-to-global acc2 D threadId)
  )

(define (test)
  (for ([w (list 4 5 32)])
    (let ([ret (run-with-warp-size mult-spec mult w)])
      (pretty-display `(test ,w ,ret))))
  )
(test)

;; warp size 4, concrete load: 2 s
;; warp size 4 & 5, concrete load: 7 s
;; warp size 4 & 5, synth load: 5/9, 3/30, 3/30 s
;; warp size 32: > 2 hrs
(define (synthesis)
  (pretty-display "solving...")
  (define sol
    (time (solve
           (assert (andmap
                    (lambda (w) (run-with-warp-size mult-spec mult-sketch w))
                    (list 4 5))))))
  (print-forms sol)
  )
(synthesis)

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