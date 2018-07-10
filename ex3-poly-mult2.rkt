#lang rosette

(require rosette/lib/synthax)
(require "util.rkt" "cuda.rkt" "cuda-synth.rkt")

(define n-block 1)

(define (create-IO warpSize n)
  (set-warpSize warpSize)
  (define block-size warpSize)
  (define sizes (x-y-z n))
  (define A (create-matrix sizes gen-uid))
  (define B (create-matrix sizes gen-uid))
  (define C (create-matrix (* 2 sizes)))
  (define C* (create-matrix (* 2 sizes)))
  (values block-size sizes A B C C*))

(define (run-with-warp-size spec kernel w n)
  (define-values (block-size sizes A B C C*)
    (create-IO w n))

  (spec A B C sizes)
  (run-kernel kernel (x-y-z block-size) (x-y-z n-block) A B C* sizes)
  ;(pretty-display ">>> C")
  ;(acc-print C)
  ;(pretty-display ">>> C*")
  ;(acc-print C*)
  (acc-equal? C C*))

(define (mult-spec A B C sizes)
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
              [b (get B (- (+ index (get-x sizes)) i))])
          (accumulate d (list a b))))
      (set C (+ (get-x sizes) index) d))))

(define (mult threadId blockID blockDim A B C sizes)
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
  (reg-to-global acc2 C (+ sizes threadId))
  )

(define my-lane-a1 (gen-lane? 2))
(define my-lane-b1 (gen-lane? 2))
(define my-lane-a2 (gen-lane? 2))
(define my-lane-b2 (gen-lane? 2))
(define (mult-sketch threadId blockID blockDim A B C sizes)
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
  (reg-to-global acc2 C (+ sizes threadId))
  )

;; conf-fw options: 740/1172
;; conf-fw=1: 28/503
(define (mult32-sketch threadId blockID blockDim A B C sizes)
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
    (let* (#;[lane-a (fan tidx warpSize 0 warpSize warpSize 1
                        i warpSize 1 warpSize)]
           #;[lane-b (fan tidx warpSize 1 warpSize warpSize 1
                        i warpSize (- warpSize 1) warpSize)]
           [lane-a (fan tidx warpSize (??) (??) (??) 1
                        i warpSize (??) (??))]
           [lane-b (fan tidx warpSize (??) (??) (??) 1
                        i warpSize (??) (??))]
           [a (shfl (get a-cached (@dup 0)) lane-a)]
           [b (shfl (get b-cached (@dup 0)) lane-b)]
          )
      (accumulate acc1 (list a b) #:pred (?cond tidx (@dup i)))
      (accumulate acc2 (list a b) #:pred (?cond tidx (@dup i)))
      ))

  (reg-to-global acc1 C threadId)
  (reg-to-global acc2 C (+ sizes threadId))
  )

(define (mult64 threadId blockID blockDim A B C sizes)
  (define warpId (get-warpId threadId))
  (define a-cached (create-matrix-local (x-y-z 2)))
  (define b-cached (create-matrix-local (x-y-z 2)))
  (global-to-local A a-cached
                   (x-y-z 1) ;; stride
                   (x-y-z (@dup 0))
                   sizes
                   #f)
  (global-to-local B b-cached
                   (x-y-z 1) ;; stride
                   (x-y-z (@dup 0))
                   sizes
                   #f)
  
  (define tidx (get-idInWarp threadId))
  (define acc1 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc2 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc3 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc4 (create-accumulator (list bvand bvxor) identity blockDim))

  (for ([i warpSize #;(choose warpSize (??))])
    (let* ([lane-a1 (fan tidx warpSize 0 warpSize warpSize 1
                        i warpSize 1 warpSize)]
           [lane-a2 (fan tidx warpSize 0 warpSize warpSize 1
                        i warpSize 1 warpSize)]
           [lane-b1 (fan tidx warpSize 1 warpSize warpSize 1
                        i warpSize (- warpSize 1) warpSize)]
           [lane-b2 (fan tidx warpSize 1 warpSize warpSize 1
                        i warpSize (- warpSize 1) warpSize)]
           [a1 (shfl (get a-cached (@dup 0)) lane-a1)]
           [a2 (shfl (get a-cached (@dup 1)) lane-a2)]
           [b1 (shfl (get b-cached (@dup 0)) lane-b1)]
           [b2 (shfl (get b-cached (@dup 1)) lane-b2)]
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

  (reg-to-global acc1 C threadId)
  (reg-to-global acc2 C (+ warpSize threadId))
  (reg-to-global acc3 C (+ (* 2 warpSize) threadId))
  (reg-to-global acc4 C (+ (* 3 warpSize) threadId))
  )

(define (mult64-sketch threadId blockID blockDim A B C sizes)
  (define warpId (get-warpId threadId))
  (define a-cached (create-matrix-local (x-y-z 2)))
  (define b-cached (create-matrix-local (x-y-z 2)))
  (global-to-local A a-cached
                   (x-y-z 1) ;; stride
                   (x-y-z (@dup 0))
                   sizes
                   #f)
  (global-to-local B b-cached
                   (x-y-z 1) ;; stride
                   (x-y-z (@dup 0))
                   sizes
                   #f)
  
  (define tidx (get-idInWarp threadId))
  (define acc1 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc2 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc3 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc4 (create-accumulator (list bvand bvxor) identity blockDim))

  (for ([i warpSize #;(choose warpSize (??))])
    (let* ([lane-a1 (?fan tidx warpSize
                          i warpSize #:fw 1)]
           [lane-a2 (?fan tidx warpSize
                          i warpSize #:fw 1)]
           [lane-b1 (?fan tidx warpSize
                          i warpSize #:fw 1)]
           [lane-b2 (?fan tidx warpSize
                          i warpSize #:fw 1)]
           [a1 (shfl (get a-cached (@dup 0)) lane-a1)]
           [a2 (shfl (get a-cached (@dup 1)) lane-a2)]
           [b1 (shfl (get b-cached (@dup 0)) lane-b1)]
           [b2 (shfl (get b-cached (@dup 1)) lane-b2)]
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

  (reg-to-global acc1 C threadId)
  (reg-to-global acc2 C (+ warpSize threadId))
  (reg-to-global acc3 C (+ (* 2 warpSize) threadId))
  (reg-to-global acc4 C (+ (* 3 warpSize) threadId))
  )

(define (test)
  (for ([n (list 8)])
    (let ([ret (run-with-warp-size mult-spec mult64-sketch 4 n)])
      (pretty-display `(test ,n ,ret))
      (pretty-display `(cost ,(get-cost)))
      ))
  )
(test)

;; warp size 4, concrete load: 2 s
;; warp size 4 & 5, concrete load: 7 s
;; warp size 4 & 5, synth load: 5/9 s
;; warp size 32: 44/776 s
(define (synthesis)
  (pretty-display "solving...")
  (define sol
    (time (solve
           (assert (andmap
                    (lambda (n) (run-with-warp-size mult-spec mult64-sketch 32 n))
                    (list 64))))))
  (print-forms sol)
  )
;(synthesis)

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