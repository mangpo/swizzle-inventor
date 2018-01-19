#lang rosette

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
    (let ([c (create-accumulator o (list bvand bvxor) identity)])
      (for ([i (add1 index)])
        (let ([a (get A i)]
              [b (get B (- index i))])
          (accumulate c (list a b))))
      (set C index c))
    (let ([d (create-accumulator o (list bvand bvxor) identity)])
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
  (define a-cached (create-matrix (x-y-z 1)))
  (define b-cached (create-matrix (x-y-z 1)))
  (global-to-warp-reg A a-cached
                        (x-y-z 1) ;; stride
                        (x-y-z (@dup 0))
                        (x-y-z warpSize)
                        #f)
  (global-to-warp-reg B b-cached
                      (x-y-z 1) ;; stride
                      (x-y-z (@dup 0))
                      (x-y-z warpSize)
                      #f)
  
  (define tidx (get-x threadId))
  (define acc1 (create-accumulator o (list bvand bvxor) identity blockDim))
  (define acc2 (create-accumulator o (list bvand bvxor) identity blockDim))

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

(define (mult-sketch threadId blockID blockDim A B C D sizes)
  (define warpId (get-warpId threadId))
  ;(define a-cached #f)
  ;(define b-cached #f)
  ;(global-to-reg A a-cached threadId sizes)
  ;(global-to-reg B b-cached threadId sizes)
  (define a-cached (create-matrix (x-y-z 1)))
  (define b-cached (create-matrix (x-y-z 1)))
  (global-to-warp-reg A a-cached
                        (x-y-z (??)) ;; stride
                        (x-y-z (?warp-offset [(get-x blockID) (get-x blockDim)] [warpId warpSize])) ;; offset
                        (x-y-z (?warp-size warpSize 1)) ;; load size --> TODO: minimize load size
                        #f)
  (global-to-warp-reg B b-cached
                      (x-y-z (??)) ;; stride
                      (x-y-z (?warp-offset [(get-x blockID) (get-x blockDim)] [warpId warpSize])) ;; offset
                      (x-y-z (?warp-size warpSize 1)) ;; load size
                      #f)
  ;(define tidx (get-x threadId))
  (define tidx (get-idInWarp threadId))
  (define acc1 (create-accumulator o (list bvand bvxor) identity blockDim))
  (define acc2 (create-accumulator o (list bvand bvxor) identity blockDim))

  (for/bounded ([i (choose warpSize (??))])
    (let ([a (shfl (get a-cached (@dup 0)) (?lane tidx (@dup i) [warpSize] 2))]
          [b (shfl (get b-cached (@dup 0)) (?lane tidx (@dup i) [warpSize] 2))]
          )
      (accumulate acc1 (list a b) #:pred (?cond tidx (@dup i)))))
  
  (for/bounded ([i (choose warpSize (??))])
    (let ([a (shfl (get a-cached (@dup 0)) (?lane tidx (@dup i) [warpSize] 2))]
          [b (shfl (get b-cached (@dup 0)) (?lane tidx (@dup i) [warpSize] 2))]
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
;(test)

(define (synthesis)
  (pretty-display "solving...")
  (define sol
    (time (solve
           (assert (andmap
                    (lambda (w) (run-with-warp-size mult-spec mult-sketch w))
                    (list 4 5))))))
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
    (define A-cached (create-matrix (x-y-z a-regs)))
    (define B-cached (create-matrix (x-y-z b-regs)))
    (global-to-warp-reg A A-cached
                        (x-y-z 1) ;; stride
                        (x-y-z (?warp-offset [(get-x blockId) (get-x blockDim)] [warpId warpSize])) ;; offset
                        (x-y-z (?warp-size warpSize 1)) ;; load size --> TODO: minimize load size
                        #f)
    (global-to-warp-reg B B-cached
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
(load-synth)