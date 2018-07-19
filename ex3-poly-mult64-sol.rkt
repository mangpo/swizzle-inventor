#lang rosette

(require rosette/lib/synthax)
(require "util.rkt" "cuda.rkt" "cuda-synth.rkt")

(define n-block 1)
(define Y_THREADS 1)

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
  (run-kernel kernel (x-y-z w Y_THREADS) (x-y-z 1 n-block) A B C* n)
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

(define (mult32-rc threadId blockID blockDim A B C n)
  (define globalID (+ threadId (* blockID blockDim)))
  (define warpId (get-warpId threadId))
  (define a-cached 0)
  (define b-cached 0)
  (global-to-reg A a-cached globalID #:size (x-y-z n))
  (global-to-reg B b-cached globalID #:size (x-y-z n))
  
  (define tidx (get-idInWarp threadId))
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
      (accumulate acc1 (list a b) #:pred (<= i tidx))
      (accumulate acc2 (list a b) #:pred (> i tidx))
      ))

  (reg-to-global acc1 C globalID #:size (x-y-z (* 2 n)))
  (reg-to-global acc2 C (+ globalID (@dup (x-y-z n 0))) #:size (x-y-z (* 2 n)))
  )

(define (mult32-shared threadId blockID blockDim A B C n)
  (define warpId (get-warpId threadId))
  (define-shared a-cached (create-matrix (x-y-z warpSize Y_THREADS)))
  (define-shared b-cached (create-matrix (x-y-z warpSize Y_THREADS)))
  (define block-offset (* blockID blockDim))
  (define globalID (+ threadId (* blockID blockDim)))
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
           [a (get a-cached lane-a tidy)]
           [b (get b-cached lane-b tidy)]
          )
      (accumulate acc1 (list a b) #:pred (<= i tidx))
      (accumulate acc2 (list a b) #:pred (> i tidx))
      ))

  (reg-to-global acc1 C globalID #:size (x-y-z (* 2 n)))
  (reg-to-global acc2 C (+ globalID (@dup (x-y-z n 0))) #:size (x-y-z (* 2 n)))
  )

(define (mult64-rc threadId blockID blockDim A B C n)
  (define globalID (+ threadId (* blockID blockDim)))
  (define a-cached (create-matrix-local (x-y-z 2 1)))
  (define b-cached (create-matrix-local (x-y-z 2 1)))
  (global-to-local A a-cached
                   (x-y-z 1 1) ;; stride
                   (* (quotient globalID (x-y-z warpSize 1))
                      (x-y-z n 1))
                   (x-y-z n 1)
                   #f #:warp-shape (x-y-z warpSize 1) #:round (x-y-z 2 1) #:size n)
  (global-to-local B b-cached
                   (x-y-z 1 1) ;; stride
                   (* (quotient globalID (x-y-z warpSize 1))
                      (x-y-z n 1))
                   (x-y-z n 1)
                   #f #:warp-shape (x-y-z warpSize 1) #:round (x-y-z 2 1) #:size n)
  
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
           [idx-b1 (ite (< tidx (- warpSize i)) 0 1)]
           [idx-b2 (ite (>= tidx (- warpSize i)) 0 1)]
           [a1 (shfl (get a-cached (@dup 0) (@dup 0)) lane-a1)]
           [a2 (shfl (get a-cached (@dup 1) (@dup 0)) lane-a2)]
           [b1 (shfl (get b-cached idx-b1 (@dup 0)) lane-b1)]
           [b2 (shfl (get b-cached idx-b2 (@dup 0)) lane-b2)]
          )
      (accumulate acc1 (list a1 b1) #:pred (<= i tidx))
      (accumulate acc3 (list a1 b1) #:pred (> i tidx))
      
      (accumulate acc2 (list a1 b2))
      (accumulate acc3 (list a2 b2))
      
      (accumulate acc2 (list a2 b1) #:pred (<= i tidx))
      (accumulate acc4 (list a2 b1) #:pred (> i tidx))
      ))

  (reg-to-global acc1 C globalID #:size (* 2 n))
  (reg-to-global acc2 C (+ globalID (@dup (x-y-z warpSize 0))) #:size (* 2 n))
  (reg-to-global acc3 C (+ globalID (@dup (x-y-z (* 2 warpSize) 0))) #:size (* 2 n))
  (reg-to-global acc4 C (+ globalID (@dup (x-y-z (* 3 warpSize) 0))) #:size (* 2 n))
  )

(define (mult64-shared threadId blockID blockDim A B C n)
  (define block-offset (* (x-y-z 2 1) blockID blockDim))
  (define globalID (+ threadId (* blockID blockDim)))
  (define-shared a-cached (create-matrix (x-y-z n Y_THREADS)))
  (define-shared b-cached (create-matrix (x-y-z n Y_THREADS)))
  (global-to-shared A a-cached
                    (x-y-z 1 1) ;; stride
                    block-offset
                    (* (x-y-z 2 1) blockDim) #f #:round (x-y-z 2 1) #:size n)
  (global-to-shared B b-cached
                    (x-y-z 1 1) ;; stride
                    block-offset
                    (* (x-y-z 2 1) blockDim) #f #:round (x-y-z 2 1) #:size n)
  
  (define tidx (get-idInWarp threadId))
  (define tidy (get-y threadId))
  (define acc1 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc2 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc3 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc4 (create-accumulator (list bvand bvxor) identity blockDim))

  (for ([i warpSize])
    (let* ([lane-a1 (fan tidx n 0 n n 1
                         i warpSize 1 warpSize #:offset 0)]
           [lane-a2 (fan tidx n 0 n n 1
                         i warpSize 1 warpSize #:offset warpSize)]
           [lane-b1 (fan tidx n 1 n n 1
                         i warpSize (- n 1) warpSize #:offset 0)]
           [lane-b2 (fan tidx n 1 n n 1
                         i warpSize -1 warpSize #:offset warpSize)]
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

  (reg-to-global acc1 C globalID #:size (* 2 n))
  (reg-to-global acc2 C (+ globalID (@dup (x-y-z warpSize 0))) #:size (* 2 n))
  (reg-to-global acc3 C (+ globalID (@dup (x-y-z (* 2 warpSize) 0))) #:size (* 2 n))
  (reg-to-global acc4 C (+ globalID (@dup (x-y-z (* 3 warpSize) 0))) #:size (* 2 n))
  )

(define (test)
  (for ([w (list 32)])
    (let ([ret (run-with-warp-size mult-spec mult32-shared w (* 1 w))])
      (pretty-display `(test ,w ,ret))
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
  (assert (andmap
           (lambda (w) (run-with-warp-size mult-spec mult64-shared w (* 2 w)))
           (list 4)))
  (define cost (get-cost))
  (define sol (time (optimize #:minimize (list cost) #:guarantee (assert #t))))

  (define this-cost (evaluate cost sol))
  (print-forms sol)
  (pretty-display `(cost ,this-cost))
  )
;(synthesis)
