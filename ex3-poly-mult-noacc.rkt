#lang rosette

(require rosette/lib/synthax)
(require "util.rkt" "cuda.rkt" "cuda-synth.rkt")

(define n-block 1)
(define block-dim-y 1)

(define syms #f)
(define (create-IO warpSize n)
  (set-warpSize warpSize)
  (define A (create-matrix (x-y-z n n-block) gen-bv))
  (define B (create-matrix (x-y-z n n-block) gen-bv))
  (set! syms (append (symbolics A) (symbolics B)))
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
  ;(acc-equal? C C*)
  (for/and ([row C] [row* C*])
    (for/and ([e row] [e* row*])
      (bveq e e*)))
  )

(define (update c a b)
  (bvxor
   c
   (bvand a b)))

(define (mult-spec A B C n rows)
  (for ([row rows])
    (for ([index n])
      (let ([c (bv 0 4)])
        (for ([i (add1 index)])
          (let ([a (get A i row)]
                [b (get B (- index i) row)])
            (set! c (update c a b))))
        (set C index row c))
      (let ([d (bv 0 4)])
        (for ([i (range (add1 index) n)])
          (let ([a (get A i row)]
                [b (get B (- (+ index n) i) row)])
            (set! d (update d a b))))
        (set C (+ n index) row d)))))


(define (mult32 threadId blockID blockDim A B C n)
  (define globalID (+ threadId (* blockID blockDim)))
  (define a-cached 0)
  (define b-cached 0)
  (global-to-reg A a-cached globalID #:size (x-y-z n))
  (global-to-reg B b-cached globalID #:size (x-y-z n))
  
  (define tidx (modulo (get-x threadId) 32))
  (define acc1 (bv 0 4))
  (define acc2 (bv 0 4))

  (for ([i n])
    (let* ([lane-a (fan tidx warpSize 0 warpSize warpSize 1
                        i warpSize 1 warpSize)]
           [lane-b (fan tidx warpSize 1 warpSize warpSize 1
                        i warpSize (- warpSize 1) warpSize)]
           [a (shfl a-cached lane-a)]
           [b (shfl b-cached lane-b)]
          )
      (set! acc1 (ite (<= (@dup i) tidx) (update acc1 a b) acc1))
      (set! acc2 (ite (> (@dup i) tidx) (update acc2 a b) acc2))
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
  (define acc1 (bv 0 4))
  (define acc2 (bv 0 4))

  (for ([i n])
    (let* (;[lane-a (?fan-easy tidx warpSize i warpSize [])]
           ;[lane-b (?fan-easy tidx warpSize i warpSize [])]
           [lane-a (?lane-mod tidx (@dup i) 2 n [warpSize])]
           [lane-b (?lane-mod tidx (@dup i) 2 n [warpSize])]
           [a (shfl a-cached lane-a)]
           [b (shfl b-cached lane-b)]
          )
      (set! acc1 (ite (<= (@dup i) tidx) (update acc1 a b) acc1))
      (set! acc2 (ite (> (@dup i) tidx) (update acc2 a b) acc2))
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
  (define acc1 (bv 0 4))
  (define acc2 (bv 0 4))

  (for ([i n])
    (let* (;[lane-a (?fan-easy tidx warpSize i warpSize [])]
           ;[lane-b (?fan-easy tidx warpSize i warpSize [])]
           [lane-a (?lane-mod tidx (@dup i) 2 n [warpSize])]
           [lane-b (?lane-mod tidx (@dup i) 2 n [warpSize])]
           [a (get a-cached lane-a tidy)]
           [b (get b-cached lane-b tidy)]
          )
      (set! acc1 (ite (<= (@dup i) tidx) (update acc1 a b) acc1))
      (set! acc2 (ite (> (@dup i) tidx) (update acc2 a b) acc2))
      ))

  (reg-to-global acc1 C globalID #:size (x-y-z (* 2 n)))
  (reg-to-global acc2 C (+ globalID (@dup (x-y-z n 0))) #:size (x-y-z (* 2 n)))
  )


(define (test)
  (for ([w (list 32)])
    (let ([ret (run-with-warp-size mult-spec mult32 w (* 1 w))])
      (pretty-display `(test ,w ,ret))
      (pretty-display `(cost ,(get-cost)))
      ))
  )
;(test)

(define (synthesis)
  (pretty-display "solving...")
  (define t (andmap
             (lambda (w) (run-with-warp-size mult-spec mult32-shared-sketch w (* 1 w)))
             (list 4)))
  ;(define cost (get-cost))
  (define sol (time (synthesize #:forall syms
                                #:guarantee (assert t))))

  ;(define this-cost (evaluate cost sol))
  (print-forms sol)
  ;(pretty-display `(cost ,this-cost))
  )
(define t0 (current-seconds))
(synthesis)
(define t1 (current-seconds))
(- t1 t0)
