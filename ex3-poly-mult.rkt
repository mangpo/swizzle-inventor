#lang rosette

(require "util.rkt" "cuda.rkt" "cuda-synth.rkt")

(define BW 4)

(define sizes (x-y-z 4))
;(define A (create-matrix sizes (lambda () (define-symbolic* a integer?) a)))
;(define B (create-matrix sizes (lambda () (define-symbolic* b integer?) b)))
(define A (create-matrix sizes gen-uid))
(define B (create-matrix sizes gen-uid))
(define C (create-matrix sizes))
(define D (create-matrix sizes))
(define C* (create-matrix sizes))
(define D* (create-matrix sizes))

(define (mult-spec A B C D)
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

(define (mult threadId blockID blockDim A B C D)
  (define a-cached #f)
  (define b-cached #f)
  (global-to-reg A a-cached threadId sizes)
  (global-to-reg B b-cached threadId sizes)
  (define tidx (get-x threadId))
  (define acc1 (create-accumulator o (list bvand bvxor) identity blockDim))
  (define acc2 (create-accumulator o (list bvand bvxor) identity blockDim))

  (for ([i (get-x sizes)])
    (let ([a (shfl a-cached i)]
          [b (shfl b-cached (- tidx i))])
      (accumulate acc1 (list a b) #:pred (<= i tidx))))
  
  (for ([i (get-x sizes)])
    (let ([a (shfl a-cached i)]
          [b (shfl b-cached (- (- tidx (@dup i)) (@dup 1)) #;(- (+ (get-x sizes) tidx) (+ i 1)))])
      (accumulate acc2 (list a b) #:pred (> i tidx))))

  (reg-to-global acc1 C threadId sizes)
  (reg-to-global acc2 D threadId sizes)
  )

(define (mult-test threadId blockID blockDim A B C D)
   (define a-cached #f)
   (define b-cached #f)
   (global-to-reg A a-cached threadId sizes)
   (global-to-reg B b-cached threadId sizes)
   (define tidx (get-idInWarp threadId))
   (define acc1 (create-accumulator o (list bvand bvxor) identity blockDim))
   (define acc2 (create-accumulator o (list bvand bvxor) identity blockDim))
   (for
    ((i (get-x sizes)))
    (let ((a (shfl a-cached (- (@dup i) (- (@dup 1) (@dup 1)))))
          (b (shfl b-cached (+ (+ (@dup i) tidx) (+ (@dup i) (@dup i))))))
      (accumulate acc1 (list a b) #:pred (>= tidx (@dup i)))))
   (for
    ((i (get-x sizes)))
    (let ((a (shfl a-cached (@dup i)))
          (b (shfl b-cached (- tidx (+ (@dup 1) (@dup i))))))
      (accumulate acc2 (list a b) #:pred (< tidx (@dup i)))))
   (reg-to-global acc1 C threadId sizes)
   (reg-to-global acc2 D threadId sizes))

(define (mult-sketch threadId blockID blockDim A B C D)
  (define a-cached #f)
  (define b-cached #f)
  (global-to-reg A a-cached threadId sizes)
  (global-to-reg B b-cached threadId sizes)
  ;(define tidx (get-x threadId))
  (define tidx (get-idInWarp threadId))
  (define acc1 (create-accumulator o (list bvand bvxor) identity blockDim))
  (define acc2 (create-accumulator o (list bvand bvxor) identity blockDim))

  (for ([i (get-x sizes)])
    (let ([a (shfl a-cached (?lane tidx (@dup i) 2))]
          [b (shfl b-cached (?lane tidx (@dup i) 2))]
          )
      (accumulate acc1 (list a b) #:pred (?cond tidx (@dup i)))))
  
  (for ([i (get-x sizes)])
    (let ([a (shfl a-cached (?lane tidx (@dup i) 2))]
          [b (shfl b-cached (?lane tidx (@dup i) 2))]
          )
      (accumulate acc2 (list a b) #:pred (?cond tidx (@dup i)))))

  (reg-to-global acc1 C threadId sizes)
  (reg-to-global acc2 D threadId sizes)
  )

(mult-spec A B C D)
(for ([c C]) (pretty-display `(c ,(get-accumulator-val c))))
(for ([d D]) (pretty-display `(d ,(get-accumulator-val d))))

(define (validate)
  (run-kernel mult-test sizes (x-y-z 1) A B C* D*)
  (for ([c C*]) (pretty-display `(c ,(get-accumulator-val c))))
  (for ([d D*]) (pretty-display `(d ,(get-accumulator-val d))))
  
  (pretty-display `(C-equal? ,(acc-equal? C C*)))
  (pretty-display `(D-equal? ,(acc-equal? D D*)))
  ;(verify #:guarantee (assert (and (acc-equal? C C*) (acc-equal? D D*))))
  )
;(validate)

(define (synthesis)
  (run-kernel mult-sketch sizes (x-y-z 1) A B C* D*)
  (pretty-display "solving...")
  
  ;; z3 is actaully super fast
  ;; depth 0 1 0 2 --> 9s  (sym) | 2s (conc)
  ;; depth 2 --> 38s (sym) | 9s (conc)
  (define sol
    (time
     (synthesize
      #:forall (symbolics (list A B))
      #:guarantee (assert (and (acc-equal? C C*) (acc-equal? D D*))))))
  (print-forms sol)
  )
(synthesis)