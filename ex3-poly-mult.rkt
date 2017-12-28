#lang rosette

(require "util.rkt" "cuda.rkt")

(define BW 4)

(define sizes (x-y-z 4))
(define A (create-matrix sizes (lambda () (define-symbolic* a integer?) a)))
(define B (create-matrix sizes (lambda () (define-symbolic* b integer?) b)))
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
          [b (shfl b-cached (- (+ (get-x sizes) tidx) (+ i 1)))])
      (accumulate acc2 (list a b) #:pred (> i tidx))))

  (reg-to-global acc1 C threadId sizes)
  (reg-to-global acc2 D threadId sizes)
  )

(mult-spec A B C D)
(for ([c C]) (pretty-display `(c ,(get-accumulator-val c))))
(for ([d D]) (pretty-display `(d ,(get-accumulator-val d))))

(run-kernel mult sizes (x-y-z 1) A B C* D*)
(for ([c C*]) (pretty-display `(c ,(get-accumulator-val c))))
(for ([d D*]) (pretty-display `(d ,(get-accumulator-val d))))

(pretty-display `(C-equal? ,(equal? C C*)))
(pretty-display `(D-equal? ,(equal? D D*)))