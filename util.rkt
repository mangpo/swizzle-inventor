#lang rosette

(provide (all-defined-out)) 

;; dims: x y z
(define (create-matrix dims [init (lambda () 0)])
  (define (f dims)
    (if (empty? dims)
        (init)
        (let ([vec (make-vector (car dims))])
          (for ([i (car dims)])
            (vector-set! vec i (f (cdr dims))))
          vec)))
  (f (reverse dims)))

(define (my-vector-ref vec index)
  (cond
    [(and (vector? index)
          (= (vector-length vec) (vector-length index))
          (vector? (vector-ref vec 0)))
     (for/vector ([vec-i vec] [index-i index]) (vector-ref vec-i index-i))]

    [(vector? index)
     (for/vector ([index-i index]) (vector-ref vec index-i))]

    [else (vector-ref vec index)]))

(define (my-vector-set! vec index val)
  (when (and (vector? index) (vector? val))
    (assert (= (vector-length vec) (vector-length val)) `(= (vector-length vec) (vector-length val))))
  (cond
    [(and (vector? index)
          (= (vector-length vec) (vector-length index))
          (vector? (vector-ref vec 0)))
     (if (vector? val)
         (for/vector ([vec-i vec] [index-i index] [val-i val]) (vector-set! vec-i index-i val-i))
         (for/vector ([vec-i vec] [index-i index]) (vector-set! vec-i index-i val)))
     ]

    [(vector? index)
     (if (vector? val)
         (for/vector ([vec-i vec] [index-i index] [val-i val]) (vector-set! vec-i index-i val-i))
         (for/vector ([vec-i vec] [index-i index]) (vector-set! vec-i index-i val)))]

    [else
     (vector-set! vec index val)]))

(define-syntax get
  (syntax-rules ()
    ((get M i)
     (my-vector-ref M i))
    ((get M i ... j)
     (get (my-vector-ref M j) i ...))))

(define-syntax set
  (syntax-rules ()
    ((set M i v) (my-vector-set! M i v))
    ((set M i ... j v) (set (my-vector-ref M j) i ... v))))


(define-syntax-rule (get-x l) (first l))
(define-syntax-rule (get-y l) (second l))
(define-syntax-rule (get-z l) (third l))
(define-syntax x-y-z
  (syntax-rules ()
    ((x-y-z x) (list x))
    ((x-y-z x y) (list x y))
    ((x-y-z x y z) (list x y z))))

(define (global-threadID threadID blockID blockDIM)
  (map (lambda (tid bid dim) (+ tid (* bid dim))) threadID blockID blockDIM))


(define (clone x)
  (cond
    [(vector? x) (for/vector ([xi x]) (clone xi))]
    [else x]))
