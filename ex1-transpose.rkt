#lang rosette

(require "util.rkt" "cuda.rkt")

(define (transpose-spec I O sizes)
  (for* ([y (get-y sizes)]
         [x (get-x sizes)])
    (set O y x (get I x y))))

(define sizes (x-y-z 4 4))
(define I (create-matrix sizes
                         (lambda () (define-symbolic* x integer?) x)))
(define O (create-matrix (reverse sizes)))
(define O* (create-matrix (reverse sizes)))

(transpose-spec I O sizes)

(define (transpose1 threadId blockID blockDim I O)
  (define-shared I-shared (create-matrix (reverse blockDim)))
  (define offset (* blockID blockDim))
  (global-to-shared I I-shared
                    (x-y-z 1 1)
                    offset blockDim
                    #:transpose #t)
  (shared-to-global I-shared O
                    (x-y-z 1 1)
                    (reverse offset) (reverse blockDim))
  )

(define (transpose2 threadId blockID blockDim I O)
  (define tileDim (x-y-z 4 4))
  (define-shared I-shared (create-matrix (reverse tileDim)))
  (define offset (* blockID tileDim))
  (global-to-shared I I-shared
                    (x-y-z 1 1)
                    offset tileDim
                    #:transpose #t)
  (shared-to-global I-shared O
                    (x-y-z 1 1)
                    (reverse offset) (reverse tileDim))
  )

;;(run-kernel transpose1 (x-y-z 2 2) (x-y-z 2 2) I O*)
(run-kernel transpose2 (x-y-z 4 1) (x-y-z 1 1) I O*)
(verify #:guarantee (assert (equal? O O*)))
