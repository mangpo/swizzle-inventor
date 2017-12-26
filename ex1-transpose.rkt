#lang rosette

(require "util.rkt" "cuda.rkt")

(define (transpose-spec I O sizes)
  (for* ([y (get-y sizes)]
         [x (get-x sizes)])
    (set O y x (get I x y))))

(define sizes (x-y-z 5 5))
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
                    #:bounds (x-y-z 5 5)
                    #:transpose #t)
  (shared-to-global I-shared O
                    (x-y-z 1 1)
                    (reverse offset) (reverse blockDim)
                    #:bounds (x-y-z 5 5))
  )

(define (transpose2 threadId blockID blockDim I O)
  (define tileDim (x-y-z 4 4))
  (define-shared I-shared (create-matrix (reverse tileDim)))
  (define offset (* blockID tileDim))
  (global-to-shared I I-shared
                    (x-y-z 1 1)
                    offset tileDim
                    #:bounds (x-y-z 5 5)
                    #:transpose #t)
  (shared-to-global I-shared O
                    (x-y-z 1 1)
                    (reverse offset) (reverse tileDim)
                    #:bounds (x-y-z 5 5))
  )

;;(run-kernel transpose1 (x-y-z 2 2) (x-y-z 3 3) I O*)
(run-kernel transpose2 (x-y-z 4 1) (x-y-z 2 2) I O*)
(pretty-display `(O* ,O*))
;;(verify #:guarantee (assert (equal? O O*)))
