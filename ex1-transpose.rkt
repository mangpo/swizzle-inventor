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
                    offset tileDim #t
                    #:round (x-y-z 1 4) #:size sizes)
  (shared-to-global I-shared O
                    (x-y-z 1 1)
                    (reverse offset) (reverse tileDim)
                    #:round (x-y-z 1 4) #:size sizes)
  )

;;(run-kernel transpose1 (x-y-z 2 2) (x-y-z 3 3) I O*)
(run-kernel transpose2 (x-y-z 4 1) (x-y-z 2 2) I O*)
(pretty-display `(O* ,O*))
(verify #:guarantee (assert (equal? O O*)))
