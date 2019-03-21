#|
 | Copyright (c) 2018-2019, University of California, Berkeley.
 | Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

(provide (all-defined-out))

(define BW 10)
(current-bitwidth BW)

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

(define (get-dims M)
  (define (f M)
    (if (vector? M)
        (cons (vector-length M) (f (vector-ref M 0)))
        (list)))
  (reverse (f M)))

(define (my-vector-ref vec index)
  (cond
    [(and (vector? index)
          (= (vector-length vec) (vector-length index))
          (vector? (vector-ref vec 0)))
     (for*/all ([my-vec vec] [my-index index])
         (for/vector ([vec-i my-vec] [index-i my-index]) (vector-ref vec-i index-i)))]

    [(vector? index)
     (for*/all ([my-vec vec] [my-index index])
       (for/vector ([index-i my-index]) (vector-ref my-vec index-i)))]

    [else
     (for*/all ([my-vec vec] [my-index index])
       (vector-ref my-vec my-index))]))

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

(define (get* M l)
  (define (f M l)
    (if (empty? l)
        M
        (f (vector-ref M (car l)) (cdr l))))
  (f M (reverse l)))

(define (set* M l v)
  (define (f M l)
    (if (= (length l) 1)
        (vector-set! M (car l) v)
        (f (vector-ref M (car l)) (cdr l))))
  (f M (reverse l)))

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

(define (my-list-ref l index)
  (if (vector? l)
      (for/vector ([x l]) (my-list-ref x index))
      (list-ref l index)))

(define-syntax-rule (get-x l) (my-list-ref l 0))
(define-syntax-rule (get-y l) (my-list-ref l 1))
(define-syntax-rule (get-z l) (my-list-ref l 2))
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

(define (lov2vol x)
  (define vec-len (vector-length (car x)))
  (define list-len (length x))

  (for/vector ([vi vec-len])
    (for/list ([li list-len])
      (vector-ref (list-ref x li) vi))))

(define (print-vec x)
  (format "#(~a)" (string-join (for/list ([xi x]) (format "~a" xi)))))
