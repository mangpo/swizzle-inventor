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

(require "util.rkt" "cuda.rkt" "cuda-synth.rkt")

(define struct-size 3)
(define n-block 1)



(define (create-IO warpSize)
  (set-warpSize warpSize)
  (define block-size (* 2 warpSize))
  (define array-size (* n-block block-size))
  (define I-sizes (x-y-z (* array-size struct-size)))
  (define I (create-matrix I-sizes gen-uid))
  (define O (create-matrix I-sizes))
  (define O* (create-matrix I-sizes))
  (values block-size I-sizes I O O*))

(define (run-with-warp-size spec kernel w)
  (define-values (block-size I-sizes I O O*)
  (create-IO w))

  (define c (gcd struct-size warpSize))
  (define a (/ struct-size c))
  (define b (/ warpSize c))

  (run-kernel spec (x-y-z block-size) (x-y-z n-block) I O a b c)
  (run-kernel kernel (x-y-z block-size) (x-y-z n-block) I O* a b c)
  (define ret (equal? O O*))
  ;(pretty-display `(O ,O))
  ;(pretty-display `(O* ,O*))
  ret)

(define (AOS-load-spec threadId blockID blockDim I O a b c)
  (define I-cached (create-matrix-local (x-y-z struct-size)))
  (define warpID (get-warpId threadId))
  (define offset (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))  ;; warpID = (threadIdy * blockDimx + threadIdx)/warpSize
  (define gid (get-global-threadId threadId blockID))
  (global-to-local I I-cached
                 (x-y-z struct-size)
                 offset (x-y-z (* warpSize struct-size)) #f)
  (local-to-global I-cached O
                      (x-y-z 1) offset (x-y-z (* warpSize struct-size)) #f #:round struct-size)
  )

(define (print-vec x)
  (format "#(~a)" (string-join (for/list ([xi x]) (format "~a" xi)))))

;; struct size = 4
;; warpSize 2 4: > 1 hr
;; warpSize 2 4, bv, constrain row permute: 43/596 s
;; warpSize 4 8, bv, constrain row permute: 287/9840 s (still wrong for 32)
;; warpSize 32, bv, constrain row permute, depth 4/4/2: 76/1681 s

;; warpSize 32, depth 3/4/2: 142/4661 s
;; warpSize 32, depth 3/4/2, constrain row permute: 500/2817 s
;; warpSize 32, depth 3/4/2, constraint col+row permute, distinct?: 89/2518 s
;; warpSize 32, depth 3/4/2, constraint col+row permute, distinct?, ref: > 8 h

;; struct-size 2, warpSize 32, depth 3/4/2, constraint col+row permute, distinct?: 27/266 s
;; struct-size 3, warpSize 32, depth 3/4/2, constraint col+row permute, distinct?: 59/410 s
;; struct-size 4, warpSize 32, depth 3/4/2, constraint col+row permute, distinct?: 89/2518 s
;; struct-size 4, warpSize 8, depth 3/4/2, constraint col+row permute, distinct?: 57/777 s

;; Ras's sketch
;; struct-size 4, warpSize 8, constraint col+row permute, distinct?: 1/3 s
;; struct-size 2, warpSize 32, constraint col+row permute, distinct?: 2/9 s
;; struct-size 3, warpSize 32, constraint col+row permute, distinct?: 2/14 s
;; struct-size 4, warpSize 32, constraint col+row permute, distinct?: 4/42 s
;; ^ if include choose 0 in lane-mod, > 5 mins
;; struct-size 5, warpSize 32, constraint col+row permute, distinct?: 4/39 s | 242/484 s
;; struct-size 6, warpSize 32, constraint col+row permute, distinct?: unsat
;; struct-size 7, warpSize 32, constraint col+row permute, distinct?: 8/156 s
;; struct-size 8, warpSize 32, constraint col+row permute, distinct?: 6/43 s
(define (AOS-load-sketch threadId blockID blockDim I O a b c)
  #|
  (define log-a (bvlog a))
  (define log-b (bvlog b))
  (define log-c (bvlog c))
  (define log-m (bvlog struct-size))
  (define log-n (bvlog warpSize))
|#
  
  (define I-cached (create-matrix-local (x-y-z struct-size)))
  (define O-cached (create-matrix-local (x-y-z struct-size)))
  (define warpID (get-warpId threadId))
  (define offset (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))  ;; warpID = (threadIdy * blockDimx + threadIdx)/warpSize
  (define gid (get-global-threadId threadId blockID))
  (global-to-local I I-cached
                 (x-y-z 1)
                 ;;(x-y-z (?warp-offset [(get-x blockID) (get-x blockDim)] [warpID warpSize]))
                 offset
                 (x-y-z (* warpSize struct-size)) #f)

  (define indices (make-vector struct-size))
  (define indices-o (make-vector struct-size))
  (define localId (get-idInWarp threadId))
  (for ([i struct-size])
    (let* (;[index (modulo (?lane localId (@dup i) [a b c struct-size warpSize] 3) struct-size)]
           ;[lane (?lane localId (@dup i) [a b c struct-size warpSize] 4)]
           ;[index (@int (extract (?lane-log-bv (@bv localId) (@dup (@bv i)) [2 3 4 5] 4) log-m))]
           ;[lane (@int (?lane-log-bv (@bv localId) (@dup (@bv i)) [2 3 4 5] 4))]
           [index (?lane-mod2 (@dup i) localId [a b c struct-size warpSize] 0)]
           [lane (?lane-mod2 (@dup i) localId [a b c struct-size warpSize] 1)]
           #;[inter (modulo (+ (* (@dup i) (?const a b c struct-size warpSize)) (* localId (?const a b c struct-size warpSize))
                               (quotient (@dup i) (?const a b c struct-size warpSize)) (quotient localId (?const a b c struct-size warpSize))
                               (?const a b c struct-size warpSize))
                            (?const a b c struct-size warpSize))]
           #;[index (modulo (+ (* (@dup i) (?const a b c struct-size warpSize)) (* localId (?const a b c struct-size warpSize)) (* inter (?const a b c struct-size warpSize))
                            (quotient (@dup i) (?const a b c struct-size warpSize)) (quotient localId (?const a b c struct-size warpSize)) (quotient inter (?const a b c struct-size warpSize))
                            (?const a b c struct-size warpSize))
                         struct-size)]

           #;[lane (+ (modulo (+ (* (@dup i) (?const a b c struct-size warpSize)) (* localId (?const a b c struct-size warpSize))
                               (quotient (@dup i) (?const a b c struct-size warpSize)) (quotient localId (?const a b c struct-size warpSize))
                               (?const a b c struct-size warpSize))
                            (?const a b c struct-size warpSize))
                    (modulo (+ (* (@dup i) (?const a b c struct-size warpSize)) (* localId (?const a b c struct-size warpSize))
                               (quotient (@dup i) (?const a b c struct-size warpSize)) (quotient localId (?const a b c struct-size warpSize))
                               (?const a b c struct-size warpSize))
                            (?const a b c struct-size warpSize))
                    )]
           [x (shfl (get I-cached index) lane)]
           ;[index-o (modulo (?index localId (@dup i) [a b c struct-size warpSize] 2) struct-size)]
           ;[index-o (@int (extract (?lane-log-bv (@bv localId) (@dup (@bv i)) [2 3 4 5] 2) log-m))]
           [index-o (?lane-mod2 (@dup i) localId [a b c struct-size warpSize] 0)]
           #;[index-o (modulo (+ (* (@dup i) (?const a b c struct-size warpSize)) (* localId (?const a b c struct-size warpSize))
                             (quotient (@dup i) (?const a b c struct-size warpSize)) (quotient localId (?const a b c struct-size warpSize))
                             (?const a b c struct-size warpSize))
                          struct-size)]
           )
      (unique-warp (modulo lane warpSize))
      (pretty-display `(i ,i))
      (vector-set! indices i index)
      (vector-set! indices-o i index-o)
      (set O-cached index-o x))
    )
  (for ([t blockSize])
    (let ([l (for/list ([i struct-size]) (vector-ref (vector-ref indices i) t))]
          [lo (for/list ([i struct-size]) (vector-ref (vector-ref indices-o i) t))])
      (unique-list l)
      (unique-list lo)))
  
  (local-to-global O-cached O
                      (x-y-z 1)
                      ;;(x-y-z (?warp-offset [(get-x blockID) (get-x blockDim)] [warpID warpSize]))
                      offset
                      (x-y-z (* warpSize struct-size)) #f)
  )


;; newest fan sketch
;; 2: 1/10 | 2/16
;; 3: 3/17 | 63/191
;; 4:      | 33/49
;; 5:      | 16/129


;; BW=10, ?fan-easy, fw=1
;; 3: 1/2 s
(define (AOS-load-sketch-fan threadId blockID blockDim I O a b c)
  (define I-cached (create-matrix-local (x-y-z struct-size)))
  (define O-cached (create-matrix-local (x-y-z struct-size)))
  
  (define localId (modulo (get-x threadId) 32))
  (define offset (* struct-size (- (+ (* blockID blockDim) (get-x threadId)) localId)))
  
  (global-to-local I I-cached
                 (x-y-z 1)
                 offset
                 (x-y-z (* warpSize struct-size)) #f #:round struct-size)

  (define I-cached2 (permute-vector I-cached struct-size
                                    (lambda (i) (?fan-easy i struct-size localId warpSize #:fw 1))))
  
  (for ([i struct-size])
    (let* ([lane (?fan-easy localId warpSize i struct-size #:fw 1)]
           [x (shfl (get I-cached2 (@dup i)) lane)]
           )
      (set O-cached (@dup i) x))
    )
  

  (define O-cached2 (permute-vector O-cached struct-size
                                    (lambda (i) (?fan-easy i struct-size localId warpSize #:fw 1))))
  
  (local-to-global O-cached2 O
                      (x-y-z 1)
                      offset
                      (x-y-z (* warpSize struct-size)) #f #:round struct-size)
  )

;; BW=8, const=??
;; 2: 6/39
;; 3: 6/30
;; 4: 8/116
;; 5: 35/110

;; BW=10, ?fan-easy, fw=1, const=??
;; 3: 8/13 s
(define (AOS-loadsh-sketch-fan threadId blockID blockDim I O a b c)
  (define I-cached (create-matrix-local (x-y-z struct-size)))
  
  (define localId (modulo (get-x threadId) 32))
  (define offset (* struct-size (- (+ (* blockID blockDim) (get-x threadId)) localId)))
  
  (global-to-local
   I
   I-cached
   (x-y-z 1)
   offset
   (x-y-z (* warpSize struct-size))
   #f #:round struct-size
   #:shfl (lambda (localId i) (?fan-easy localId warpSize i struct-size #:fw 1)))
  
  (define O-cached (permute-vector I-cached struct-size
                                   (lambda (i) (?fan-easy i struct-size localId warpSize #:fw 1))))
  (local-to-global
   O-cached
   O
   (x-y-z 1)
   offset
   (x-y-z (* warpSize struct-size))
   #f #:round struct-size
   #:shfl (lambda (localId i)
            (?fan-easy localId warpSize i struct-size #:fw 1)))
  )

(define (test)
  (for ([w (list 32)])
    (let ([ret (run-with-warp-size AOS-load-spec AOS-loadsh-sketch-fan w)])
      (pretty-display `(test ,w ,ret))))
  )
;(test)

(define (synthesis)
  (pretty-display "solving...")
  (assert (andmap (lambda (w) (run-with-warp-size AOS-load-spec AOS-loadsh-sketch-fan w))
                                           (list 32)))
  (define cost (get-cost))
  
  (define sol (time (optimize #:minimize (list cost) #:guarantee (assert #t))))

  (define this-cost (evaluate cost sol))
  (print-forms sol)
  (pretty-display `(cost ,this-cost))
  )
(synthesis)

