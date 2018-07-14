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
  (pretty-display `(O ,(print-vec O)))
  (pretty-display `(O* ,(print-vec O*)))
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

(define (AOS-loadsh4 threadId blockID blockDim I O a b c)
   (define I-cached (create-matrix-local (x-y-z struct-size)))
   (define temp (create-matrix-local (x-y-z struct-size)))
   (define O-cached (create-matrix-local (x-y-z struct-size)))
   (define warpID (get-warpId threadId))
   (define offset
     (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))
   (define gid (get-global-threadId threadId blockID))
   (global-to-local
    I
    I-cached
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f)
   (define indices (make-vector struct-size))
   (define indices-o (make-vector struct-size))
   (define localId (get-idInWarp threadId))
   (for
    ((i struct-size))
    (let* ((lane1
            (+
             (modulo
              (+
               (+ (* (@dup i) -1) (quotient (@dup i) -1)) ;; -2i
               (+ (* localId warpSize) (quotient localId warpSize)) ;; 0
               struct-size)
              warpSize)
             (modulo
              (+
               (+ (* (@dup i) warpSize) (quotient (@dup i) -1)) ;; -i
               (+ (* localId c) (quotient localId -1)) ;; 7j
               -1)
              warpSize))) ;; (7j - 3i - 1) % 32
           (x (shfl (get I-cached (@dup i)) lane1)))
      (set temp (@dup i) x)))
   (for
    ((i struct-size))
    (let* ((index
            (modulo
             (+
              (+ (* (@dup i) warpSize) (quotient (@dup i) 1)) ;; i
              (+ (* localId a) (quotient localId warpSize)) ;; j
              a)
             c)) ;; (i + j + 1) % 4
           (lane2
            (+
             (modulo
              (+
               (+ (* (@dup i) b) (quotient (@dup i) struct-size)) ;; 8i + i/4
               (+ (* localId b) (quotient localId warpSize)) ;; 8j
               warpSize)
              warpSize)
             (modulo
              (+
               (+ (* (@dup i) struct-size) (quotient (@dup i) -1)) ;; 3i
               (+ (* localId struct-size) (quotient localId b)) ;; 4j + j/8
               -1) ;; -1
              warpSize))) ;;  (11i + i/4 + 12j + j/8 - 1) % 32
           (x (shfl (get temp index) lane2)))
      (pretty-display `(lane ,(print-vec (modulo lane2 32))))
      (set O-cached (@dup i) x)))
   (local-to-global
    O-cached
    O
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f))

(define (AOS-loadsh-v2 threadId blockID blockDim I O a b c)
   (define I-cached (create-matrix-local (x-y-z struct-size)))
   (define temp (create-matrix-local (x-y-z struct-size)))
   (define O-cached (create-matrix-local (x-y-z struct-size)))
   (define warpID (get-warpId threadId))
   (define offset
     (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))
   (define gid (get-global-threadId threadId blockID))
   (global-to-local
    I
    I-cached
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f)
   (define indices (make-vector struct-size))
   (define indices-o (make-vector struct-size))
   (define localId (get-idInWarp threadId))
   (for
    ((i struct-size))
    (let* ((lane1
            (+
             (modulo
              (+
               (+ (* (@dup i) a) (* (- 0 a) (quotient (@dup i) a)))
               (+ (* localId 0) (* struct-size (quotient localId a)))
               0)
              warpSize)
             (modulo
              (+
               (+ (* (@dup i) a) (* 0 (quotient (@dup i) 1)))
               (+ (* localId 0) (* -1 (quotient localId b)))
               0)
              struct-size)))
           (x (shfl (get I-cached (@dup i)) lane1)))
      (set temp (@dup i) x)))
   (for
    ((i struct-size))
    (let* ((index
            (modulo
             (+
              (+ (* (@dup i) 1) (* 0 (quotient (@dup i) a)))
              (+ (* localId 0) (* a (quotient localId b)))
              0)
             c))
           (lane2
            (modulo
             (+
              (+ (* (@dup i) b) (* 0 (quotient (@dup i) 1)))
              (+ (* localId 0) (* a (quotient localId 1)))
              0)
             warpSize))
           (x (shfl-send (get temp index) lane2)))
      (set O-cached (@dup i) x)))
   (local-to-global
    O-cached
    O
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f))

(define (AOS-loadsh4-mp threadId blockID blockDim I O a b c)
   (define I-cached (create-matrix-local (x-y-z struct-size)))
   (define temp (create-matrix-local (x-y-z struct-size)))
   (define O-cached (create-matrix-local (x-y-z struct-size)))
   (define warpID (get-warpId threadId))
   (define offset
     (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))
   (define gid (get-global-threadId threadId blockID))
   (global-to-local
    I
    I-cached
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f)
   (define indices (make-vector struct-size))
   (define indices-o (make-vector struct-size))
   (define localId (get-idInWarp threadId))
   (for
    ((i struct-size))
    (let* ((lane1
            (+ (* (quotient localId 4) 4)
               (modulo (- localId i) 4)
             ))
           (x (shfl (get I-cached (@dup i)) lane1)))
      (set temp (@dup i) x)))
   (for
    ((i struct-size))
    (let* ((index
            (modulo (- localId i) 4)) 
           (lane2
            (modulo
             (+ (* 8 localId) (quotient localId 4) (* -8 i))
             32))
           (x (shfl-send (get temp index) lane2)))
      ;(pretty-display `(lane ,(print-vec (modulo lane2 32))))
      (set O-cached (@dup i) x)))
   (local-to-global
    O-cached
    O
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f))


(define (AOS-load5 threadId blockID blockDim I O a b c)
   (define I-cached (create-matrix-local (x-y-z struct-size)))
   (define O-cached (create-matrix-local (x-y-z struct-size)))
   (define warpID (get-warpId threadId))
   (define offset
     (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))
   (define gid (get-global-threadId threadId blockID))
   (global-to-local
    I
    I-cached
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f)
   (define indices (make-vector struct-size))
   (define indices-o (make-vector struct-size))
   (define localId (get-idInWarp threadId))
   (for
    ((i struct-size))
    (let* ((index (fan 5 5 3 2 0 i localId 5))
           #;(index
            (modulo
             (+
              (+ (* (@dup i) -1) (quotient (@dup i) -1))
              (+ (* localId c) (quotient localId 1))
              0)
             a)) ;; (2j - 2i) % 5
           (lane (fan 32 32 5 1 0 localId i 32))
           #;(lane
            (+
             (modulo
              (+
               (+ (* (@dup i) 0) (quotient (@dup i) 1))
               (+ (* localId a) (quotient localId warpSize))
               0)
              b) 
             (modulo
              (+
               (+ (* (@dup i) -1) (quotient (@dup i) 1))
               (+ (* localId -1) (quotient localId 1))
               0)
              c))) ;; (i + 5j) % 32
           (x (shfl (get I-cached index) lane))
           (index-o (fan 5 5 1 0 0 i localId 5))
           ;(index-o (fan 5 5 0 0 0 i localId 1))
           #;(index-o
            (modulo
             (+
              (+ (* (@dup i) 0) (quotient (@dup i) c))
              (+ (* localId -1) (quotient localId c))
              warpSize) ;; i % 5
             warpSize)))
      (unique-warp (modulo lane warpSize))
      (pretty-display `(i ,i))
      (vector-set! indices i index)
      (vector-set! indices-o i index-o)
      (set O-cached index-o x)))
   (for
    ((t blockSize))
    (let ((l
           (for/list ((i struct-size)) (vector-ref (vector-ref indices i) t)))
          (lo
           (for/list
            ((i struct-size))
            (vector-ref (vector-ref indices-o i) t))))
      (unique-list l)
      (unique-list lo)))
   (local-to-global
    O-cached
    O
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f))

(define (AOS-loadsh5 threadId blockID blockDim I O a b c)
   (define I-cached (create-matrix-local (x-y-z struct-size)))
   (define temp (create-matrix-local (x-y-z struct-size)))
   (define O-cached (create-matrix-local (x-y-z struct-size)))
   (define warpID (get-warpId threadId))
   (define offset
     (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))
   (define gid (get-global-threadId threadId blockID))
   (global-to-local
    I
    I-cached
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f)
   (define indices (make-vector struct-size))
   (define indices-o (make-vector struct-size))
   (define localId (get-idInWarp threadId))
   (for
    ((i struct-size))
    (let* ((x (get I-cached (@dup i))))
      (set temp (@dup i) x)))
   (for
    ((i struct-size))
    (let* ((index (fan i 5 3 5 5 1
                       localId warpSize 2 warpSize))
           #;(index
            (modulo (+ (* 3 i) (* localId 2)) 5))
           (lane2 (fan localId 32 13 32 32 1
                       i 5 19 5))
           #;(lane2
            (modulo
             (- (* 13 localId) (* 13 i))
             32))
           (x (shfl-send (get temp index) lane2)))
      ;(pretty-display `(lane ,(print-vec (modulo lane2 32))))
      (set O-cached (@dup i) x)))
   (local-to-global
    O-cached
    O
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f))


(define (AOS-load7 threadId blockID blockDim I O a b c)
   (define I-cached (create-matrix-local (x-y-z struct-size)))
   (define O-cached (create-matrix-local (x-y-z struct-size)))
   (define warpID (get-warpId threadId))
   (define offset
     (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))
   (define gid (get-global-threadId threadId blockID))
   (global-to-local
    I
    I-cached
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f)
   (define indices (make-vector struct-size))
   (define indices-o (make-vector struct-size))
   (define localId (get-idInWarp threadId))
   (for
    ((i struct-size))
    (let* ((index (fan struct-size struct-size 2 5 0 i localId struct-size))
           #;(index
            (modulo
             (+
              (* i 2)
              (* localId -2)
              0)
             struct-size)) ;; (2*i - 2*j) % M;
           (lane (fan 32 32 struct-size 1 0 localId i 32))
           #;(lane
             (modulo
              (+
               (* struct-size localId)
               i
               )
              warpSize)) ;; (i + M*j) % WARP_SIZE;
           (x (shfl (get I-cached index) lane))
           (index-o (fan struct-size struct-size 1 0 0 i localId struct-size))
           ;(index-o (@dup i))
           )
      (unique-warp (modulo lane warpSize))
      (pretty-display `(i ,i))
      (pretty-display `(index ,(print-vec index)))
      (vector-set! indices i index)
      (vector-set! indices-o i index-o)
      (set O-cached index-o x)))
   (for
    ((t blockSize))
    (let ((l
           (for/list ((i struct-size)) (vector-ref (vector-ref indices i) t)))
          (lo
           (for/list
            ((i struct-size))
            (vector-ref (vector-ref indices-o i) t))))
      (unique-list l)
      (unique-list lo)))
   (local-to-global
    O-cached
    O
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f))

(define (AOS-loadsh7 threadId blockID blockDim I O a b c)
   (define I-cached (create-matrix-local (x-y-z struct-size)))
   (define temp (create-matrix-local (x-y-z struct-size)))
   (define O-cached (create-matrix-local (x-y-z struct-size)))
   (define warpID (get-warpId threadId))
   (define offset
     (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))
   (define gid (get-global-threadId threadId blockID))
   (global-to-local
    I
    I-cached
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f)
   (define indices (make-vector struct-size))
   (define indices-o (make-vector struct-size))
   (define localId (get-idInWarp threadId))
   (for
    ((i struct-size))
    (let* ((lane1 (fan warpSize 32 0 0 0 localId i 1))
           (x (shfl (get I-cached (@dup i)) lane1)))
      (set temp (@dup i) x)))
   (for
    ((i struct-size))
    (let* ((index (fan struct-size struct-size 2 5 0 i localId 7))
           (lane2 (fan warpSize 32 23 9 0 localId i 32))
           (x (shfl-send (get temp index) lane2)))
      (set O-cached (@dup i) x)))
   (local-to-global
    O-cached
    O
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f))

(define (AOS-load3 threadId blockID blockDim I O a b c)
   (define I-cached (create-matrix-local (x-y-z struct-size)))
   (define O-cached (create-matrix-local (x-y-z struct-size)))
   (define warpID (get-warpId threadId))
   (define offset
     (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))
   (define gid (get-global-threadId threadId blockID))
   (global-to-local
    I
    I-cached
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f)
   (define indices (make-vector struct-size))
   (define indices-o (make-vector struct-size))
   (define localId (get-idInWarp threadId))
   (for
    ((i struct-size))
    (let* ((index (fan struct-size 3 2 1 0 i localId 3))
           (lane (fan warpSize 32 3 1 0 localId i 32))
           (x (shfl (get I-cached index) lane))
           (index-o (fan struct-size 3 0 0 0 i localId 1)))
      (unique-warp (modulo lane warpSize))
      (vector-set! indices i index)
      (vector-set! indices-o i index-o)
      (set O-cached index-o x)))
   (for
    ((t blockSize))
    (let ((l
           (for/list ((i struct-size)) (vector-ref (vector-ref indices i) t)))
          (lo
           (for/list
            ((i struct-size))
            (vector-ref (vector-ref indices-o i) t))))
      (unique-list l)
      (unique-list lo)))
   (local-to-global
    O-cached
    O
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f))

(define (AOS-loadsh3 threadId blockID blockDim I O a b c)
   (define I-cached (create-matrix-local (x-y-z struct-size)))
   (define temp (create-matrix-local (x-y-z struct-size)))
   (define O-cached (create-matrix-local (x-y-z struct-size)))
   (define warpID (get-warpId threadId))
   (define offset
     (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))
   (define gid (get-global-threadId threadId blockID))
   (global-to-local
    I
    I-cached
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f)
   (define indices (make-vector struct-size))
   (define indices-o (make-vector struct-size))
   (define localId (get-idInWarp threadId))
   (for
    ((i struct-size))
    (let* (;(lane1 (fan warpSize 32 0 0 0 localId i 1))
           (lane1 (fan localId warpSize 1 warpSize warpSize 1
                       i struct-size 0 struct-size))
           (x (shfl (get I-cached (@dup i)) lane1)))
      (set temp (@dup i) x)))
   (for
    ((i struct-size))
    (let* (;(index (fan struct-size struct-size 2 1 0 i localId 3))
           (index (fan i struct-size 2 struct-size struct-size 1
                       localId warpSize 1 warpSize))
           ;(lane2 (fan warpSize 32 11 21 0 localId i 32))
           (lane2 (fan localId warpSize 11 warpSize warpSize 1
                       i struct-size 21 struct-size))
           (x (shfl-send (get temp index) lane2)))
      (set O-cached (@dup i) x)))
   (local-to-global
    O-cached
    O
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f))

(define (AOS-load2 threadId blockID blockDim I O a b c)
  (define I-cached (create-matrix-local (x-y-z struct-size)))
  (define O-cached (create-matrix-local (x-y-z struct-size)))
  (define warpID (get-warpId threadId))
  (define offset (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))  ;; warpID = (threadIdy * blockDimx + threadIdx)/warpSize
  (define gid (get-global-threadId threadId blockID))
  (global-to-local I I-cached
                 (x-y-z 1)
                 offset
                 (x-y-z (* warpSize struct-size)) #f)

  (define indices (make-vector struct-size))
  (define indices-o (make-vector struct-size))
  (define localId (get-idInWarp threadId))
  (for ([i struct-size])
    (let* ([index (fan-prime i struct-size 1
                             localId warpSize 1 warpSize)]
           [lane (fan localId warpSize 2 16 warpSize -1
                      i struct-size 1 struct-size)]
           [x (shfl (get I-cached index) lane)]
           [index-o (fan-prime i struct-size 1
                               localId warpSize 0 16)]
           )
      (pretty-display `(lane ,(print-vec lane)))
      (unique-warp (modulo lane warpSize))
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
                      offset
                      (x-y-z (* warpSize struct-size)) #f)
  )

(define (AOS-loadsh2 threadId blockID blockDim I O a b c)
   (define I-cached (create-matrix-local (x-y-z struct-size)))
   (define temp (create-matrix-local (x-y-z struct-size)))
   (define O-cached (create-matrix-local (x-y-z struct-size)))
   (define warpID (get-warpId threadId))
   (define offset
     (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))
   (define gid (get-global-threadId threadId blockID))
   (global-to-local
    I
    I-cached
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f)
   (define indices (make-vector struct-size))
   (define indices-o (make-vector struct-size))
   (define localId (get-idInWarp threadId))
   (for
    ((i struct-size))
    (let* ((lane1 (fan localId warpSize 0 1 2 1
                       i struct-size 1 struct-size))
           [x (shfl (get I-cached (@dup i)) lane1)])
      (set temp (@dup i) x)))
   (for
    ((i struct-size))
    (let* ((index (fan-prime i struct-size 1
                             localId warpSize 1 warpSize))
           (lane2 (fan localId warpSize 16 2 32 1
                       i struct-size 16 struct-size))
           (x (shfl-send (get temp index) lane2)))
      (set O-cached (@dup i) x)))
   (local-to-global
    O-cached
    O
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f))

(define (AOS-load3-fan threadId blockID blockDim I O a b c)
     (define I-cached (create-matrix-local (x-y-z struct-size)))
     (define O-cached (create-matrix-local (x-y-z struct-size)))
     (define warpID (get-warpId threadId))
     (define offset
       (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))
     (global-to-local
      I
      I-cached
      (x-y-z 1)
      offset
      (x-y-z (* warpSize struct-size))
      #f #:round struct-size)
     (define localId (get-idInWarp threadId))
     (define I-cached2 (permute-vector I-cached struct-size
                                       (lambda (i)
                                         (fan i struct-size 2 3 3 1 localId warpSize 0 1))))
     
     (for
         ((i struct-size))
       (let* ((lane (fan localId warpSize 3 32 32 1 i struct-size 0 1))
              (x (shfl (get I-cached2 (@dup i)) lane))
              ;(index-o (fan i struct-size 1 3 3 1 localId warpSize 0 warpSize))
              )
         (set O-cached (@dup i) x)))
     (define O-cached2 (permute-vector O-cached struct-size
                                       (lambda (i)
                                         (fan i struct-size 1 3 3 1 localId warpSize 0 warpSize))))
     (local-to-global
      O-cached2
      O
      (x-y-z 1)
      offset
      (x-y-z (* warpSize struct-size))
      #f #:round struct-size))

(define (test)
  (for ([w (list 32)])
    (let ([ret (run-with-warp-size AOS-load-spec AOS-load3-fan w)])
      (pretty-display `(test ,w ,ret))))
  )
(test)

