#lang rosette

(require "util.rkt" "cuda.rkt" "cuda-synth.rkt")

(define struct-size 5)
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
  (pretty-display `(O* ,O*))
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
                      (x-y-z 1) offset (x-y-z (* warpSize struct-size)) #f)
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

(define (AOS-loadsh5-mp threadId blockID blockDim I O a b c)
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
    (let* ((index
            (modulo (+ (* 3 i) (* localId 2)) 5)) 
           (lane2
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

(define (test)
  (for ([w (list 32)])
    (let ([ret (run-with-warp-size AOS-load-spec AOS-loadsh5-mp w)])
      (pretty-display `(test ,w ,ret))))
  )
(test)

