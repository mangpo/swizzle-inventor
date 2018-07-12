#lang racket

(define data-type "int")
(define index-type "int")

(define dims 1)
(define thread-id #f)
(define block-id #f)
(define block-size #f)

(define env (make-hash))
(define matrix-size (make-hash))
(define cuda-vars (make-hash))
(define env-consts (hash 'struct-size 2 'warpSize 32))

(define (cuda-var? v)
  (hash-has-key? cuda-vars v))
(define (load-store? f)
  (member f '(global-to-local local-to-global global-to-shared shared-to-global)))
(define (load? f)
  (member f '(global-to-local global-to-shared)))
(define (store? f)
  (member f '(local-to-global shared-to-global)))
(define (let? x)
  (member x '(let let*)))

(define (eval-const v)
  (if (hash-has-key? env-consts v)
      (hash-ref env-consts v)
      v))

(define (print-cuda l)
  (for ([s (flatten l)])
    (pretty-display s)))

(define (racket2cuda expr d)
  (set! dims d)
  (match expr
    [(list 'define (list func-name tid bid bsize args ...) body ...)
     (set! thread-id tid) 
     (set! block-id bid) 
     (set! block-size bsize)
     (hash-set! env tid d)
     (hash-set! env block-id d)
     (hash-set! env block-size d)
     (hash-set! cuda-vars tid "threadIdx")
     (hash-set! cuda-vars block-id "blockIdx")
     (hash-set! cuda-vars block-size "blockDim")
     
     (for/list ([st body]) (convert-statement st))
     ]))

(define (dims-str sizes)
  (define dims (map (lambda (s) (format "[~a]" s)) (reverse sizes)))
  (string-join dims ""))

(define indent-str "")
(define (inc-indent) (set! indent-str (string-append indent-str "  ")))
(define (dec-indent) (set! indent-str (substring indent-str 2)))
(define-syntax-rule (format-indent s args ...)
  (string-append indent-str (format s args ...)))

(define (convert-statement st)
  (match st
    [(list 'define matrix (list 'create-matrix-local (list 'x-y-z sizes ...) '#:type type))
     ;(hash-set! env matrix (length sizes))
     (hash-set! matrix-size matrix sizes)
     (format-indent "~a ~a~a;" type matrix (dims-str sizes))]

    [(list 'define matrix (list 'create-matrix-local (list 'x-y-z sizes ...)))
     ;(hash-set! env matrix (length sizes))
     (hash-set! matrix-size matrix sizes)
     (format-indent "~a ~a~a;" data-type matrix (dims-str sizes))]

    [(list 'define-shared matrix (list 'create-matrix (list 'x-y-z sizes ...) '#:type type))
     ;(hash-set! env matrix (length sizes))
     (hash-set! matrix-size matrix sizes)
     (format-indent "__shared__ ~a ~a~a;" type matrix (dims-str sizes))]

    [(list 'define-shared matrix (list 'create-matrix (list 'x-y-z sizes ...)))
     ;(hash-set! env matrix (length sizes))
     (hash-set! matrix-size matrix sizes)
     (format-indent "__shared__ ~a ~a~a;" data-type matrix (dims-str sizes))]

    [(list 'define matrix (list 'make-vector size))
     ;(hash-set! env matrix 1)
     (hash-set! matrix-size matrix (list size))
     (format-indent "~a ~a~a;" data-type matrix (dims-str (list size)))]

    [(list 'define var e)
     (define-values (n f) (convert-expr e))
     (hash-set! env var n)
     (if (= n 1)
         (format-indent "~a ~a = ~a;" index-type var (f 0))
         (for/list ([i n])
           (format-indent "~a ~a~a = ~a;" index-type var i (f i))))]

    [(list (? load-store? f) A B stride offset size transpose)
     (define warp-shape
       (cond [(= dims 1) 32]
             [(= dims 2) '(x-y-z 32 1)]
             [(= dims 3) '(x-y-z 32 1 1)]))
     (define cuda-f (string-replace (symbol->string f) "-" "_"))
     (define global (if (load? f) A B))
     (define local (if (load? f) B A))
     (convert-global-to-local cuda-f
                              global local stride offset size transpose warp-shape
                              (hash-ref matrix-size local))
     ]
    

    [(list (? load-store? f) A B stride offset size transpose '#:warp-shape warp-shape)
     (define cuda-f (string-replace (symbol->string f) "-" "_"))
     (define global (if (load? f) A B))
     (define local (if (load? f) B A))
     (convert-global-to-local cuda-f
                              global local stride offset size transpose warp-shape
                              (hash-ref matrix-size local))
     ]

    [(list 'for (list (list vs ls) ...) body)
     (for ([v vs])
       (hash-set! env v 1))
     (append
      (for/list ([v vs] [l vs])
        (let ([temp (format-indent "for(int ~a = 0; ~a < ~a; ~a++) {" v v l v)])
          (inc-indent)
          temp))
      (list (convert-statement body))
      (for/list ([v vs])
        (dec-indent)
        (format-indent "}")))
     ]

    [(list let? (list (list vs es) ...) body)
     (append
      (for/list ([e es] [v vs])
        (let-values ([(n f) (convert-expr e)])
          (hash-set! env v n)
          (cond
            [(= n 1)
             (format-indent "int ~a = ~a;" v (f 0))]
           [else
            (for/list ([i n]) (format-indent "int ~a~a = ~a;" v i (f i)))])
          ))
      (list (convert-statement body)))
     ]

    [(list 'set matrix idxs ... v)
     (define str-list
       (for/list ([idx idxs])
         (let-values ([(n idx-f) (convert-expr idx)])
           (idx-f 0))))

     (define v-str
       (let-values ([(n v-f) (convert-expr v)])
         (v-f 0)))

     (format-indent "~a~a = ~a;" matrix (dims-str str-list) v-str)
     ]
    
    ))

(define (convert-global-to-local name A B stride offset size transpose warp-shape local-size)
  (define-values (stride-n stride-f) (convert-expr stride))
  (define-values (offset-n offset-f) (convert-expr offset))
  (define-values (size-n size-f) (convert-expr size))
  (define-values (shape-n shape-f) (convert-expr warp-shape))
  (define d (max stride-n offset-n size-n))
  (define str-list
    (list (format "~a~a(~a, ~a, " name d A B)
          "," (string-join (for/list ([i d]) (stride-f i)) ", ")
          "," (string-join (for/list ([i d]) (offset-f i)) ", ")
          "," (string-join (for/list ([i d]) (size-f i)) ", ")
          "," (string-join (for/list ([i d]) (shape-f i)) ", ")
          "," (string-join (for/list ([i d]) (format "~a" (list-ref local-size i))) ", ")
          "," (if transpose "1" "0") ");"))
  (format-indent "~a" (string-join (flatten str-list) ""))
  )

(define (convert-expr expr)
  (match expr
    [(list 'get-warpId tid)
     (values
      1
      (lambda (i)
        (cond
          [(= dims 1) "(threadIdx.x/32)"]
          [(= dims 2) "(threadIdx.y*blockDim.x + threadIdx.x)/32)"]
        [(= dims 3) "(threadIdx.z*blockDim.y*blockDim.x + threadIdx.y*blockDim.x + threadIdx.x)/32)"]
        )))]

    [(list 'get-idInWarp tid)
     (values 1 (lambda (i) "(threadIdx.x&31)"))]

    [(list 'get-global-threadId tid bid)
     (values
      dims
      (lambda (i)
        (cond
          [(= i 0) "(blockIdx.x * blockDim.x + threadIdx.x)"]
          [(= i 1) "(blockIdx.y * blockDim.y + threadIdx.y)"]
          [(= i 2) "(blockIdx.z * blockDim.z + threadIdx.z)"])))]

    [(list 'shfl e lane)
     (define-values (e-n e-f) (convert-expr e))
     (define-values (lane-n lane-f) (convert-expr lane))
     (values 1 (lambda (i) (format "__shfl_sync(FULL_MASK, ~a, ~a)" (e-f 0) (lane-f 0))))
     ]

    [(list 'get matrix idxs ...)
     (define str-list
       (for/list ([idx idxs])
         (let-values ([(n idx-f) (convert-expr idx)])
           (idx-f 0))))

     (values 1 (lambda (i) (format "~a~a" matrix (dims-str str-list))))
     ]

    [(list 'fan j n* cj* dj* group* conf-fw
           k m* ck* dk* #:offset offset)

     (define n (eval-const n*))
     (define cj (eval-const cj*))
     (define dj (eval-const dj*))
     (define group (eval-const group*))
     (define m (eval-const m*))
     (define ck (eval-const ck*))
     (define dk (eval-const dk*))
     
     (define offset1-a
       (cond
         [(equal? dj group) 0]
         [(equal? group n) (@quotient j dj)]
         [else (@quotient (@modulo j group) dj)]))

     (define offset1-b (@* k ck))

     (define offset1-c
       (cond
         [(equal? dk group) 0] [else (@quotient k dk)]))

     (define offset1 (@+ offset1-a offset1-b offset1-c offset))

     (define common (quotient group dj))
     (define offset2
       (cond
         [(or (= conf-fw 1) (equal? common group)) offset1]
         [else (@modulo offset1 common)]))

     (define group-offset
       (cond
         [(equal? group n) 0] [else (@* (@quotient j group) group)]))

     (pretty-display `(offset2 ,offset2))
     (define all (@+ group-offset
                     (@modulo (@+ (@* j cj) offset2) group)))

     (convert-expr all)
     ]

    ;(fan i struct-size 0 1 2 1 localId warpSize 0 1)

    [(list '@dup x)
     (convert-expr x)
     ]

    [(list op args ...)
     (define max-d 1)
     (define rets 
       (for/list ([arg args])
         (let-values ([(n f) (convert-expr arg)])
           (cons n f))))

     (values max-d
             (lambda (i)
               (format "(~a)"
                       (string-join
                        (for/list ([ret rets]) ((cdr ret) i))
                        (convert-op op)))))
     ]

    [(list 'x-y-z xs ...)
     (values (length xs) (lambda (i) (list-ref xs i)))
     ]

    [(? cuda-var? v)
     (define name (hash-ref cuda-vars v))
     (values dims
             (lambda (i)
               (cond [(= i 0) (format "~a.x" name)]
                     [(= i 1) (format "~a.y" name)]
                     [(= i 2) (format "~a.z" name)])))
     ]

    [v
     (define d (if (hash-has-key? env v) (hash-ref env v) 1))
     (cond
       [(= d 1) (values d (lambda (i) (format "~a" v)))]
       [else (values d (lambda (i) (format "~a~a" v i)))])
     ]
    ))

(define (convert-op op)
  (match op
    ['quotient "/"]
    ['modulo "%"]
    [x (symbol->string x)]))

(define (@++ xs)
  (cond
    [(= (length xs) 1) (car xs)]
    [else
     (define y (@++ (cdr xs)))
     (define x (car xs))
     (cond
       [(equal? x 0) y]
       [(equal? y 0) x]
       [else `(+ ,x ,y)])
     ]))

(define (@** xs)
  (define ret
  (cond
    [(= (length xs) 1) (car xs)]
    [else
     (define y (@++ (cdr xs)))
     (define x (car xs))
     (cond
       [(equal? x 0) 0]
       [(equal? y 0) 0]
       [(equal? x 1) y]
       [(equal? y 1) x]
       [else `(* ,x ,y)])
     ]))
  (pretty-display `(@** ,xs ,ret))
  ret
  )

(define-syntax-rule (@+ x ...) (@++ (list x ...)))
(define-syntax-rule (@* x ...) (@** (list x ...)))

(define (@quotient x y)
  (cond
    [(equal? y 1) x]
    [(equal? x 0) 0]
    [else `(quotient ,x ,y)]))

(define (@modulo x y)
  (cond
    [(equal? y 1) 0]
    [else `(modulo ,x ,y)]))

(define func
  '(define (AOS-load-spec threadId blockID blockDim I O a b c)
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
  )

(define loop
  '(for
    ((i struct-size))
    (let* ((lane1
            (+ (* (quotient localId 4) 4)
               (modulo (- localId i) 4)
             ))
           (x (shfl (get I-cached (@dup i)) lane1)))
      (set temp (@dup i) x))))

(define fan
  '(define lane (fan i struct-size 0 1 2 1 localId warpSize 0 1 #:offset 0)))

;(racket2cuda func 1)
;(print-cuda (convert-statement loop))
(print-cuda (convert-statement fan))