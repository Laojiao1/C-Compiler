; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"

@global_smem = external addrspace(3) global [0 x i8], align 16

; Function Attrs: nounwind
define ptx_kernel void @softmax_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) readnone captures(none) %2, ptr addrspace(1) readnone captures(none) %3) local_unnamed_addr #0 !dbg !4 {
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !dbg !7
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !dbg !8
  %7 = and i32 %6, 255, !dbg !8
  %8 = and i32 %6, 31, !dbg !8
  %9 = lshr i32 %7, 5, !dbg !8
  %10 = shl nuw nsw i32 %6, 2, !dbg !8
  %11 = and i32 %10, 1020, !dbg !8
  %12 = mul i32 %5, 1000, !dbg !9
  %13 = sext i32 %12 to i64, !dbg !10
  %14 = getelementptr float, ptr addrspace(1) %0, i64 %13, !dbg !10
  %15 = zext nneg i32 %11 to i64, !dbg !11
  %16 = getelementptr float, ptr addrspace(1) %14, i64 %15, !dbg !11
  %17 = icmp samesign ult i32 %11, 1000, !dbg !12
  %18 = tail call { i32, i32, i32, i32 } asm sideeffect "mov.u32 $0, $4;\0A\09mov.u32 $1, $5;\0A\09mov.u32 $2, $6;\0A\09mov.u32 $3, $7;\0A\09@$9 ld.global.v4.b32 { $0, $1, $2, $3 }, [ $8 + 0 ];", "=r,=r,=r,=r,r,r,r,r,l,b"(i32 -8388608, i32 -8388608, i32 -8388608, i32 -8388608, ptr addrspace(1) %16, i1 %17) #5, !dbg !13
  %19 = extractvalue { i32, i32, i32, i32 } %18, 0, !dbg !13
  %20 = extractvalue { i32, i32, i32, i32 } %18, 1, !dbg !13
  %21 = extractvalue { i32, i32, i32, i32 } %18, 2, !dbg !13
  %22 = extractvalue { i32, i32, i32, i32 } %18, 3, !dbg !13
  %23 = bitcast i32 %19 to float, !dbg !13
  %24 = bitcast i32 %20 to float, !dbg !13
  %25 = bitcast i32 %21 to float, !dbg !13
  %26 = bitcast i32 %22 to float, !dbg !13
  %27 = tail call float @llvm.maxnum.f32(float %23, float %24), !dbg !14
  %28 = tail call float @llvm.maxnum.f32(float %27, float %25), !dbg !14
  %29 = tail call float @llvm.maxnum.f32(float %28, float %26), !dbg !14
  %30 = bitcast float %29 to i32, !dbg !17
  %31 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %30, i32 16, i32 31), !dbg !17
  %32 = bitcast i32 %31 to float, !dbg !17
  %33 = tail call float @llvm.maxnum.f32(float %29, float %32), !dbg !14
  %34 = bitcast float %33 to i32, !dbg !17
  %35 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %34, i32 8, i32 31), !dbg !17
  %36 = bitcast i32 %35 to float, !dbg !17
  %37 = tail call float @llvm.maxnum.f32(float %33, float %36), !dbg !14
  %38 = bitcast float %37 to i32, !dbg !17
  %39 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %38, i32 4, i32 31), !dbg !17
  %40 = bitcast i32 %39 to float, !dbg !17
  %41 = tail call float @llvm.maxnum.f32(float %37, float %40), !dbg !14
  %42 = bitcast float %41 to i32, !dbg !17
  %43 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %42, i32 2, i32 31), !dbg !17
  %44 = bitcast i32 %43 to float, !dbg !17
  %45 = tail call float @llvm.maxnum.f32(float %41, float %44), !dbg !14
  %46 = bitcast float %45 to i32, !dbg !17
  %47 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %46, i32 1, i32 31), !dbg !17
  %48 = bitcast i32 %47 to float, !dbg !17
  %49 = tail call float @llvm.maxnum.f32(float %45, float %48), !dbg !14
  %50 = icmp eq i32 %8, 0, !dbg !17
  %51 = getelementptr float, ptr addrspace(3) @global_smem, i32 %9, !dbg !17
  %52 = bitcast float %49 to <1 x i32>, !dbg !17
  tail call void asm sideeffect "@$2 st.shared.b32 [ $0 + 0 ], $1;", "r,r,b"(ptr addrspace(3) %51, <1 x i32> %52, i1 %50) #5, !dbg !17
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0), !dbg !17
  %53 = icmp samesign ult i32 %7, 8, !dbg !17
  %54 = getelementptr float, ptr addrspace(3) @global_smem, i32 %7, !dbg !17
  %55 = tail call i32 asm sideeffect "@$2 ld.shared.b32 $0, [ $1 + 0 ];", "=r,r,b"(ptr addrspace(3) %54, i1 %53) #5, !dbg !17
  %56 = bitcast i32 %55 to float, !dbg !17
  %57 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %55, i32 4, i32 31), !dbg !17
  %58 = bitcast i32 %57 to float, !dbg !17
  %59 = tail call float @llvm.maxnum.f32(float %56, float %58), !dbg !14
  %60 = bitcast float %59 to i32, !dbg !17
  %61 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %60, i32 2, i32 31), !dbg !17
  %62 = bitcast i32 %61 to float, !dbg !17
  %63 = tail call float @llvm.maxnum.f32(float %59, float %62), !dbg !14
  %64 = bitcast float %63 to i32, !dbg !17
  %65 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %64, i32 1, i32 31), !dbg !17
  %66 = bitcast i32 %65 to float, !dbg !17
  %67 = tail call float @llvm.maxnum.f32(float %63, float %66), !dbg !14
  %68 = and i32 %6, 7, !dbg !17
  %69 = icmp eq i32 %68, 0, !dbg !17
  %70 = and i1 %53, %69, !dbg !17
  %71 = bitcast float %67 to <1 x i32>, !dbg !17
  tail call void asm sideeffect "@$2 st.shared.b32 [ $0 + 0 ], $1;", "r,r,b"(ptr addrspace(3) %54, <1 x i32> %71, i1 %70) #5, !dbg !17
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0), !dbg !17
  %72 = load float, ptr addrspace(3) @global_smem, align 16, !dbg !17
  %73 = fsub float %23, %72, !dbg !20
  %74 = fsub float %24, %72, !dbg !20
  %75 = fsub float %25, %72, !dbg !20
  %76 = fsub float %26, %72, !dbg !20
  %77 = fmul float %73, 0x3FF7154760000000, !dbg !21
  %78 = tail call float @llvm.nvvm.ex2.approx.f(float %77), !dbg !21
  %79 = fmul float %74, 0x3FF7154760000000, !dbg !21
  %80 = tail call float @llvm.nvvm.ex2.approx.f(float %79), !dbg !21
  %81 = fmul float %75, 0x3FF7154760000000, !dbg !21
  %82 = tail call float @llvm.nvvm.ex2.approx.f(float %81), !dbg !21
  %83 = fmul float %76, 0x3FF7154760000000, !dbg !21
  %84 = tail call float @llvm.nvvm.ex2.approx.f(float %83), !dbg !21
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0), !dbg !22
  %85 = fadd float %78, %80, !dbg !24
  %86 = fadd float %85, %82, !dbg !24
  %87 = fadd float %86, %84, !dbg !24
  %88 = bitcast float %87 to i32, !dbg !22
  %89 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %88, i32 16, i32 31), !dbg !22
  %90 = bitcast i32 %89 to float, !dbg !22
  %91 = fadd float %87, %90, !dbg !24
  %92 = bitcast float %91 to i32, !dbg !22
  %93 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %92, i32 8, i32 31), !dbg !22
  %94 = bitcast i32 %93 to float, !dbg !22
  %95 = fadd float %91, %94, !dbg !24
  %96 = bitcast float %95 to i32, !dbg !22
  %97 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %96, i32 4, i32 31), !dbg !22
  %98 = bitcast i32 %97 to float, !dbg !22
  %99 = fadd float %95, %98, !dbg !24
  %100 = bitcast float %99 to i32, !dbg !22
  %101 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %100, i32 2, i32 31), !dbg !22
  %102 = bitcast i32 %101 to float, !dbg !22
  %103 = fadd float %99, %102, !dbg !24
  %104 = bitcast float %103 to i32, !dbg !22
  %105 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %104, i32 1, i32 31), !dbg !22
  %106 = bitcast i32 %105 to float, !dbg !22
  %107 = fadd float %103, %106, !dbg !24
  %108 = bitcast float %107 to <1 x i32>, !dbg !22
  tail call void asm sideeffect "@$2 st.shared.b32 [ $0 + 0 ], $1;", "r,r,b"(ptr addrspace(3) %51, <1 x i32> %108, i1 %50) #5, !dbg !22
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0), !dbg !22
  %109 = tail call i32 asm sideeffect "@$2 ld.shared.b32 $0, [ $1 + 0 ];", "=r,r,b"(ptr addrspace(3) %54, i1 %53) #5, !dbg !22
  %110 = bitcast i32 %109 to float, !dbg !22
  %111 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %109, i32 4, i32 31), !dbg !22
  %112 = bitcast i32 %111 to float, !dbg !22
  %113 = fadd float %110, %112, !dbg !24
  %114 = bitcast float %113 to i32, !dbg !22
  %115 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %114, i32 2, i32 31), !dbg !22
  %116 = bitcast i32 %115 to float, !dbg !22
  %117 = fadd float %113, %116, !dbg !24
  %118 = bitcast float %117 to i32, !dbg !22
  %119 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %118, i32 1, i32 31), !dbg !22
  %120 = bitcast i32 %119 to float, !dbg !22
  %121 = fadd float %117, %120, !dbg !24
  %122 = bitcast float %121 to <1 x i32>, !dbg !22
  tail call void asm sideeffect "@$2 st.shared.b32 [ $0 + 0 ], $1;", "r,r,b"(ptr addrspace(3) %54, <1 x i32> %122, i1 %70) #5, !dbg !22
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0), !dbg !22
  %123 = load float, ptr addrspace(3) @global_smem, align 16, !dbg !22
  %124 = tail call float @llvm.nvvm.div.full(float %78, float %123), !dbg !25
  %125 = tail call float @llvm.nvvm.div.full(float %80, float %123), !dbg !25
  %126 = tail call float @llvm.nvvm.div.full(float %82, float %123), !dbg !25
  %127 = tail call float @llvm.nvvm.div.full(float %84, float %123), !dbg !25
  %128 = getelementptr float, ptr addrspace(1) %1, i64 %13, !dbg !26
  %129 = getelementptr float, ptr addrspace(1) %128, i64 %15, !dbg !27
  %130 = bitcast float %124 to i32, !dbg !28
  %131 = bitcast float %125 to i32, !dbg !28
  %132 = bitcast float %126 to i32, !dbg !28
  %133 = bitcast float %127 to i32, !dbg !28
  tail call void asm sideeffect "@$5 st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l,b"(i32 %130, i32 %131, i32 %132, i32 %133, ptr addrspace(1) %129, i1 %17) #5, !dbg !28
  ret void, !dbg !29
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #1

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.bfly.i32(i32, i32, i32, i32) #2

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.ex2.approx.f(float) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.div.full(float, float) #4

attributes #0 = { nounwind "nvvm.reqntid"="256" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #3 = { convergent nocallback nounwind }
attributes #4 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #5 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "triton", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "dump_softmax.py", directory: "/mnt/e/Code/C++Compiler/4_GPU_DeepDive/01_triton_lowering/softmax_ir")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 4, !"nvvm-reflect-ftz", i32 1}
!4 = distinct !DISubprogram(name: "softmax_kernel", linkageName: "softmax_kernel", scope: !1, file: !1, line: 11, type: !5, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DISubroutineType(cc: DW_CC_normal, types: !6)
!6 = !{}
!7 = !DILocation(line: 18, column: 28, scope: !4)
!8 = !DILocation(line: 21, column: 31, scope: !4)
!9 = !DILocation(line: 24, column: 26, scope: !4)
!10 = !DILocation(line: 27, column: 29, scope: !4)
!11 = !DILocation(line: 27, column: 41, scope: !4)
!12 = !DILocation(line: 30, column: 25, scope: !4)
!13 = !DILocation(line: 34, column: 8, scope: !4)
!14 = !DILocation(line: 170, column: 27, scope: !15, inlinedAt: !17)
!15 = distinct !DILexicalBlockFile(scope: !4, file: !16, discriminator: 0)
!16 = !DIFile(filename: "standard.py", directory: "/home/laojiao/miniconda3/envs/triton/lib/python3.10/site-packages/triton/language")
!17 = !DILocation(line: 191, column: 40, scope: !15, inlinedAt: !18)
!18 = !DILocation(line: 40, column: 21, scope: !19)
!19 = distinct !DILexicalBlockFile(scope: !4, file: !1, discriminator: 0)
!20 = !DILocation(line: 43, column: 29, scope: !4)
!21 = !DILocation(line: 43, column: 23, scope: !4)
!22 = !DILocation(line: 293, column: 36, scope: !15, inlinedAt: !23)
!23 = !DILocation(line: 46, column: 25, scope: !19)
!24 = !DILocation(line: 263, column: 15, scope: !15, inlinedAt: !22)
!25 = !DILocation(line: 49, column: 33, scope: !4)
!26 = !DILocation(line: 52, column: 31, scope: !4)
!27 = !DILocation(line: 52, column: 43, scope: !4)
!28 = !DILocation(line: 55, column: 8, scope: !4)
!29 = !DILocation(line: 53, column: 4, scope: !4)
