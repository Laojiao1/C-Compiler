; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"

@global_smem = external addrspace(3) global [0 x i8], align 16

; Function Attrs: nounwind
define ptx_kernel void @matmul_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8, ptr addrspace(1) readnone captures(none) %9, ptr addrspace(1) readnone captures(none) %10) local_unnamed_addr #0 !dbg !4 {
  %12 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !dbg !7
  %13 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y(), !dbg !8
  %14 = shl i32 %12, 6, !dbg !9
  %15 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !dbg !10
  %16 = lshr i32 %15, 2, !dbg !10
  %17 = and i32 %16, 31, !dbg !10
  %18 = and i32 %15, 120, !dbg !10
  %19 = lshr exact i32 %18, 3, !dbg !10
  %20 = or disjoint i32 %19, 16, !dbg !10
  %21 = and i32 %15, 7, !dbg !10
  %22 = shl nuw nsw i32 %21, 3, !dbg !10
  %23 = or disjoint i32 %17, %14, !dbg !11
  %24 = or disjoint i32 %23, 32, !dbg !11
  %25 = shl nuw nsw i32 %13, 6, !dbg !12
  %26 = or disjoint i32 %22, %25, !dbg !13
  %27 = shl nuw nsw i32 %15, 3, !dbg !14
  %28 = and i32 %27, 24, !dbg !14
  %29 = mul i32 %6, %23, !dbg !15
  %30 = mul i32 %6, %24, !dbg !15
  %31 = add i32 %29, %28, !dbg !16
  %32 = add i32 %30, %28, !dbg !16
  %33 = sext i32 %31 to i64, !dbg !17
  %34 = getelementptr half, ptr addrspace(1) %0, i64 %33, !dbg !17
  %35 = sext i32 %32 to i64, !dbg !17
  %36 = getelementptr half, ptr addrspace(1) %0, i64 %35, !dbg !17
  %37 = mul i32 %7, %19, !dbg !18
  %38 = mul i32 %7, %20, !dbg !18
  %39 = add i32 %37, %26, !dbg !19
  %40 = add i32 %38, %26, !dbg !19
  %41 = sext i32 %39 to i64, !dbg !20
  %42 = getelementptr half, ptr addrspace(1) %1, i64 %41, !dbg !20
  %43 = sext i32 %40 to i64, !dbg !20
  %44 = getelementptr half, ptr addrspace(1) %1, i64 %43, !dbg !20
  %45 = icmp slt i32 %23, %3, !dbg !21
  %46 = icmp slt i32 %24, %3, !dbg !21
  %47 = icmp slt i32 %26, %4, !dbg !22
  %48 = icmp sgt i32 %5, 0, !dbg !23
  %49 = icmp slt i32 %28, %5, !dbg !24
  %50 = and i1 %45, %49, !dbg !25
  %51 = and i1 %46, %49, !dbg !25
  %52 = shl nuw nsw i32 %15, 4, !dbg !26
  %53 = and i32 %52, 2032, !dbg !26
  %54 = shl nuw nsw i32 %15, 1, !dbg !26
  %55 = and i32 %54, 48, !dbg !26
  %56 = xor i32 %53, %55, !dbg !26
  %57 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %56, !dbg !26
  %58 = select i1 %50, i32 16, i32 0, !dbg !26
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, $2;", "r,l,r"(ptr addrspace(3) %57, ptr addrspace(1) %34, i32 %58) #2, !dbg !26
  %59 = or disjoint i32 %56, 2048, !dbg !26
  %60 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %59, !dbg !26
  %61 = select i1 %51, i32 16, i32 0, !dbg !26
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, $2;", "r,l,r"(ptr addrspace(3) nonnull %60, ptr addrspace(1) %36, i32 %61) #2, !dbg !26
  tail call void @llvm.nvvm.cp.async.commit.group(), !dbg !26
  %62 = icmp slt i32 %19, %5, !dbg !27
  %63 = icmp slt i32 %20, %5, !dbg !27
  %64 = and i1 %47, %62, !dbg !28
  %65 = and i1 %63, %48, !dbg !23
  %66 = and i1 %47, %65, !dbg !23
  %67 = and i32 %15, 56, !dbg !29
  %68 = shl nuw nsw i32 %67, 1, !dbg !29
  %69 = xor i32 %53, %68, !dbg !29
  %70 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 8192), i32 %69, !dbg !29
  %71 = select i1 %64, i32 16, i32 0, !dbg !29
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, $2;", "r,l,r"(ptr addrspace(3) %70, ptr addrspace(1) %42, i32 %71) #2, !dbg !29
  %72 = or disjoint i32 %69, 2048, !dbg !29
  %73 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 8192), i32 %72, !dbg !29
  %74 = select i1 %66, i32 16, i32 0, !dbg !29
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, $2;", "r,l,r"(ptr addrspace(3) nonnull %73, ptr addrspace(1) %44, i32 %74) #2, !dbg !29
  tail call void @llvm.nvvm.cp.async.commit.group(), !dbg !29
  %75 = icmp sgt i32 %5, 32, !dbg !23
  %76 = or disjoint i32 %28, 32, !dbg !30
  %77 = or disjoint i32 %19, 32, !dbg !30
  %78 = or disjoint i32 %19, 48, !dbg !30
  %79 = icmp slt i32 %76, %5, !dbg !24
  %80 = and i1 %45, %79, !dbg !25
  %81 = and i1 %46, %79, !dbg !25
  %82 = getelementptr i8, ptr addrspace(1) %34, i64 64, !dbg !31
  %83 = getelementptr i8, ptr addrspace(1) %36, i64 64, !dbg !31
  %84 = and i1 %75, %80, !dbg !23
  %85 = and i1 %75, %81, !dbg !23
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0), !dbg !26
  %86 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 4096), i32 %56, !dbg !26
  %87 = select i1 %84, i32 16, i32 0, !dbg !26
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, $2;", "r,l,r"(ptr addrspace(3) %86, ptr addrspace(1) %82, i32 %87) #2, !dbg !26
  %88 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 4096), i32 %59, !dbg !26
  %89 = select i1 %85, i32 16, i32 0, !dbg !26
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, $2;", "r,l,r"(ptr addrspace(3) nonnull %88, ptr addrspace(1) %83, i32 %89) #2, !dbg !26
  tail call void @llvm.nvvm.cp.async.commit.group(), !dbg !26
  %90 = icmp slt i32 %77, %5, !dbg !27
  %91 = icmp slt i32 %78, %5, !dbg !27
  %92 = and i1 %47, %90, !dbg !28
  %93 = and i1 %47, %91, !dbg !28
  %94 = shl i32 %7, 5, !dbg !32
  %95 = sext i32 %94 to i64, !dbg !33
  %96 = getelementptr half, ptr addrspace(1) %42, i64 %95, !dbg !33
  %97 = getelementptr half, ptr addrspace(1) %44, i64 %95, !dbg !33
  %98 = and i1 %75, %92, !dbg !23
  %99 = and i1 %75, %93, !dbg !23
  %100 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 12288), i32 %69, !dbg !29
  %101 = select i1 %98, i32 16, i32 0, !dbg !29
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, $2;", "r,l,r"(ptr addrspace(3) %100, ptr addrspace(1) %96, i32 %101) #2, !dbg !29
  %102 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 12288), i32 %72, !dbg !29
  %103 = select i1 %99, i32 16, i32 0, !dbg !29
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, $2;", "r,l,r"(ptr addrspace(3) nonnull %102, ptr addrspace(1) %97, i32 %103) #2, !dbg !29
  tail call void @llvm.nvvm.cp.async.commit.group(), !dbg !29
  br i1 %48, label %.lr.ph, label %.._crit_edge_crit_edge, !dbg !23

.._crit_edge_crit_edge:                           ; preds = %11
  %.pre = lshr i32 %15, 1, !dbg !34
  %.pre79 = and i32 %.pre, 16, !dbg !34
  %.pre81 = shl nuw nsw i32 %21, 4, !dbg !34
  br label %._crit_edge, !dbg !23

.lr.ph:                                           ; preds = %11
  %104 = add nsw i32 %5, -64
  %105 = shl nuw nsw i32 %15, 6
  %106 = and i32 %105, 960
  %107 = and i32 %27, 48
  %108 = and i32 %15, 16
  %109 = and i32 %52, 1024
  %110 = or disjoint i32 %106, %107
  %111 = xor i32 %110, %108
  %112 = or disjoint i32 %111, %109
  %113 = xor i32 %112, 32
  %114 = shl nuw nsw i32 %15, 7
  %115 = and i32 %114, 3968
  %116 = shl nuw nsw i32 %21, 4
  %117 = lshr i32 %15, 1
  %118 = and i32 %117, 16
  %119 = xor i32 %116, %118
  %120 = or disjoint i32 %119, %115
  %121 = xor i32 %120, 32
  %122 = xor i32 %120, 64
  %123 = xor i32 %120, 96
  br label %124, !dbg !23

124:                                              ; preds = %.lr.ph, %124
  %125 = phi i32 [ -1, %.lr.ph ], [ %163, %124 ]
  %126 = phi i32 [ 1, %.lr.ph ], [ %297, %124 ]
  %127 = phi float [ 0.000000e+00, %.lr.ph ], [ %256, %124 ]
  %128 = phi float [ 0.000000e+00, %.lr.ph ], [ %257, %124 ]
  %129 = phi float [ 0.000000e+00, %.lr.ph ], [ %258, %124 ]
  %130 = phi float [ 0.000000e+00, %.lr.ph ], [ %259, %124 ]
  %131 = phi float [ 0.000000e+00, %.lr.ph ], [ %261, %124 ]
  %132 = phi float [ 0.000000e+00, %.lr.ph ], [ %262, %124 ]
  %133 = phi float [ 0.000000e+00, %.lr.ph ], [ %263, %124 ]
  %134 = phi float [ 0.000000e+00, %.lr.ph ], [ %264, %124 ]
  %135 = phi float [ 0.000000e+00, %.lr.ph ], [ %266, %124 ]
  %136 = phi float [ 0.000000e+00, %.lr.ph ], [ %267, %124 ]
  %137 = phi float [ 0.000000e+00, %.lr.ph ], [ %268, %124 ]
  %138 = phi float [ 0.000000e+00, %.lr.ph ], [ %269, %124 ]
  %139 = phi float [ 0.000000e+00, %.lr.ph ], [ %271, %124 ]
  %140 = phi float [ 0.000000e+00, %.lr.ph ], [ %272, %124 ]
  %141 = phi float [ 0.000000e+00, %.lr.ph ], [ %273, %124 ]
  %142 = phi float [ 0.000000e+00, %.lr.ph ], [ %274, %124 ]
  %143 = phi float [ 0.000000e+00, %.lr.ph ], [ %276, %124 ]
  %144 = phi float [ 0.000000e+00, %.lr.ph ], [ %277, %124 ]
  %145 = phi float [ 0.000000e+00, %.lr.ph ], [ %278, %124 ]
  %146 = phi float [ 0.000000e+00, %.lr.ph ], [ %279, %124 ]
  %147 = phi float [ 0.000000e+00, %.lr.ph ], [ %281, %124 ]
  %148 = phi float [ 0.000000e+00, %.lr.ph ], [ %282, %124 ]
  %149 = phi float [ 0.000000e+00, %.lr.ph ], [ %283, %124 ]
  %150 = phi float [ 0.000000e+00, %.lr.ph ], [ %284, %124 ]
  %151 = phi float [ 0.000000e+00, %.lr.ph ], [ %286, %124 ]
  %152 = phi float [ 0.000000e+00, %.lr.ph ], [ %287, %124 ]
  %153 = phi float [ 0.000000e+00, %.lr.ph ], [ %288, %124 ]
  %154 = phi float [ 0.000000e+00, %.lr.ph ], [ %289, %124 ]
  %155 = phi float [ 0.000000e+00, %.lr.ph ], [ %291, %124 ]
  %156 = phi float [ 0.000000e+00, %.lr.ph ], [ %292, %124 ]
  %157 = phi float [ 0.000000e+00, %.lr.ph ], [ %293, %124 ]
  %158 = phi float [ 0.000000e+00, %.lr.ph ], [ %294, %124 ]
  %159 = phi i32 [ 0, %.lr.ph ], [ %331, %124 ]
  %160 = icmp slt i32 %159, %104, !dbg !23
  %161 = add i32 %125, 1, !dbg !23
  %162 = icmp sgt i32 %161, 1, !dbg !23
  %163 = select i1 %162, i32 0, i32 %161, !dbg !23
  tail call void @llvm.nvvm.cp.async.wait.group(i32 2), !dbg !26
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0), !dbg !26
  %164 = shl i32 %163, 11, !dbg !26
  %165 = getelementptr half, ptr addrspace(3) @global_smem, i32 %164, !dbg !26
  %166 = getelementptr inbounds nuw i8, ptr addrspace(3) %165, i32 %112, !dbg !26
  %167 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %166), !dbg !26
  %168 = extractvalue { i32, i32, i32, i32 } %167, 0, !dbg !26
  %169 = extractvalue { i32, i32, i32, i32 } %167, 1, !dbg !26
  %170 = extractvalue { i32, i32, i32, i32 } %167, 2, !dbg !26
  %171 = extractvalue { i32, i32, i32, i32 } %167, 3, !dbg !26
  %172 = getelementptr inbounds nuw i8, ptr addrspace(3) %166, i32 2048, !dbg !26
  %173 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) nonnull %172), !dbg !26
  %174 = extractvalue { i32, i32, i32, i32 } %173, 0, !dbg !26
  %175 = extractvalue { i32, i32, i32, i32 } %173, 1, !dbg !26
  %176 = extractvalue { i32, i32, i32, i32 } %173, 2, !dbg !26
  %177 = extractvalue { i32, i32, i32, i32 } %173, 3, !dbg !26
  %178 = getelementptr inbounds nuw i8, ptr addrspace(3) %165, i32 %113, !dbg !26
  %179 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %178), !dbg !26
  %180 = extractvalue { i32, i32, i32, i32 } %179, 0, !dbg !26
  %181 = extractvalue { i32, i32, i32, i32 } %179, 1, !dbg !26
  %182 = extractvalue { i32, i32, i32, i32 } %179, 2, !dbg !26
  %183 = extractvalue { i32, i32, i32, i32 } %179, 3, !dbg !26
  %184 = getelementptr inbounds nuw i8, ptr addrspace(3) %178, i32 2048, !dbg !26
  %185 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) nonnull %184), !dbg !26
  %186 = extractvalue { i32, i32, i32, i32 } %185, 0, !dbg !26
  %187 = extractvalue { i32, i32, i32, i32 } %185, 1, !dbg !26
  %188 = extractvalue { i32, i32, i32, i32 } %185, 2, !dbg !26
  %189 = extractvalue { i32, i32, i32, i32 } %185, 3, !dbg !26
  %190 = getelementptr half, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 8192), i32 %164, !dbg !29
  %191 = getelementptr inbounds nuw i8, ptr addrspace(3) %190, i32 %120, !dbg !29
  %192 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16.p3(ptr addrspace(3) %191), !dbg !29
  %193 = extractvalue { i32, i32, i32, i32 } %192, 0, !dbg !29
  %194 = extractvalue { i32, i32, i32, i32 } %192, 1, !dbg !29
  %195 = extractvalue { i32, i32, i32, i32 } %192, 2, !dbg !29
  %196 = extractvalue { i32, i32, i32, i32 } %192, 3, !dbg !29
  %197 = getelementptr inbounds nuw i8, ptr addrspace(3) %190, i32 %121, !dbg !29
  %198 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16.p3(ptr addrspace(3) %197), !dbg !29
  %199 = extractvalue { i32, i32, i32, i32 } %198, 0, !dbg !29
  %200 = extractvalue { i32, i32, i32, i32 } %198, 1, !dbg !29
  %201 = extractvalue { i32, i32, i32, i32 } %198, 2, !dbg !29
  %202 = extractvalue { i32, i32, i32, i32 } %198, 3, !dbg !29
  %203 = getelementptr inbounds nuw i8, ptr addrspace(3) %190, i32 %122, !dbg !29
  %204 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16.p3(ptr addrspace(3) %203), !dbg !29
  %205 = extractvalue { i32, i32, i32, i32 } %204, 0, !dbg !29
  %206 = extractvalue { i32, i32, i32, i32 } %204, 1, !dbg !29
  %207 = extractvalue { i32, i32, i32, i32 } %204, 2, !dbg !29
  %208 = extractvalue { i32, i32, i32, i32 } %204, 3, !dbg !29
  %209 = getelementptr inbounds nuw i8, ptr addrspace(3) %190, i32 %123, !dbg !29
  %210 = tail call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16.p3(ptr addrspace(3) %209), !dbg !29
  %211 = extractvalue { i32, i32, i32, i32 } %210, 0, !dbg !29
  %212 = extractvalue { i32, i32, i32, i32 } %210, 1, !dbg !29
  %213 = extractvalue { i32, i32, i32, i32 } %210, 2, !dbg !29
  %214 = extractvalue { i32, i32, i32, i32 } %210, 3, !dbg !29
  %215 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %127, float %128, float %129, float %130, i32 %168, i32 %169, i32 %170, i32 %171, i32 %193, i32 %194) #2, !dbg !35
  %216 = extractvalue { float, float, float, float } %215, 0, !dbg !35
  %217 = extractvalue { float, float, float, float } %215, 1, !dbg !35
  %218 = extractvalue { float, float, float, float } %215, 2, !dbg !35
  %219 = extractvalue { float, float, float, float } %215, 3, !dbg !35
  %220 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %131, float %132, float %133, float %134, i32 %168, i32 %169, i32 %170, i32 %171, i32 %199, i32 %200) #2, !dbg !35
  %221 = extractvalue { float, float, float, float } %220, 0, !dbg !35
  %222 = extractvalue { float, float, float, float } %220, 1, !dbg !35
  %223 = extractvalue { float, float, float, float } %220, 2, !dbg !35
  %224 = extractvalue { float, float, float, float } %220, 3, !dbg !35
  %225 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %135, float %136, float %137, float %138, i32 %168, i32 %169, i32 %170, i32 %171, i32 %205, i32 %206) #2, !dbg !35
  %226 = extractvalue { float, float, float, float } %225, 0, !dbg !35
  %227 = extractvalue { float, float, float, float } %225, 1, !dbg !35
  %228 = extractvalue { float, float, float, float } %225, 2, !dbg !35
  %229 = extractvalue { float, float, float, float } %225, 3, !dbg !35
  %230 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %139, float %140, float %141, float %142, i32 %168, i32 %169, i32 %170, i32 %171, i32 %211, i32 %212) #2, !dbg !35
  %231 = extractvalue { float, float, float, float } %230, 0, !dbg !35
  %232 = extractvalue { float, float, float, float } %230, 1, !dbg !35
  %233 = extractvalue { float, float, float, float } %230, 2, !dbg !35
  %234 = extractvalue { float, float, float, float } %230, 3, !dbg !35
  %235 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %143, float %144, float %145, float %146, i32 %174, i32 %175, i32 %176, i32 %177, i32 %193, i32 %194) #2, !dbg !35
  %236 = extractvalue { float, float, float, float } %235, 0, !dbg !35
  %237 = extractvalue { float, float, float, float } %235, 1, !dbg !35
  %238 = extractvalue { float, float, float, float } %235, 2, !dbg !35
  %239 = extractvalue { float, float, float, float } %235, 3, !dbg !35
  %240 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %147, float %148, float %149, float %150, i32 %174, i32 %175, i32 %176, i32 %177, i32 %199, i32 %200) #2, !dbg !35
  %241 = extractvalue { float, float, float, float } %240, 0, !dbg !35
  %242 = extractvalue { float, float, float, float } %240, 1, !dbg !35
  %243 = extractvalue { float, float, float, float } %240, 2, !dbg !35
  %244 = extractvalue { float, float, float, float } %240, 3, !dbg !35
  %245 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %151, float %152, float %153, float %154, i32 %174, i32 %175, i32 %176, i32 %177, i32 %205, i32 %206) #2, !dbg !35
  %246 = extractvalue { float, float, float, float } %245, 0, !dbg !35
  %247 = extractvalue { float, float, float, float } %245, 1, !dbg !35
  %248 = extractvalue { float, float, float, float } %245, 2, !dbg !35
  %249 = extractvalue { float, float, float, float } %245, 3, !dbg !35
  %250 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %155, float %156, float %157, float %158, i32 %174, i32 %175, i32 %176, i32 %177, i32 %211, i32 %212) #2, !dbg !35
  %251 = extractvalue { float, float, float, float } %250, 0, !dbg !35
  %252 = extractvalue { float, float, float, float } %250, 1, !dbg !35
  %253 = extractvalue { float, float, float, float } %250, 2, !dbg !35
  %254 = extractvalue { float, float, float, float } %250, 3, !dbg !35
  %255 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %216, float %217, float %218, float %219, i32 %180, i32 %181, i32 %182, i32 %183, i32 %195, i32 %196) #2, !dbg !35
  %256 = extractvalue { float, float, float, float } %255, 0, !dbg !35
  %257 = extractvalue { float, float, float, float } %255, 1, !dbg !35
  %258 = extractvalue { float, float, float, float } %255, 2, !dbg !35
  %259 = extractvalue { float, float, float, float } %255, 3, !dbg !35
  %260 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %221, float %222, float %223, float %224, i32 %180, i32 %181, i32 %182, i32 %183, i32 %201, i32 %202) #2, !dbg !35
  %261 = extractvalue { float, float, float, float } %260, 0, !dbg !35
  %262 = extractvalue { float, float, float, float } %260, 1, !dbg !35
  %263 = extractvalue { float, float, float, float } %260, 2, !dbg !35
  %264 = extractvalue { float, float, float, float } %260, 3, !dbg !35
  %265 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %226, float %227, float %228, float %229, i32 %180, i32 %181, i32 %182, i32 %183, i32 %207, i32 %208) #2, !dbg !35
  %266 = extractvalue { float, float, float, float } %265, 0, !dbg !35
  %267 = extractvalue { float, float, float, float } %265, 1, !dbg !35
  %268 = extractvalue { float, float, float, float } %265, 2, !dbg !35
  %269 = extractvalue { float, float, float, float } %265, 3, !dbg !35
  %270 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %231, float %232, float %233, float %234, i32 %180, i32 %181, i32 %182, i32 %183, i32 %213, i32 %214) #2, !dbg !35
  %271 = extractvalue { float, float, float, float } %270, 0, !dbg !35
  %272 = extractvalue { float, float, float, float } %270, 1, !dbg !35
  %273 = extractvalue { float, float, float, float } %270, 2, !dbg !35
  %274 = extractvalue { float, float, float, float } %270, 3, !dbg !35
  %275 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %236, float %237, float %238, float %239, i32 %186, i32 %187, i32 %188, i32 %189, i32 %195, i32 %196) #2, !dbg !35
  %276 = extractvalue { float, float, float, float } %275, 0, !dbg !35
  %277 = extractvalue { float, float, float, float } %275, 1, !dbg !35
  %278 = extractvalue { float, float, float, float } %275, 2, !dbg !35
  %279 = extractvalue { float, float, float, float } %275, 3, !dbg !35
  %280 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %241, float %242, float %243, float %244, i32 %186, i32 %187, i32 %188, i32 %189, i32 %201, i32 %202) #2, !dbg !35
  %281 = extractvalue { float, float, float, float } %280, 0, !dbg !35
  %282 = extractvalue { float, float, float, float } %280, 1, !dbg !35
  %283 = extractvalue { float, float, float, float } %280, 2, !dbg !35
  %284 = extractvalue { float, float, float, float } %280, 3, !dbg !35
  %285 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %246, float %247, float %248, float %249, i32 %186, i32 %187, i32 %188, i32 %189, i32 %207, i32 %208) #2, !dbg !35
  %286 = extractvalue { float, float, float, float } %285, 0, !dbg !35
  %287 = extractvalue { float, float, float, float } %285, 1, !dbg !35
  %288 = extractvalue { float, float, float, float } %285, 2, !dbg !35
  %289 = extractvalue { float, float, float, float } %285, 3, !dbg !35
  %290 = tail call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %251, float %252, float %253, float %254, i32 %186, i32 %187, i32 %188, i32 %189, i32 %213, i32 %214) #2, !dbg !35
  %291 = extractvalue { float, float, float, float } %290, 0, !dbg !35
  %292 = extractvalue { float, float, float, float } %290, 1, !dbg !35
  %293 = extractvalue { float, float, float, float } %290, 2, !dbg !35
  %294 = extractvalue { float, float, float, float } %290, 3, !dbg !35
  %295 = add i32 %126, 1, !dbg !23
  %296 = icmp sgt i32 %295, 1, !dbg !23
  %297 = select i1 %296, i32 0, i32 %295, !dbg !23
  %298 = add i32 %159, 64, !dbg !23
  %299 = or disjoint i32 %298, %28, !dbg !30
  %300 = or disjoint i32 %298, %19, !dbg !30
  %301 = or disjoint i32 %298, %20, !dbg !30
  %302 = icmp slt i32 %299, %5, !dbg !24
  %303 = and i1 %45, %302, !dbg !25
  %304 = and i1 %46, %302, !dbg !25
  %305 = sext i32 %298 to i64, !dbg !31
  %306 = getelementptr half, ptr addrspace(1) %34, i64 %305, !dbg !31
  %307 = getelementptr half, ptr addrspace(1) %36, i64 %305, !dbg !31
  %308 = shl i32 %297, 11, !dbg !26
  %309 = getelementptr half, ptr addrspace(3) @global_smem, i32 %308, !dbg !26
  %310 = and i1 %160, %303, !dbg !23
  %311 = and i1 %160, %304, !dbg !23
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0), !dbg !26
  %312 = getelementptr inbounds nuw i8, ptr addrspace(3) %309, i32 %56, !dbg !26
  %313 = select i1 %310, i32 16, i32 0, !dbg !26
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, $2;", "r,l,r"(ptr addrspace(3) %312, ptr addrspace(1) %306, i32 %313) #2, !dbg !26
  %314 = getelementptr inbounds nuw i8, ptr addrspace(3) %309, i32 %59, !dbg !26
  %315 = select i1 %311, i32 16, i32 0, !dbg !26
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, $2;", "r,l,r"(ptr addrspace(3) nonnull %314, ptr addrspace(1) %307, i32 %315) #2, !dbg !26
  tail call void @llvm.nvvm.cp.async.commit.group(), !dbg !26
  %316 = icmp slt i32 %300, %5, !dbg !27
  %317 = icmp slt i32 %301, %5, !dbg !27
  %318 = and i1 %47, %316, !dbg !28
  %319 = and i1 %47, %317, !dbg !28
  %320 = mul i32 %298, %7, !dbg !32
  %321 = sext i32 %320 to i64, !dbg !33
  %322 = getelementptr half, ptr addrspace(1) %42, i64 %321, !dbg !33
  %323 = getelementptr half, ptr addrspace(1) %44, i64 %321, !dbg !33
  %324 = getelementptr half, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 8192), i32 %308, !dbg !29
  %325 = and i1 %160, %318, !dbg !23
  %326 = and i1 %160, %319, !dbg !23
  %327 = getelementptr inbounds nuw i8, ptr addrspace(3) %324, i32 %69, !dbg !29
  %328 = select i1 %325, i32 16, i32 0, !dbg !29
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, $2;", "r,l,r"(ptr addrspace(3) %327, ptr addrspace(1) %322, i32 %328) #2, !dbg !29
  %329 = getelementptr inbounds nuw i8, ptr addrspace(3) %324, i32 %72, !dbg !29
  %330 = select i1 %326, i32 16, i32 0, !dbg !29
  tail call void asm sideeffect "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, $2;", "r,l,r"(ptr addrspace(3) nonnull %329, ptr addrspace(1) %323, i32 %330) #2, !dbg !29
  tail call void @llvm.nvvm.cp.async.commit.group(), !dbg !29
  %331 = add i32 %159, 32, !dbg !23
  %332 = icmp slt i32 %331, %5, !dbg !23
  br i1 %332, label %124, label %._crit_edge.loopexit, !dbg !23

._crit_edge.loopexit:                             ; preds = %124
  %333 = insertelement <32 x float> poison, float %256, i64 0, !dbg !36
  %334 = insertelement <32 x float> %333, float %257, i64 1, !dbg !36
  %335 = insertelement <32 x float> %334, float %258, i64 2, !dbg !36
  %336 = insertelement <32 x float> %335, float %259, i64 3, !dbg !36
  %337 = insertelement <32 x float> %336, float %261, i64 4, !dbg !36
  %338 = insertelement <32 x float> %337, float %262, i64 5, !dbg !36
  %339 = insertelement <32 x float> %338, float %263, i64 6, !dbg !36
  %340 = insertelement <32 x float> %339, float %264, i64 7, !dbg !36
  %341 = insertelement <32 x float> %340, float %266, i64 8, !dbg !36
  %342 = insertelement <32 x float> %341, float %267, i64 9, !dbg !36
  %343 = insertelement <32 x float> %342, float %268, i64 10, !dbg !36
  %344 = insertelement <32 x float> %343, float %269, i64 11, !dbg !36
  %345 = insertelement <32 x float> %344, float %271, i64 12, !dbg !36
  %346 = insertelement <32 x float> %345, float %272, i64 13, !dbg !36
  %347 = insertelement <32 x float> %346, float %273, i64 14, !dbg !36
  %348 = insertelement <32 x float> %347, float %274, i64 15, !dbg !36
  %349 = insertelement <32 x float> %348, float %276, i64 16, !dbg !36
  %350 = insertelement <32 x float> %349, float %277, i64 17, !dbg !36
  %351 = insertelement <32 x float> %350, float %278, i64 18, !dbg !36
  %352 = insertelement <32 x float> %351, float %279, i64 19, !dbg !36
  %353 = insertelement <32 x float> %352, float %281, i64 20, !dbg !36
  %354 = insertelement <32 x float> %353, float %282, i64 21, !dbg !36
  %355 = insertelement <32 x float> %354, float %283, i64 22, !dbg !36
  %356 = insertelement <32 x float> %355, float %284, i64 23, !dbg !36
  %357 = insertelement <32 x float> %356, float %286, i64 24, !dbg !36
  %358 = insertelement <32 x float> %357, float %287, i64 25, !dbg !36
  %359 = insertelement <32 x float> %358, float %288, i64 26, !dbg !36
  %360 = insertelement <32 x float> %359, float %289, i64 27, !dbg !36
  %361 = insertelement <32 x float> %360, float %291, i64 28, !dbg !36
  %362 = insertelement <32 x float> %361, float %292, i64 29, !dbg !36
  %363 = insertelement <32 x float> %362, float %293, i64 30, !dbg !36
  %364 = insertelement <32 x float> %363, float %294, i64 31, !dbg !36
  %365 = fptrunc <32 x float> %364 to <32 x half>, !dbg !36
  %366 = bitcast <32 x half> %365 to <16 x i32>, !dbg !34
  br label %._crit_edge, !dbg !11

._crit_edge:                                      ; preds = %._crit_edge.loopexit, %.._crit_edge_crit_edge
  %.pre-phi82 = phi i32 [ %.pre81, %.._crit_edge_crit_edge ], [ %116, %._crit_edge.loopexit ], !dbg !34
  %.pre-phi80 = phi i32 [ %.pre79, %.._crit_edge_crit_edge ], [ %118, %._crit_edge.loopexit ], !dbg !34
  %.bc144 = phi <16 x i32> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %366, %._crit_edge.loopexit ]
  %367 = or disjoint i32 %78, %14, !dbg !11
  %368 = or disjoint i32 %77, %14, !dbg !11
  %369 = or disjoint i32 %20, %14, !dbg !11
  %370 = or disjoint i32 %19, %14, !dbg !11
  tail call void @llvm.nvvm.cp.async.wait.group(i32 0), !dbg !23
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0), !dbg !23
  %371 = mul i32 %8, %370, !dbg !37
  %372 = mul i32 %8, %369, !dbg !37
  %373 = mul i32 %8, %368, !dbg !37
  %374 = mul i32 %8, %367, !dbg !37
  %375 = add i32 %371, %26, !dbg !38
  %376 = add i32 %372, %26, !dbg !38
  %377 = add i32 %373, %26, !dbg !38
  %378 = add i32 %374, %26, !dbg !38
  %379 = sext i32 %375 to i64, !dbg !39
  %380 = getelementptr half, ptr addrspace(1) %2, i64 %379, !dbg !39
  %381 = sext i32 %376 to i64, !dbg !39
  %382 = getelementptr half, ptr addrspace(1) %2, i64 %381, !dbg !39
  %383 = sext i32 %377 to i64, !dbg !39
  %384 = getelementptr half, ptr addrspace(1) %2, i64 %383, !dbg !39
  %385 = sext i32 %378 to i64, !dbg !39
  %386 = getelementptr half, ptr addrspace(1) %2, i64 %385, !dbg !39
  %387 = icmp slt i32 %370, %3, !dbg !40
  %388 = icmp slt i32 %369, %3, !dbg !40
  %389 = icmp slt i32 %368, %3, !dbg !40
  %390 = icmp slt i32 %367, %3, !dbg !40
  %391 = and i1 %387, %47, !dbg !41
  %392 = and i1 %388, %47, !dbg !41
  %393 = and i1 %389, %47, !dbg !41
  %394 = and i1 %390, %47, !dbg !41
  %395 = shl nuw nsw i32 %21, 9, !dbg !34
  %396 = and i32 %52, 240, !dbg !34
  %397 = and i32 %54, 32, !dbg !34
  %398 = shl nuw nsw i32 %15, 2, !dbg !34
  %399 = and i32 %398, 256, !dbg !34
  %400 = or disjoint i32 %395, %399, !dbg !34
  %401 = xor i32 %396, %397, !dbg !34
  %402 = xor i32 %401, %.pre-phi80, !dbg !34
  %403 = or disjoint i32 %402, %400, !dbg !34
  %404 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %403, !dbg !34
  %.extract115 = extractelement <16 x i32> %.bc144, i64 0, !dbg !34
  %.extract117 = extractelement <16 x i32> %.bc144, i64 1, !dbg !34
  %.extract119 = extractelement <16 x i32> %.bc144, i64 2, !dbg !34
  %.extract121 = extractelement <16 x i32> %.bc144, i64 3, !dbg !34
  tail call void @llvm.nvvm.stmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %404, i32 %.extract115, i32 %.extract117, i32 %.extract119, i32 %.extract121), !dbg !34
  %405 = xor i32 %403, 64, !dbg !34
  %406 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %405, !dbg !34
  %.extract123 = extractelement <16 x i32> %.bc144, i64 4, !dbg !34
  %.extract125 = extractelement <16 x i32> %.bc144, i64 5, !dbg !34
  %.extract127 = extractelement <16 x i32> %.bc144, i64 6, !dbg !34
  %.extract129 = extractelement <16 x i32> %.bc144, i64 7, !dbg !34
  tail call void @llvm.nvvm.stmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %406, i32 %.extract123, i32 %.extract125, i32 %.extract127, i32 %.extract129), !dbg !34
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0), !dbg !34
  %407 = shl nuw nsw i32 %67, 6, !dbg !34
  %408 = shl nuw nsw i32 %18, 1, !dbg !34
  %409 = or disjoint i32 %407, %.pre-phi82, !dbg !34
  %410 = xor i32 %409, %408, !dbg !34
  %411 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %410, !dbg !34
  %412 = load <4 x i32>, ptr addrspace(3) %411, align 16, !dbg !34
  %413 = getelementptr inbounds nuw i8, ptr addrspace(3) %411, i32 256, !dbg !34
  %414 = load <4 x i32>, ptr addrspace(3) %413, align 16, !dbg !34
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0), !dbg !34
  %.extract131 = extractelement <16 x i32> %.bc144, i64 8, !dbg !34
  %.extract133 = extractelement <16 x i32> %.bc144, i64 9, !dbg !34
  %.extract135 = extractelement <16 x i32> %.bc144, i64 10, !dbg !34
  %.extract137 = extractelement <16 x i32> %.bc144, i64 11, !dbg !34
  tail call void @llvm.nvvm.stmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %404, i32 %.extract131, i32 %.extract133, i32 %.extract135, i32 %.extract137), !dbg !34
  %.extract139 = extractelement <16 x i32> %.bc144, i64 12, !dbg !34
  %.extract141 = extractelement <16 x i32> %.bc144, i64 13, !dbg !34
  %.extract143 = extractelement <16 x i32> %.bc144, i64 14, !dbg !34
  %.extract145 = extractelement <16 x i32> %.bc144, i64 15, !dbg !34
  tail call void @llvm.nvvm.stmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %406, i32 %.extract139, i32 %.extract141, i32 %.extract143, i32 %.extract145), !dbg !34
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0), !dbg !34
  %415 = load <4 x i32>, ptr addrspace(3) %411, align 16, !dbg !34
  %416 = load <4 x i32>, ptr addrspace(3) %413, align 16, !dbg !34
  %.extract = extractelement <4 x i32> %412, i64 0, !dbg !34
  %.extract49 = extractelement <4 x i32> %412, i64 1, !dbg !34
  %.extract51 = extractelement <4 x i32> %412, i64 2, !dbg !34
  %.extract53 = extractelement <4 x i32> %412, i64 3, !dbg !34
  tail call void asm sideeffect "@$5 st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l,b"(i32 %.extract, i32 %.extract49, i32 %.extract51, i32 %.extract53, ptr addrspace(1) %380, i1 %391) #2, !dbg !34
  %.extract55 = extractelement <4 x i32> %414, i64 0, !dbg !34
  %.extract57 = extractelement <4 x i32> %414, i64 1, !dbg !34
  %.extract59 = extractelement <4 x i32> %414, i64 2, !dbg !34
  %.extract61 = extractelement <4 x i32> %414, i64 3, !dbg !34
  tail call void asm sideeffect "@$5 st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l,b"(i32 %.extract55, i32 %.extract57, i32 %.extract59, i32 %.extract61, ptr addrspace(1) %382, i1 %392) #2, !dbg !34
  %.extract63 = extractelement <4 x i32> %415, i64 0, !dbg !34
  %.extract65 = extractelement <4 x i32> %415, i64 1, !dbg !34
  %.extract67 = extractelement <4 x i32> %415, i64 2, !dbg !34
  %.extract69 = extractelement <4 x i32> %415, i64 3, !dbg !34
  tail call void asm sideeffect "@$5 st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l,b"(i32 %.extract63, i32 %.extract65, i32 %.extract67, i32 %.extract69, ptr addrspace(1) %384, i1 %393) #2, !dbg !34
  %.extract71 = extractelement <4 x i32> %416, i64 0, !dbg !34
  %.extract73 = extractelement <4 x i32> %416, i64 1, !dbg !34
  %.extract75 = extractelement <4 x i32> %416, i64 2, !dbg !34
  %.extract77 = extractelement <4 x i32> %416, i64 3, !dbg !34
  tail call void asm sideeffect "@$5 st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l,b"(i32 %.extract71, i32 %.extract73, i32 %.extract75, i32 %.extract77, ptr addrspace(1) %386, i1 %394) #2, !dbg !34
  ret void, !dbg !42
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 65535) i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nounwind
declare void @llvm.nvvm.cp.async.commit.group() #2

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #3

; Function Attrs: nounwind
declare void @llvm.nvvm.cp.async.wait.group(i32 immarg) #2

; Function Attrs: nocallback nounwind memory(argmem: write)
declare void @llvm.nvvm.stmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) writeonly captures(none), i32, i32, i32, i32) #4

; Function Attrs: nocallback nofree nounwind memory(argmem: read)
declare { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) readonly captures(none)) #5

; Function Attrs: nocallback nofree nounwind memory(argmem: read)
declare { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16.p3(ptr addrspace(3) readonly captures(none)) #5

attributes #0 = { nounwind "nvvm.reqntid"="128" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nounwind }
attributes #3 = { convergent nocallback nounwind }
attributes #4 = { nocallback nounwind memory(argmem: write) }
attributes #5 = { nocallback nofree nounwind memory(argmem: read) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "triton", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "dump_matmul.py", directory: "/mnt/e/Code/C++Compiler/4_GPU_DeepDive/01_triton_lowering/matmul_ir")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 4, !"nvvm-reflect-ftz", i32 1}
!4 = distinct !DISubprogram(name: "matmul_kernel", linkageName: "matmul_kernel", scope: !1, file: !1, line: 8, type: !5, scopeLine: 8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DISubroutineType(cc: DW_CC_normal, types: !6)
!6 = !{}
!7 = !DILocation(line: 26, column: 26, scope: !4)
!8 = !DILocation(line: 27, column: 26, scope: !4)
!9 = !DILocation(line: 30, column: 21, scope: !4)
!10 = !DILocation(line: 30, column: 44, scope: !4)
!11 = !DILocation(line: 30, column: 31, scope: !4)
!12 = !DILocation(line: 31, column: 21, scope: !4)
!13 = !DILocation(line: 31, column: 31, scope: !4)
!14 = !DILocation(line: 34, column: 26, scope: !4)
!15 = !DILocation(line: 38, column: 26, scope: !4)
!16 = !DILocation(line: 39, column: 8, scope: !4)
!17 = !DILocation(line: 38, column: 8, scope: !4)
!18 = !DILocation(line: 44, column: 26, scope: !4)
!19 = !DILocation(line: 45, column: 8, scope: !4)
!20 = !DILocation(line: 44, column: 8, scope: !4)
!21 = !DILocation(line: 61, column: 35, scope: !4)
!22 = !DILocation(line: 71, column: 35, scope: !4)
!23 = !DILocation(line: 55, column: 25, scope: !4)
!24 = !DILocation(line: 62, column: 38, scope: !4)
!25 = !DILocation(line: 62, column: 17, scope: !4)
!26 = !DILocation(line: 59, column: 12, scope: !4)
!27 = !DILocation(line: 70, column: 38, scope: !4)
!28 = !DILocation(line: 71, column: 17, scope: !4)
!29 = !DILocation(line: 68, column: 12, scope: !4)
!30 = !DILocation(line: 56, column: 24, scope: !4)
!31 = !DILocation(line: 59, column: 21, scope: !4)
!32 = !DILocation(line: 68, column: 25, scope: !4)
!33 = !DILocation(line: 68, column: 21, scope: !4)
!34 = !DILocation(line: 95, column: 8, scope: !4)
!35 = !DILocation(line: 78, column: 25, scope: !4)
!36 = !DILocation(line: 81, column: 15, scope: !4)
!37 = !DILocation(line: 84, column: 26, scope: !4)
!38 = !DILocation(line: 85, column: 8, scope: !4)
!39 = !DILocation(line: 84, column: 8, scope: !4)
!40 = !DILocation(line: 89, column: 27, scope: !4)
!41 = !DILocation(line: 90, column: 9, scope: !4)
!42 = !DILocation(line: 93, column: 4, scope: !4)
