# 隐式 non-ideal MHD：完整旋钮清单（单一来源）

*2026-10-25 生成/更新。来源：四个头文件的 `args.get` 扫描 + 运行目录与仓库内每一份 `*.par` 的设置扫描 + **成员名**在非注释代码里的使用计数。*

**判据**：`uses` 用**成员名**计数（不是键后缀——踩过 dead-ends M19）。退役判据是两面的：**无任何 par 设置** ∪ **无任何 pgen/测试引用其成员**。

## 已退役（归档于 `backup/nonideal-retired-2026-10-25/`）

| 键/模式 | 退役日期 | 原因 | 归档 |
|---|---|---|---|
| `mg_cc_ene = 1`（隐式+显式通量式） | 2026-10-25 | B1 无 `A_f>0` 门 / B2 时间两端不同平均 / B3 横向场用格心值喂两个面；极环实测小 84–1.6e4× | `mg_ni_ene_k.h.retired`、`mg_ni_ene_mode1_branch.cc.retired`、`ni_ene_du_e_loop.cc.retired` |
| `mg_cc_wsync = 0 / ≤-1` + `mg_cc_hydro(/ _k)` | 2026-10-25 | partial refresh 只写 rho/v/bcc，`w[ene]`(=p) 与组分留在 RK 中点（相对化学步陈旧） | `mg_cc_hydro_wsync0.h.retired` |

**默认值变更**：`mg_cc_ene` 默认 **1 → 2**（2 是经 P1 证明与 EOS 逐位一致的形式）。



## 设置部分已删（值写死）——2026-10-25

这些键在**整个运行语料（1629 个 par）里 0 次被设置**，且成员都有与预期一致的初始化，
故**删去 `args.get` 读入、把值写死**（行为逐位不变，已由回归确认）。**保留**了对应成员与分支。

| 键（已删读入） | 写死值 | 成员/初始化 | 备注 |
|---|---|---|---|
| ~~`mg_cc_time`~~ | — | — | **已恢复读入**（2026-10-25 同日）：CN/BDF2 是**要保留**的能力，写死 BE 会让它们不可达，与保留的意图相反 |
| `mg_cc_bhat` | 0 | `bhat = 0` | b̂ 取自场 |
| `mg_cc_cgrad` | true | `mg_operation_t::cgrad = true` | 对角耦合梯度 |
| `mg_cc_romega` | 1 | `r_om = 1` | Richardson 不阻尼 |
| `mg_cc_rem_eps` | 1e-6 | `rem_eps = 1e-6` | 余量源阈 |
| `mg_cc_rmin` | 8 | `r_min = 8` | 停滞退出前最少 pass 数 |
| `mg_cc_eta_a` | 0 | `eta_a = 0` | 均匀 η 的**各向异性**幅度（`mg_cc_eta_var` provider 才是基准） |
| `mg_cc_eta_a_z` | 0 | `eta_a_z = 0` | 其 z-剖面线性项 |
| `mg_cc_eta_o_z` | 0 | `eta_o_z = 0` | Ohmic 的 z-剖面线性项 |
| `mg_cc_aniso_base` | **1** | `aniso_base = 1` | ⚠️ 其 `0` 档（标量预条件 + 各向异性精确行）实测**差 2.9e4**（否证）⇒ 整档退役 |
| `mg_cc_divclean` | 0 | `dvg_clean = 0` | **实测有害**（确定性极区栈：投影前 `max\|D.B\|` 2.2e-2 → **2.5e3**，dt 从 2.6e-4 塌到 2.4e-5 并最终 fuse）⇒ 退役；粗层清理路径标 `UNREACHABLE`，**注释保留读入以便一键拉回** |
| `proj_mg` / `proj_mg_v` / `proj_mg_dtol` / `proj_mg_sc` | 0 / 20 / 1e-9 / 0 | 成员初始化 | **投影的 MG 求解器退役**（基准走 SOR）；⚠️ **23 个历史 par 用过 `proj_mg`** ⇒ 那些算例今后落到 SOR 路径（这是退役的本意）；MG 启动块已标 `UNREACHABLE`，待归档轮切除 |
| `mg_cc_rharm` | 0（算术平均） | `r_harm = 0` | 尖锐跳跃的调和平均限制在 1e8 对比测试台**无收益** ⇒ 退役；`r_harm` **参数已从 `mg_cc_rcoarse`/`_k` 移除**（`h`/`h_ok` 累加器留待归档轮） |

> ⚠️ **`mg_cc_eta_o` 是特例**：它的默认表达式是**旧的 `mg_cc_eta` 值**（扇出），所以不能直接删读入——
> 那样会让 31 个使用旧键的 par 的 `eta_o` 归零。因此写成 **`eta_o = eta;`**（保留扇出），行为逐位不变。

**已恢复**：`mg_cc_time`（CN/BDF2 **保留**）。其余五项保持写死。

**`proj_mg` 是做什么的（判读材料）**：它是**同一个散度投影**（解 `∇²φ = ∇·B*`，再 `B -= ∇φ`）的
**另一种 Poisson 求解器**——把 RB-SOR（`proj_iter × 2` 色，默认 2000）换成**共享多重网格 V-cycle**。
*动机（`proj-multigrid-plan.md`）*：3D `64×512×8` 上 RB-SOR 收敛率不够（`resid ~1.4e-4`、`proj ~20–50`）
且每步约 **4000 次 kernel launch**；V-cycle 只需 O(10) 次循环。
⚠️ **但该文档 §0 自己标了警告**：那些 3D 数是**在被否证的掩码**（dead-ends D43）上测的；
**用正确掩码时 3D 已是 `resid 1.3e-18`、`proj 2.2e-9`** ⇒ **原本的动机可能已经不成立**。
*子旋钮*：`proj_mg_v`（V-cycle 预算 20）、`proj_mg_dtol`（相对缺陷退出 1e-9）、
`proj_mg_sc`（粗层模板尺度指数；实测 0 收敛、1 破坏求解）。
*现状*：读入已注释、调用点标 `UNREACHABLE` ⇒ **`proj_mg = 1` 目前无效**（需按 RESTORE 标记恢复）。
**建议**：先做一次 SOR vs MG 的现测（当前 build、生产分辨率），再决定切除或保留——它的存在理由本就是"SOR 停滞"。

**保留（按你的指示）**：显式 / STS / Hall 一族 —— `mode`、`sts`、`sts_nu`、`bc_edge`、`hall_chyp`、
`hall_upw`、`eta_o`、`eta_a`、`eta_h`、`hall_l`。

## 最终基准的模式集（**以三个权威算例的基础 par 为准**，不看历史测试 par）

基准 = `prod/`（生产极区）、`t1ppd_T1/`（确定性极区）、`npole/`（非极区生产）三处的
`m0p / chk / runp6 / meshp / m3sor (+ t3dstab)`。历史目录（1629 个 par）**只用于判断
"某分支是否曾被验证过"**，不作为基准。

### A. 基准**采用**的模式

| 族 | 键 | 基准值 | 说明 |
|---|---|---|---|
| 求解器 | `mg_cc` | **1** | 隐式 curl-curl（生产腿） |
| | `mode` / `sts` | 1 / 0 | 显式腿按 split enrolled（Hall/STS 用），RKC 子步关 |
| | `mg_cc_line` | **2** | 交替 zebra 线松弛 |
| | `proj_mg` | **0** | 散度投影走 **SOR**（非 MG 版） |
| | `mg_cc_proj` / `_iter` / `_projchk` | 1 / 2000 / 1 | 投影 + 每步残差自检 |
| η | `mg_cc_eta_var` | **1** | 逐格 provider（`chem_eta = 1`，`eta_max = 0.3`） |
| | `eta_o` / `eta_a` / `eta_h` / `hall_l` | 0 / 0 / 0 / 0 | 由 provider 覆盖；**Hall 关** |
| BC | `mg_cc_bc` | **1** | essential |
| | `mg_cc_bclose` | **0** ⚠️ | 基准**正是** `bc=1 + bclose=0` 这一对；dead-ends D11 / 开放问题 P4 记录它在均匀球坐标上 FPE |
| 能量 | `mg_cc_ene` | **2**（新默认） | 实测式；P1 证 `rho/mom` 逐位一致 |
| | `mg_cc_ene_heat` | 1 | 局部 `dt(η_o|J|²+η_a|J_⊥|²)` |
| 时间 | `mg_cc_time` | 0（默认 BE） | 基准未设该键 |
| 极区 | `pole_rebin_bf` / `pole_rebin_lvl` | 1 / 6 | |
| | `va_cap` | **5**（生产）/ 3（确定性栈） | 极冠 `v_A` 上限 |
| | `dt_collapse` | 2e-5 | dt fuse |

### B. 本次**已退役**（归档于 `backup/nonideal-retired-2026-10-25/`）

| 模式 | 退役日 | 原因 |
|---|---|---|
| `mg_cc_ene = 1`（两腿的通量式） | 2026-10-25 | B1 无 `A_f>0` 门 / B2 时间两端不同平均 / B3 横向场用格心值喂两个面；极环小 84–1.6e4× |
| `mg_cc_wsync = 0 / ≤−1` + `mg_cc_hydro(/ _k)` | 2026-10-25 | partial refresh 只写 rho/v/bcc，`w[ene]`(=p) 与组分留在 RK 中点 |

### C. 基准**未采用**、备选分支**未被基准验证**（退役候选，分两档）

**(c1) 整块可去（实验性子功能，基准默认关）**
`proj_mg > 0`（投影的 MG 版；基准走 SOR）、`mg_cc_exact` / `exact_line` / `exact_omega`（FEEC 精确行）、
`mg_cc_eta*`（均匀/z-profile η 家族；基准走 `eta_var` provider）、`mg_cc_divclean`、`mg_cc_fproj`、
`mg_cc_rharm`、`mg_cc_rem_eps`、`mg_cc_romega`、`mg_cc_rstag`、`mg_cc_rmin`、`mg_cc_bhat`、`mg_cc_cgrad`、
`mg_cc_aniso_base`、`sts_nu`、`bc_edge`、`hall_chyp`、`hall_upw`、`eta_out_kill`、`prof_*`

**(c2) 默认值就是基准路径 ⇒ 只能删"备选分支"、把默认写死**
`mg_cc_ct`(1)、`mg_cc_fmatch`(2)、`mg_cc_time`(0 BE；CN/BDF2 是备选，但 `th_op`/`b_prev`/`src0` 织入算子核心)、
`mg_cc_bclose`(0 —— **危险但基准在用，须同时保留守卫**)、`mg_cc_fpoint`

> ⚠️ **c1 里 `mg_cc_divclean` / `fproj` / `rharm` / `mg_cc_eta` / `exact*` 在历史实验 par 里各被用过
> 1–34 次**（如 `pdc.par`、`pH1.par`、`t25dH1.par`、`sph21_25d.par`、`pex2.par`）。它们**不在最终基准里**，
> 但"从未使用"是错的（我先前只扫了 3 个目录）——是否退役由你定，证据属过时测试材料。

### D. **不是**候选（别处在用）

| 键/模式 | 为什么留 |
|---|---|
| `mg_diffusion` / `mg_eta` / `mg_lvl` | `usr_ext/std_tst/mhd_ni_tst` 与 `mhd_ni_sharp` 实例化 `mg_diffusion_t` —— 求解器的**回归试验台** |
| `prof_eta` + `prof_alpha_*` / `prof_rho_ref` | η provider 的参数（provider 是基准路径） |
| `sts` / `sts_nu` / `hall_*` / `bc_edge` | 显式 / STS / Hall 腿 |
| `mg_cc_bclose` | 基准启用的危险组合，需**守卫**而非删除 |

## 推荐集（userguide 只应暴露这些）

`mg_cc` · `mg_cc_time` · `mg_cc_bc` · `mg_cc_bclose` · `mg_cc_ct` · `mg_cc_fmatch` · `mg_cc_eta_var` · `mg_cc_line` · **`mg_cc_ene`** · `mg_cc_ene_heat` · `mg_cc_proj` · `mg_cc_proj_iter` · `mg_cc_dtchk` · `mg_cc_projchk`

## 全表（66 键）

| 键 | 默认 | 被 par 设置次数 | uses | 成员 | 状态 |
|---|---|---|---|---|---|
| `bc_edge` | `0` | 0 | 10 | `bc_edge` | — |
| `bf_dump_dir` | `std::string(` | 3 | 6 | `bf_dump_dir` | — |
| `bf_dump_file` | `std::string(` | 3 | 6 | `bf_dump_file` | — |
| `bf_dump_n` | `bf_dump_n` | 3 | 6 | `bf_dump_n` | — |
| `cfl` | `cfl_dt` | 39 | 4 | `cfl_dt` | — |
| `eta_a` | `0` | 31 | 35 | `eta_a` | — |
| `eta_dump_file` | `std::string(` | 4 | 6 | `eta_dump_file` | — |
| `eta_h` | `0` | 26 | 10 | `eta_h` | — |
| `eta_o` | `0` | 31 | 35 | `eta_o` | — |
| `eta_out_kill` | `0` | 5 | 7 | `out_kill` | — |
| `eta_replay_file` | `std::string(` | 12 | 10 | `eta_replay_file` | — |
| `gamma` | `gam_dt` | 68 | 5 | `gam_dt` | — |
| `hall_chyp` | `0.05` | 0 | 5 | `hall_chyp` | — |
| `hall_l` | `0` | 12 | 5 | `hall_l` | — |
| `hall_upw` | `0` | 0 | 5 | `hall_upw` | — |
| `mg_cc` | `enabled ? 1 : 0` | 71 | 14 | `enabled` | **生产腿**（隐式 curl-curl） |
| `mg_cc_aniso_base` | `aniso_base` | 2 | 6 | `aniso_base` | — |
| `mg_cc_bc` | `bc_mode` | 98 | 106 | `bc_mode` | −1 auto / 0 / 1（生产）/ 2；0/2 有实测差异（P1/P2 开放问题） |
| `mg_cc_bclose` | `bc_close` | 56 | 6 | `bc_close` | **勿与 `bc=1` 配 0**（NaN/SIGFPE，D11） |
| `mg_cc_bhat` | `bhat` | 0 | 6 | `bhat` | — |
| `mg_cc_cgrad` | `1` | 0 | 4 | `cgrad` | — |
| `mg_cc_ct` | `ct_adj` | 3 | 7 | `ct_adj` | 默认 1 在关键路径 → backlog (a) |
| `mg_cc_divclean` | `0` | 1 | 8 | `dvg_clean` | 从未执行 → backlog (b) |
| `mg_cc_dtchk` | `dtchk` | 107 | 4 | `dtchk` | — |
| `mg_cc_ene` | `2` | 33 | 8 | `ene_dep` | 0 关 / **2 默认（已验证）** / 3 SBP 参考实现；1 已退役 |
| `mg_cc_ene_heat` | `1` | 16 | 7 | `ene_heat` | 仅对 `mg_cc_ene = 2` 有意义 |
| `mg_cc_eta` | `eta` | 30 | 14 | `eta` | 均匀/z-profile η → backlog (b) |
| `mg_cc_eta_a` | `eta_a` | 0 | 35 | `eta_a` | — |
| `mg_cc_eta_a_z` | `eta_a_z` | 0 | 19 | `eta_a_z` | — |
| `mg_cc_eta_o` | `eta` | 0 | 35 | `eta_o` | — |
| `mg_cc_eta_o_z` | `eta_o_z` | 0 | 20 | `eta_o_z` | — |
| `mg_cc_eta_var` | `eta_var` | 144 | 13 | `eta_var` | 逐格 η provider（生产） |
| `mg_cc_exact` | `exact` | 12 | 6 | `exact` | FEEC 精确行，从未被执行 → backlog (b) |
| `mg_cc_exact_line` | `0` | 4 | 4 | `exact_alt` | — |
| `mg_cc_exact_omega` | `type::float_t( 1` | 2 | 4 | `exact_omega` | — |
| `mg_cc_fmatch` | `fmatch` | 4 | 9 | `fmatch` | 只允许 2 → backlog (a) |
| `mg_cc_fpoint` | `time_mode == 1 ? 0 : 1` | 6 | 5 | `force_point_` | — |
| `mg_cc_fproj` | `fproj` | 1 | 13 | `fproj` | — |
| `mg_cc_line` | `0` | 17 | 6 | `line_axis` | — |
| `mg_cc_lvl` | `mg_lvl` | 36 | 13 | `mg_lvl` | — |
| `mg_cc_npass` | `npass` | 37 | 6 | `npass` | — |
| `mg_cc_proj` | `proj` | 137 | 8 | `proj` | 生产开；两套解法见 `proj_mg*` |
| `mg_cc_proj_iter` | `proj_iter` | 136 | 5 | `proj_iter` | — |
| `mg_cc_proj_om` | `proj_om` | 4 | 4 | `proj_om` | — |
| `mg_cc_projchk` | `projchk` | 141 | 7 | `projchk` | — |
| `mg_cc_rem` | `n_rem` | 1 | 9 | `n_rem` | — |
| `mg_cc_rharm` | `0` | 1 | 8 | `r_harm` | — |
| `mg_cc_rmin` | `8` | 0 | 5 | `r_min` | — |
| `mg_cc_time` | `time_mode` | 0 | 13 | `time_mode` | **0 BE 是默认（生产）**；1 CN / 2 BDF2 从未被任何 par 执行 → backlog (a) |
| `mg_cc_wbnd` | `wbnd` | 3 | 4 | `wbnd` | — |
| `mg_diffusion` | `enabled ? 1 : 0` | 0 | 14 | `enabled` | **测试腿**：`usr_ext/std_tst/mhd_ni_{tst,sharp}` 实例化 `mg_diffusion_t` → **保留** |
| `mg_eta` | `eta` | 0 | 14 | `eta` | — |
| `mg_lvl` | `mg_lvl` | 0 | 13 | `mg_lvl` | — |
| `mode` | `0` | 223 | 35 | `mode` | — |
| `p_floor_eps` | `p_floor_eps` | 2 | 5 | `p_floor_eps` | — |
| `prof_alpha_a` | `0` | 0 | 4 | `alpha_a` | — |
| `prof_alpha_o` | `0` | 0 | 4 | `alpha_o` | — |
| `prof_eta_a` | `0` | 0 | 35 | `eta_a` | — |
| `prof_eta_o` | `0` | 0 | 35 | `eta_o` | — |
| `prof_rho_ref` | `1` | 0 | 3 | `rho_ref` | — |
| `proj_mg` | `proj_mg` | 23 | 6 | `proj_mg` | 投影的 MG 版（生产走 SOR）→ backlog (b) |
| `proj_mg_dtol` | `proj_mg_dtol` | 15 | 5 | `proj_mg_dtol` | — |
| `proj_mg_sc` | `proj_mg_sc` | 6 | 5 | `proj_mg_sc` | — |
| `proj_mg_v` | `proj_mg_v` | 15 | 8 | `proj_mg_v` | — |
| `sts` | `0` | 59 | 13 | `sts` | — |
| `sts_nu` | `0.01` | 0 | 4 | `sts_nu` | — |

## 为何"多迭代即爆炸"长期未被发现，以及收窄 η 的真实作用（2026-10-26 实测）

### 逐 pass 缺陷（`[multigrid] print_iter = 1`，32 步、971 个相邻比值）

| 步骤 | 缺陷序列 | 逐次比值 |
|---|---|---|
| #0（116 次）| `9.85e-5 -> 8.55e-5 -> 7.24e-5 -> 5.37e-5 -> 2.56e-5 -> 2.10e-5 -> 2.40e-5 -> 2.70e-5 ...` | `0.87 0.85 0.74 0.48 0.82` **`1.15 1.12`** |
| #1 | `1.61e-3 -> 1.12e-1 -> 3.47e-1 -> ...` | **`69.4`** `3.11 0.19 ...` |
| #2-#7 | `3.8e-1 -> 3.8e-1 -> 3.8e-1 ...`（**恒定平台**）| `1.00 1.00 1.00 ...` |

中位比值 0.899（部分收缩），但**最大 1.4e21** ⇒ **V-cycle 的迭代矩阵谱半径 > 1，且部分模态完全不被压缩**。

### 为何先前没发现（四个"遮蔽"同时存在）

1. **设计上只跑一次**：`it_max` 在非 deep 周期恒为 1（`cycle % cycle_per_deep == 0 ? n_iter : 1`）
   ⇒ **没有第二次迭代可暴露 ρ>1**。故障只在"被人为强制多迭代"时才显形（本次 `iter_adapt`）。
2. **探针默认关闭**：`[multigrid] print_iter / print_info` 在生产栈里都是 0；唯一能看到逐 pass 缺陷的仪器关着。
3. **症状被吸收**：`guard_fac = 100` 的回退、非有限值守卫、以及**事后散度投影**都把它压成"看起来只是散度不够干净"，
   而不是"求解器不收敛"；生产验收判据（`proj <= 2e-9` 之类）因此照样能过。
4. **机制被"记录"而非"定位"**：`mg_ni_base.h` 的 `stag` 注释 / D29 早已写下 "that operator AMPLIFIES
   (|B| ~ x10 per pass)" —— 它读起来像"已知行为"，于是只被**规避**（把 `stag` 设 0），从未被**测量/定位**。
   同时可见量（注入的 div-B）被归因于"算子"（P1 / §3quaterdecies 的"非理想步注入散度"），而非"近似逆不是逆"。

### 收窄 η 动态范围：**有帮助，但只是遮住症状，且有窗口**

- **有帮助的机理**：收窄 η 直接降低 (a) 相邻格 **η 对比**、(b) **抛物数 pn = η dt/dx^2**，从而收窄 A 的谱
  ⇒ 弱近似逆需要控制的谱更窄，**放大（ρ>1）不易被激发**。实测：`pn ~ 2.5e5` 仍健康，`~1.4e7`（cap 1e24）NaN。
- **但它治不了根**：平台（比值恒 1.00）在**健康默认配置**里也存在 ⇒ V-cycle 即便在温和算子下也**不收缩**，
  只是"没有被逼到暴露"。层次不一致（粗算子非相容限制）不会因收窄 η 而消失。
- **收得太紧反而有害**：`eta_cap_ref = 1`（η <= 0.3、pn <= 74）时 **div-B 退化 4-5 个数量级**
  （`in` 5.52e3 / `solved` 36.9 vs 对照 5.7e-1 / 1.9e-4）—— 因为**扩散本身被削掉了**，不是求解器的功劳。
- ⇒ 存在一个**窗口**：`pn` 从 ~1（扩散仍在做功）到 ~2.5e5（单次近似逆仍可控）。定稿默认
  （cgs 1e22 / `cap_c` 1 / `cap_ref` 1e4 ⇒ eta_max 25 code、pn <= 1.2e4）正落在窗口内：
  **既不崩、散度质量又与对照同级**。
- **真正的修法**（不靠收窄）：把 V-cycle 当预条件子做 **PCG**（A 是 SPD ⇒ 残差单调下降，
  放大模态被 Krylov 子空间对消）；或让粗层算子成为**相容的 Galerkin 限制**。见 dead-ends M26。

## η 上限体系定稿（2026-10-26）—— 三条限制，全部施加在**化学求得的 η** 处

在 `usr/intermediate/disk_eta.h` 的 `eval()` 里、算出 `eo_cgs`/`ea_cgs` 之后、**ring/funnel/wedge 附加项之前**：

| 键 | 默认 | 形式 | 单位处理 |
|---|---|---|---|
| `[nonideal] eta_cap_cgs` | **`1e22`** | 标量物理天花板 | `eta_cap = eta_cap_cgs * eta2code`（`eta2code = t0/l0^2`）|
| `[nonideal] eta_cap_c` | **`1`**（开）| **因果**：`eta <= eta_c * dx`，`eta_c = c t0 / l0` | **需要换算**（1 au/yr ⇒ `eta_c = 6.32e4`）|
| `[nonideal] eta_cap_ref` | **`1e4`** | **抛物**：`eta <= eta_cap_ref * dx^2 / dt_0`（`dt_0` = **整步**）| **不换算**（两边同为 code 单位）|

- `dx` = **三方向弧长之最小者**：`dr`、`r dtheta`、`r sin(theta) dphi`。
  `DBG_ETA_DX` 探针实测（格 (0,2) / (32,256)）：`dx_cfl(theta)` **就是** `r dtheta`（比值 0.835 / 1.0000）、
  `dx_cfl(phi)` **就是** `r sin(theta) dphi`（0.987）、`dx_cfl(r)` 是精确间距 ⇒ 逐方向取 `dx_cfl` 最小
  即该规则（近轴处略更保守）。
- `eta_cap_ref = 0.25` 是**显式稳定**选择；`1e4` ⇒ `pn <= 1e4`；`<= 0` 关闭。`eta_cap_c = 0` 关闭因果限制。

### A/B（同一归并 base `mgt/ppd_base.par`，2-par 约定）

| 配置 | 周期 | Duration | `eta_max`(code) | `pn_max` | `MGCCDIV in` | `solved` | 崩溃 |
|---|---|---|---|---|---|---|---|
| 两限制关闭（cgs `1e23`）| 37 | 279.3 s | 1.25e3 | 6.6e4 | 5.72e-1 | 1.93e-4 | 无 |
| `cap_ref = 1` | 35 | 359.4 s | 0.30 | 74 | 5.52e3 | 36.9 | 无（质量崩）|
| `cap_ref = 1e10`、cgs `1e23` | 35 | 279.9 s | 42.4 | 2.45e5 | 5.69e-1 | 9.87e-3 | 无 |
| **定稿默认**（cgs `1e22` / `cap_c 1` / `cap_ref 1e4`）| **37** | **279.9 s** | 25.4 | 1.2e4 | 5.73e-1 | 3.07e-4 | **无** ✓ |

**量到的可用区间**：`pn` 到 **2.5e5** 仍健康；`1e24`（pn 1.4e7）在中面化学 η 尖峰处 NaN。

**要点**：`cap_ref = 1`（pn <= 1）会把死区 eta 削掉约 4 个数量级、并让 div-B 退化 4-5 个数量级 ——
"隐式腿换来大 eta 空间"这件事会被过紧的限制抵消。定稿默认下**散度质量与对照同级、墙钟无代价**。

## 两个 η 上限的作用域不同（易被误解，2026-10-26）

`usr/intermediate/disk_eta.h` 有**两个**上限，作用对象与时机不同：

| 键 | 默认 | 作用对象 / 时机 | 公式 |
|---|---|---|---|
| `[nonideal] eta_cap_cgs` | **`1e19`** | **物理** η（profile 剖面 + chemistry 算出的 eo/ea），施加在 **ring / funnel / wedge 附加项之前** | `:547-548`：`eo = min( eo_cgs*eta2code, eta_cap )`（ea 同）|
| `[nonideal] eta_cap_dt` | **`0`（关闭）** | **最终**的 `eo`/`ea`（wedge blend 与附加项**都包含**）| `η <= 0.25 dx_min^2 / eta_cap_dt` |

代码注释（`:586-588, :620-624`）明写：wedge blend "**Applied AFTER the physical eta ( and its cap )** …
the ring / funnel ADD parts ride inside eo"，随后 "**The TOTAL parabolic cap** ( `[nonideal] eta_cap_dt` ):
bound the **FINAL** eo / ea -- wedge blend included"。⇒ **`eta_cap_cgs` 管不到数值抑制项**；
要约束**总** η，得打开 `eta_cap_dt`（本会话在极轴问题上实测：`1e-3` 给出 `pn <= 0.22`）。

⚠️ **数值差异**：注释把两件事写在一起 —— "Physical cap ( WBG19: ~ 10 c_s h at r_in ) and the hard
ceiling ( WBG19's eta_max = **1e20** )" —— 但代码只有一个标量上限 `eta_cap_cgs`，**默认 `1e19`**，
即比 WBG19 陈述的硬上限**紧 10 倍**。PPD 生产栈显式设 `eta_cap_cgs = 1e19`，与本侧默认一致。
PPD 侧 2026-10-15 的分析结论是"1e19 是 WBG19 物理上限、**不建议动**；数值问题应由
`[funnel] eta_max/width` 的主动抑制解决"（`/mnt/scratch/ppd_tst/handover/run11-dt-drop-analysis-2026-10-15.md`）。

### 实测：`eta_cap_cgs = 1e24` **不安全**（2026-10-26）

同栈 A/B（polar private dir `mgt/`，`rho_floor_dyn = 5e7`，`proj_mg = 1`，仅改该键）：

| 键 | `MGCCETA eta`(code) | `pn` 最大 | `MGCCDIV in`（扩散后/投影前）| 终止 |
|---|---|---|---|---|
| `1e19`（默认）| 1.7 | 9.8 | **5.9e-1** → solved 2.1e-9 | 29 周期到 `t = 2e-2` ✓ |
| **`1e24`** | **1.43e5** | **6.2** | **3.4e32** → solved 5.4e16 | **cycle 17 `t = 1.18e-2` NaN 熔断**（`GUARD 2.62e5 bad cells`）|

**关键：失效模式不是 dt 塌陷** —— dt 全程稳定（`6.9e-4`，`fuse = 0`），是**场本身**先炸：
`∇·B` 的 `in` 值比健康臂大 **14 个数量级**（3.4e32 vs 5.9e-1），随后 EOS 出 NaN 被 `GUARD` 拦下。

**机理（推论）**：η 到 ~1e5（code）后 `pn = η dt/dx² ≈ 6 > 1`，隐式算子 `I + dt·R·CᵀC` 变成扩散主导，
且相邻格 η 对比达 **5 个数量级以上** ⇒ V-cycle 的近似逆对强对比行失控，CT/精确行把散度注入到 1e32
（`mg_cc_proj` 虽把 6e15 倍降下来，仍剩 5e16）。⇒ **这个 cap 的真实职责是算子条件数 / η 对比度控制，
不是抛物 dt**；dt 限制器根本来不及拦。

### 失效机理（2026-10-26，逐步取证）

逐步解析 `tB.log`（18 步）：**不是** dt 塌陷（dt 稳定 6.9e-4、`fuse = 0`），也**不是** div-B 渐进增长
（`in` 在 12 个周期里稳定在 ~2.5e-3）。真因是 **η 的间歇性尖峰**：

| 步 | `eta_max`(code) | `pn_max = η dt/dx^2` |
|---|---|---|
| 典型步 | 10 – 80 | 500 – 3000 |
| **#11 / #16** | **2.82e5** | **1.4e7 / 1.8e7** |
| **#18** | 1.43e5 | 6.2e6 |
| 对照（`1e19` 臂，全程）| **<= 1.7** | **<= 9.8** |

`2.82e5` 反复出现 = 新 cap（`1e24 * eta2code`）**饱和**处 ⇒ 尖峰的未截断值更高。

**尖峰来源**：η ∝ 1/x_e 由**化学电离度**给出，而电离度在稠密内盘的刚性瞬变中会崩溃
（`m0p.par` 记录过 `x_e/rho` 一周期 4e-9 -> 0.99）。**`1e19` 的 cap 一直在把这些尖峰削平**
（η_max 恒 <= 1.7）；抬到 `1e24` 后尖峰以 2.8e5 通过 ⇒ 单步内 `pn ~ 1e7`，隐式算子在该格的谱半径
~1e7，而求解器**每次调用只做 1 个 V-cycle**（实测）⇒ 该格更新是垃圾，几次尖峰后状态退化到 NaN
（`GUARD` cycle 17，首个坏格 (31,231,0) 在**盘中面**，不在极轴）。

⇒ **`eta_cap_cgs` 的真实职责 = 压缩化学驱动的 η 尖峰（数值稳定性），而不是"物理上限"。**

### cap 的**作用区域**（2026-10-26，取 `MGCCETA` 的 `max pn at (r,θ,φ)`）

两臂的位置分布（`pn = η dt/dx²` 的 argmax）：

| 臂 | 样本 | θ 范围 | r 分布 |
|---|---|---|---|
| `1e19`（健康，29 步）| 29 | **221 – 282** | r=0 占 6 次，其余散布 4…41 |
| `1e24`（失效，18 步）| 18 | **227 – 279** | 0,17,23,25,30,31,33,34,42,48 |

θ 的中面是 256；**全部样本都在中面 ±35 格内**，φ 任意（轴对称）⇒

⇒ **cap 的作用区 = 内盘的中面带**（θ ≈ π/2 ± 0.2 rad、r 从内边界到 ~40/64），即**致密、冷、
电离度最低的死区**（η ∝ 1/x_e 最大处）。**不是极区、不是外边界。**

**移除后错误出现在同一处**：`GUARD ... first at (31,231,0)`、崩溃前最后一个散度 argmax
`MGCCDIVARG-post at (33,279,5)` —— 都是 r ≈ 31–33 的**中面**格。

**顺带交叉验证**：`max pn` 从不出现在极轴 —— 与本会话早先 `[wedge] polar_gate` 的修法一致
（极轴 η 被关掉后，pn 的 argmax 就从极区移到中面；健康臂全程 `pn <= 9.8`）。

### 边界夹准 + 真 MG/SOR 对比（2026-10-26，归并 base `mgt/ppd_base.par` + 单臂）

| 臂 | 投影 | cap | 周期 | Duration | s/周期 | 结果 |
|---|---|---|---|---|---|---|
| `armE` | **MG** | `1e22` | **37** | **280.5 s** | 7.58 | 到 t = 2.0346e-2，`fuse/guard = 0` |
| `armS` | **SOR** | `1e22` | **32** | **232.2 s** | 7.26 | 到 t >= 2e-2，`fuse/guard = 0` |
| `armF` | **MG** | **`1e23`** | **35** | **282.3 s** | 8.07 | **到 t = 2.0178e-2，`fuse/guard = 0`** ✓ |
| （旧 8-par SOR 臂）| SOR | `1e22` | 31 | 224.2 s | 7.23 | 与 `armS` 在 M8 散布内一致 ⇒ 归并忠实 |

**真 MG vs SOR（更正）**：MG 比 SOR **慢 ~21%**（280.5 vs 232.2 s）且多 5 个周期 ——
因为 MG 路径的 dt 更小（5.87e-4 vs ~6.79e-4，投影对场的影响反作用于 CFL）。**早先"MG ≈ SOR"是无效比较**
（两臂皆为 SOR，dead-ends M25）；MG 的正当理由是**适应性**，不是速度。

**失效边界**：`pn_max` 在 `1e22` 为 <=1.5e5、`1e23` 约 3e5（尖峰期），**两者都健康**；`1e24` 的 1.4e7 才 NaN
⇒ **阈值约在 `pn ~ 1e6`**。故 `eta_cap_cgs` 可安全抬到 **`1e23`**（比默认 `1e19` 宽 **4 个数量级**）——
隐式腿确实给出了极大的 η 动态范围。

注意 `armF` 末段的 `pn_max` 回落到 ~4e2-1e3 ⇒ **尖峰是暂态的**，典型刚度只有 1e2-1e3，
只有少数步极端 —— 这为"**自适应迭代预算**"（只在极端步多迭代）提供了直接依据。

### 标定：`eta_cap_cgs = 1e22` **安全**（2026-10-26）

同栈（private `mgt/`，`rho_floor_dyn = 5e7`，`proj_mg = 1`）：

| cap | `eta_max`(code) | `pn_max = η dt/dx^2` | 结果 |
|---|---|---|---|
| `1e19` | <= 1.7 | <= 9.8 | 健康，29 周期 |
| **`1e22`** | **<= 2.82e3**（cap 饱和）| **<= 1.6e5** | **健康：31 周期到 t = 2.048e-2、`fuse = 0`、`GUARD = 0`、Duration 224.2 s** |
| `1e24` | 2.82e5 | 1.4e7 | cycle 17 NaN |

⇒ **隐式腿已经给了极大的 η 动态范围**：`pn` 到 **1.6e5**（远超显式极限 ~0.25，约 6 个数量级）仍稳定，
且墙钟没有增加（7.23 vs 7.3-7.5 s/周期）——因为求解器**并未因刚度多迭代**。

**失效阈值夹在 `pn` 1.6e5（活）与 1.4e7（死）之间**；根因是**近似逆**（每次调用 1 个 V-cycle）
在高 `pn` 下不再收缩，而不是精确算子失稳（`I + dt R C^T C` 的特征值恒 >= 1，永不奇异）。

### 解决方向（推荐）

**把两个职责分开**：让 `eta_cap_cgs` 做"物理上限"（可以放宽甚至到 1e24），而**数值尖峰交给
现成的抛物上限** `[nonideal] eta_cap_dt`（默认 0 = 关闭）：`η <= 0.25 dx_min^2 / eta_cap_dt`
⇒ `pn <= 0.25 dt / eta_cap_dt`（`dt ~ 7e-4`、`eta_cap_dt = 1e-3` 时 `pn <= ~0.2`，与早先实测一致），
且它作用在**最终** η 上（含 ring/funnel/wedge 附加项）。

待验 A/B：**`eta_cap_cgs = 1e24` + `eta_cap_dt = 1e-3`** 是否既保住物理上限的宽松、又不产生尖峰
（判据：`MGCCETA` 的 `eta_max`/`pn_max` 有界、无 `GUARD` NaN、`MGCCDIV` 的 `in` 保持 ~1e-1）。

**建议**：保持 `1e19`（PPD 侧现状一致）。若要放松，必须在**安全区间做扫描**（1e20 / 1e21），
判据用 `MGCCDIV` 的 `in`（保持 ~1e-1 量级）与**无 `GUARD` NaN**，而不是 dt。

**物理判断**：`η = 1e19 cm^2/s` 在内盘已进入强解耦区 —— 取 `c_s H` 为特征尺度（r ~ 1 au、
`H/r ~ 0.05`、`c_s ~ 1e5-1e6 cm/s` ⇒ `c_s H ~ 1e18 cm^2/s`），则 `Rm = c_s H / η`
在 `1e19` 处约 **0.1**、在 `1e20` 处约 **0.01**：**两者都是 `Rm << 1`**，抬高 10 倍只改变解耦的
**程度/速率**，不改变"是否解耦"。故只要触顶区的未截断 η 确实远大于 1e19（物理上已完全解耦），
该 cap **不破坏物理基础**，而是数值上必要的保护。真正的风险是 cap 的**削平效应**（抹掉 η 的空间
梯度、形成人为平台边界），若该边界扫过细网格带会同时影响 dt 与隐式算子的刚性。

## 默认模式的实例化（库侧，2026-10-25）

`src/modules/dynamics/mhd/nonideal/` 现在**有 `.cpp`**（对齐 `dynamics/mhd/mhd.cpp`、`dynamics/hydro/hydro.cpp`
的约定——那两个文件装的是**非模板**基类方法的出线定义）：

| 文件 | 作用 |
|---|---|
| `nonideal.cpp` | **默认模式显式实例化**：`ni_t<>`（= `rk2_t<>` + `const_eta_t`）、`ni_eos_t<>`、`mg_cc_diffusion_t< prx_default_t, const_eta_t >`；并定义 `prx_default_t` 的静态成员。**实例化 TU 必须先包含 `../mhd.h`**（`dynamics::base_t` 必须完整），否则报一墙 `incomplete type` 并**级联**出假的 EOS 错误（`"no member read"`） |
| `proxy.h` | **库级默认代理** `prx_default_t`：`bdt_cc_t`（MHD block data + MG 数组的 no-image 视图）、`d_nimg` / `set_module` / `act_mhd_flx`。采纳它的 pgen 只需赋值 `prx_default_t::p_mhd`，不必再抄那 ~70 行 |

**为什么 pgen 现在还拿不到这份实例化**：η 仍是**模板参数**，且 `mg_cc_freeze_k< bdt_T, eta_T >` 是
`__global__`、按值接收并在**设备端**调用它（`implicit_cc.h:240-262, 2476`）。ppd 用 `interm::disk_eta_ni_t`、
std_tst 用 `prof_eta_t` / `hall_len_eta_t` ⇒ 各是**独立实例化**，无法 `extern template`。
**落地加速的前提**：把 η **类型擦除**为一个固定类型（设备函数指针 + 参数指针），使
`< prx_default_t, ni_eta_erased_t >` 成为**所有** pgen 的唯一实例化——那样改 `implicit_mg.h`/`implicit_cc.h`
只重编 `nonideal.o`（**4.09 MB**）而不必重编 `usr.o`（**生产构建 9.26 MB**；带 `-DDBG_F2_CHK_OP`
探针的构建是 16.75 MB —— **不要用探针构建量编译代价**）。详见 `STATUS.md` §3septendecies。

## `[nonideal] ee_*` —— edge-MG 隐式步的新旋钮（2026-10-30，`STATUS.md`（六十一））

这组键只属于 **edge-MG 的隐式步**（`src/modules/dynamics/mhd/nonideal/edge_mg/`），与上表的
`mg_cc_*`（格心 MG）互不相干；两者不要混用（生产 par 仍是 `mg_cc = 0` + `mg_cc_eta_var = 0`）。

| 键 | 默认 | 含义 | 备注 |
|---|---|---|---|
| `ee_enable` | 0 | 整条 leg 的开关 | 0 = 完全不介入（逐位理想） |
| `ee_solve` | **0** | 0 = 只做 S0' 的 `div|max` 诊断（**逐位不变**）；1 = 组装 + 求解 + 写回 | 打开后状态会真的扩散 |
| ~~`ee_method`~~ | — | **已删除（2026-11-06）**：只剩 method 5（设备端 matrix-free MG），`par_t` 里不再有 `method` 字段，`ee_method` 写在 par 里会被忽略（未知键） | 五个 host 后端（0 `pcg` / 1 `mg` / 2 `mgpcg` / 3 `pi` / 4 `mgpcg_dev`）连同 HX 辅助空间与 Π 一起退役：见 `STATUS.md`（八十九）与 `dead-ends.md` M78/M82 |
| `ee_subcell` | 0 | 0 = 一格一张量；1 = 每个求积点取一阶 Taylor（∇η 由邻居 stencil **现算**，存储仍 6 数/单元） | **1 需要单块**：梯度还缺一层 `r_arr` ghost 交换 |
| `ee_iter` | 100 | 迭代（`mg` 时是**循环**）上限 | 两种计数**不可混比**，门里都打印 |
| `ee_tol` | `1e-12` (P2) / `1e-5` (P1/P0) | 相对残差目标 `‖b−Ax‖/‖b‖` | 与 A5 同口径；按精度自动分档 |
| `ee_nlevel` | 8 | 层级数上限 | 粗化中途停 |
| `ee_n_smooth` | 2 | 每层 pre/post 的光滑次数；`block_cheb` 时是多项式次数 | 无着色块 Chebyshev 推荐 4，即 V(4,4) |
| `ee_coarse_iter` | **2** | **最粗层**嵌套 PCG 的迭代上限 | 粗层解只决定预条件子的强弱，不决定答案。三网格上上限 1 / 2 / 10 的收缩因子与外层 it/res **逐位相同**，而每步：8×16×32 上 35 / 36 / 41 ms、8×16×128 上 57 / 59 / 81 ms、极轴 36 / 43 / 59 ms（`STATUS.md`（九十）（九十二））。2 是出厂值：1 同样好，第 2 次是留给粗层不那么温和的网格的余量 |
| `ee_coarse_tol` | `1e-10` | 最粗层嵌套 PCG 的相对残差目标 | 与 `ee_coarse_iter` 配对使用；两者都是**求解器**设置，不是方法设置，故放 `[nonideal]` 且 pgen/par 都可改 |
| `ee_dt` | 0 | 0 = cycle dt；>0 = 强制 cycle dt 与扩散 dt 都为该值，并跳过抛物 cap | 只用于 STS/隐式 A/B；会同时覆盖 `ni_eos_t` 的 CFL/cap，调用方自担理想步稳定性 |
| `ee_writeback` | 0 | 0 = 强形式 `-dt d1(ηJ)`；1 = `b + P_out(a_new) − P_out(a_old)`，`P_out=M2⁻¹M12ᵀ` | 1 = 当前解析对照的修正腿；缓存旧表示，每步一次 M2 解。纯扩散已验收，理想步耦合另设门；0 保留负对照 |
| `bc_edge`（edge-MG） | 0 | 1 = 强写回墙切向 E 置零；2 = PCG/MG 各层消元墙内切向 B 棱 | `j1` 零点模必须用 2（`n×B=0`），不能误当零自然电流；M2 不加同样的面约束 |
| `ee_a_bridge` | 0 | 1 = 首次步 / restart 后把 `bf` 乘 CT 面积成面通量，解 `M1 a = M12 bf` 初始化 `a_edge` | A7.1 representation-in；旧 `M12ᵀM12` 法方程已移除。通用 `e_z` 投影仍有 `1e-3` 漂移，解析 T0 可用 `cmp_a_from_exact` 分离该误差 |
| `ee_coarsen` | 3 | bit 1 = r、2 = θ、4 = φ | **φ 不粗化是规范要求**（spec §5.6）；全粗化必须显式打开 |

另：`ee_solve = 1` 目前**只接受球坐标块**（`geo_sph_from` + μ 参数化）；笛卡尔块会**报错**而不是猜。
写回、面更新与右端规则（被丢 DOF 必须连右端一起丢，`dead-ends` **M56**）都在驱动里，门见
`mhd_ee_e2e{,_polar}.par`（`test_mode 9`）与 `mhd_ee_a7.par`（A7.5）。

### 旋钮 → 后端退役图（P6，2026-11-06；执行记录）

`ee_method` 曾有 6 个值，26 个 `ee_*`/`bc_edge` 旋钮并不均匀分布其上。P6 先走保守路线（只删死旋钮
`ee_bdf`，26 → 25）；**2026-11-06 的第二个裁决把另外五个后端也删了**，于是那张"退役条件"表里的
一整片同时到期：

| 旋钮 | 唯一服务对象 | 现在 |
|---|---|---|
| `ee_enable` / `ee_solve` / `ee_tol` / `ee_iter` / `bc_edge` / `ee_writeback` | 全部 | 保留（核心 6 个） |
| `ee_subcell` | 块的几何选项 | 保留（是物理选项） |
| `ee_coarsen` / `ee_nlevel` / `ee_n_smooth` | MG | 保留（φ 不粗化仍是规范要求，spec §5.6） |
| `ee_coarse_iter` / `ee_coarse_tol` | MG 的最粗层求解 | 新增（2026-11-06）：默认就是实测最优的那一档 |
| `ee_dt` | STS/隐式 A/B 对照 | 保留（那条对照线还在） |
| `ee_a_bridge` | A7.1 representation-in | 保留（method 5 现在支持它，setup 步跑一次） |
| `ee_verb` | 诊断输出 | 保留（17 个 par 在用） |
| `ee_method` / `ee_smoother` / `ee_refine` / 9 个 `ee_pi_*` / `ee_bdf` | 已退役的五个后端 | **已删除**（字段、初值、读取、校验、回退逻辑全删） |

净结果：**25 → 15 个旋钮**（12 个 + `ee_verb` + 粗层那对），比原计划的 10 多 5 个 —— 而这次没有为此删任何
后端能力，因为后端本身已经被删到只剩一个。剩下那 5 个差额（`ee_verb`/`ee_coarsen`/`ee_n_smooth`
以及粗层那对）
要砍，需要先把诊断输出与层次构建改成不可配置的，属于 P6 的"第二阶段"，不在删除范围内。
