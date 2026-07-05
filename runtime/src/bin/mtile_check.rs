//! Correctness of the batched-forward building blocks vs their M=1 references.
fn main(){
  let mut bad=0;
  println!("== small-M decode GEMM (gemm_mtile) ==");
  for m in [1usize,2,3,4]{
    let e=trapetum::check_mtile(4096,4096,m);
    let ok=e<1e-3; println!("  gemm_mtile M={m}  rel_err={e:.2e}  {}", if ok{"OK"}else{"FAIL"});
    if !ok{bad+=1;}
  }
  println!("== batched RMSNorm (rmsnorm_m) ==");
  for m in [1usize,2,4]{
    let e=trapetum::check_rmsnorm_m(4096,m);
    let ok=e<5e-3; println!("  rmsnorm_m M={m}  rel_err={e:.2e}  {}", if ok{"OK"}else{"FAIL"});
    if !ok{bad+=1;}
  }
  println!("== batched causal attention (attn_m) ==");
  for (nh,nkv,hd) in [(32usize,32usize,128usize),(32,8,64)] {
    for m in [1usize,2] {
      let e=trapetum::check_attn_m(nh,nkv,hd,5,m);
      let ok=e<2e-2; println!("  attn_m nh={nh} nkv={nkv} hd={hd} M={m}  rel_err={e:.2e}  {}", if ok{"OK"}else{"FAIL"});
      if !ok{bad+=1;}
    }
  }
  println!("== batched RoPE (rope_m) ==");
  for (nh,hd) in [(32usize,128usize),(32,64)] { for m in [1usize,2] {
    let e=trapetum::check_rope_m(nh,hd,5,m);
    let ok=e<5e-3; println!("  rope_m nh={nh} hd={hd} M={m}  rel_err={e:.2e}  {}", if ok{"OK"}else{"FAIL"});
    if !ok{bad+=1;} } }
  if bad==0{println!("\nALL PASS (batched forward blocks correct at M<=4)");}else{println!("\n{bad} FAIL");std::process::exit(1);}
}
