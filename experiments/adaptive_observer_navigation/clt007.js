"use strict";
// CLT-007: source-qualified adaptive navigation under one charged envelope.
function experiment(){
 const assert=(b,m)=>{if(!b)throw Error(m)}, pop=x=>{let k=0;while(x){x&=x-1;k++}return k}, bit=(x,i)=>(x>>>i)&1;
 class Meter{constructor(limit){this.limit=limit;this.used=0}tick(k=1){if(this.used+k>this.limit)throw Error("budget");this.used+=k}}
 function world(kind,n,seed){
  const mask=(1<<n)-1,p=Array.from({length:n},(_,i)=>i);let r=seed;
  for(let i=n-1;i>0;i--){r=(Math.imul(r,1664525)+1013904223)>>>0;let j=r%(i+1);[p[i],p[j]]=[p[j],p[i]]}
  const enc=x=>p.reduce((v,j,i)=>v|(bit(x,i)<<j),0),dec=x=>p.reduce((v,j,i)=>v|(bit(x,j)<<i),0);
  function raw(s){
   if(kind==="affine")return ((s<<1)&mask)^(bit(s,n-1)?3:0);
   if(kind==="triangular"){const l=s&15,h=s>>>4;return (((l<<1)&15)|(bit(l,3)^(bit(l,1)&bit(h,0))))|((((h<<1)&((1<<(n-4))-1))|bit(h,n-5))<<4)}
   const v=bit(s,n-1)^(bit(s,1)&bit(s,2))^(kind==="cubic"?(bit(s,3)&bit(s,4)&bit(s,5)):0);
   return ((s<<1)&mask)|v;
  }
  function terms(j){
   const i=p.indexOf(j);let t;
   if(kind==="triangular"){
    if(i>=4)t=[1<<p[i===4?n-1:i-1]];
    else if(i===0)t=[1<<p[3],(1<<p[1])|(1<<p[4])];
    else t=[1<<p[i-1]];
   }else if(i===0)t=kind==="affine"?[1<<p[n-1]]:[1<<p[n-1],(1<<p[1])|(1<<p[2]),...(kind==="cubic"?[(1<<p[3])|(1<<p[4])|(1<<p[5])]:[])];
   else if(kind==="affine"&&i===1)t=[1<<p[0],1<<p[n-1]];
   else t=[1<<p[i-1]];
   return t.sort((a,b)=>a-b);
  }
  return {n,observer:p[0],source:x=>enc(raw(dec(x))),terms,enc,dec,raw};
 }
 function evaluate(terms,x,m){let v=0;for(const t of terms){m.tick();v^=Number((x&t)===t)}return v}
 function learn(w,m){
  const cache=new Map(),laws=new Map(),todo=[w.observer],known=new Set(todo);let counterexamples=0,checks=0;
  const ask=x=>{m.tick();if(!cache.has(x)){m.tick();cache.set(x,w.source(x))}return cache.get(x)};
  const linear=[0,...Array.from({length:w.n},(_,i)=>1<<i)],quad=linear.slice();
  for(let i=0;i<w.n;i++)for(let j=i+1;j<w.n;j++){m.tick();quad.push((1<<i)|(1<<j))}
  for(let i=0;i<todo.length;i++){
   const j=todo[i];let found=null;
   for(const probes of [linear,quad]){
    const t=[];for(const x of probes){m.tick();if(evaluate(t,x,m)!==bit(ask(x),j))t.push(x)}
    const truth=w.terms(j),diff=new Set(t);for(const u of truth){m.tick();diff.has(u)?diff.delete(u):diff.add(u)}checks++;
    if(!diff.size){found=t;break}
    counterexamples++;const witness=[...diff].sort((a,b)=>pop(a)-pop(b)||a-b)[0];
    assert(evaluate(t,witness,m)!==bit(ask(witness),j),"invalid witness");
   }
   if(!found)return {status:"UNRESOLVED",reason:"degree-bound",queries:cache.size,checks,counterexamples};
   laws.set(j,found);
   for(const t of found)for(let k=0;k<w.n;k++){m.tick();if(bit(t,k)&&!known.has(k)){known.add(k);todo.push(k)}}
  }
  return {status:"CLOSED",laws,coords:[...known].sort((a,b)=>a-b),queries:cache.size,checks,counterexamples};
 }
 function navigator(w,g,m){
  const coords=g.coords,laws=g.laws,mask=coords.reduce((v,j)=>v|(1<<j),0);
  const step=(x,js=coords)=>{let y=0;for(const j of js){m.tick();y|=evaluate(laws.get(j),x,m)<<j}return y};
  const depend=new Map(coords.map(j=>[j,laws.get(j).reduce((v,t)=>v|t,0)]));
  function reachable(start){let seen=1<<start,front=seen;while(front){const j=31-Math.clz32(front);front^=1<<j;m.tick();const more=depend.get(j)&~seen;seen|=more;front|=more}return seen}
  const reach=new Map(coords.map(j=>[j,reachable(j)]));
  const sink=coords.filter(j=>bit(reach.get(w.observer),j)&&bit(reach.get(j),w.observer));
  const driver=coords.filter(j=>!sink.includes(j)),sm=sink.reduce((v,j)=>v|(1<<j),0),dm=mask^sm;
  assert(driver.every(j=>!(depend.get(j)&sm)),"driver not closed");
  const affine=coords.every(j=>laws.get(j).every(t=>pop(t)<=1));
  const trace=[];
  function direct(x,t){
   let y=x&mask;const seen=new Map(),hist=[];
   while(t&&!seen.has(y)){m.tick();seen.set(y,hist.length);hist.push(y);y=step(y);t--}
   if(t){const s=seen.get(y);y=hist[s+Number(t%BigInt(hist.length-s))]}return y;
  }
  function affinePlan(t){
   const d=w.n,one=1<<d,identity=Array.from({length:d+1},(_,i)=>1<<i);
   let a=identity.slice();
   for(const j of coords)a[j]=laws.get(j).reduce((v,k)=>v^(k===0?one:k),0);
   const compose=(a,b)=>a.map(row=>{let out=0;while(row){m.tick();const j=31-Math.clz32(row);row^=1<<j;out^=b[j]}return out});
   let r=identity;while(t){m.tick();if(t&1n)r=compose(a,r);t>>=1n;if(t)a=compose(a,a)}
   return x=>{let y=0;for(const j of coords){m.tick();y|=(pop(r[j]&(x|one))&1)<<j}return y};
  }
  function phasePlan(anchor,t){
   let h=anchor&dm,hs=[],seen=new Map();
   while(!seen.has(h)){m.tick();seen.set(h,hs.length);hs.push(h);h=step(h,driver)}
   const pre=seen.get(h),period=hs.length-pre;
   const pack=z=>sink.reduce((v,j,i)=>v|(bit(z,j)<<i),0),unpack=z=>sink.reduce((v,j,i)=>v|(bit(z,i)<<j),0);
   const advance=(z,h)=>pack(step(unpack(z)|h,sink));
   const lead=Number(t<BigInt(pre)?t:BigInt(pre));
   let remainder=t-BigInt(lead),reps=remainder/BigInt(period),tail=Number(remainder%BigInt(period));
   const dim=1<<sink.length;
   let transform=Array.from({length:dim},(_,z)=>z);
   if(reps){let p=transform.map(z=>{for(let i=pre;i<hs.length;i++)z=advance(z,hs[i]);return z});
    while(reps){m.tick();if(reps&1n)transform=transform.map(z=>{m.tick();return p[z]});reps>>=1n;if(reps)p=p.map(z=>{m.tick();return p[z]})}}
   trace.push({preperiod:pre,period,sinkStates:dim});
   return x=>{
    let z=pack(x);for(let i=0;i<lead;i++)z=advance(z,hs[i]);
    z=transform[z];for(let i=0;i<tail;i++)z=advance(z,hs[pre+i]);
    const idx=t<BigInt(hs.length)?Number(t):pre+Number((t-BigInt(pre))%BigInt(period));
    return unpack(z)|hs[idx];
   };
  }
  function batch(xs,t,forceDirect=false){
   const route=forceDirect?"direct-cycle":affine?"affine-power":driver.length&&sink.length<=8?"conditional-phase":"direct-cycle";
   const output=[],start=m.used,attempts=[];
   try{
    if(route==="affine-power"){const run=affinePlan(t);for(const x of xs)output.push(run(x))}
    else if(route==="conditional-phase"){
     const plans=new Map();for(const x of xs){m.tick();const h=x&dm;if(!plans.has(h))plans.set(h,phasePlan(x,t));output.push(plans.get(h)(x))}
    }else for(const x of xs)output.push(direct(x,t));
    attempts.push({route,status:"EXACT",work:m.used-start});return {status:"EXACT",route,work:m.used-start,output,trace,attempts,sink:sink.length,driver:driver.length};
   }catch(e){if(e.message!=="budget")throw e;attempts.push({route,status:"UNRESOLVED",work:m.used-start});return {status:"UNRESOLVED",route,reason:"shared-work-budget",work:m.used-start,completed:output.length,trace,attempts,sink:sink.length,driver:driver.length}}
  }
  return {batch,mask};
 }
 function rawCycle(w,x,t,max=1000000){
  let y=x,seen=new Map(),hist=[];
  while(t&&!seen.has(y)){if(hist.length===max)return null;seen.set(y,hist.length);hist.push(y);y=w.source(y);t--}
  if(t){const s=seen.get(y);y=hist[s+Number(t%BigInt(hist.length-s))]}return y;
 }
 function affineOracle(w,x,t){
  const n=w.n,mask=(1<<n)-1,shift=a=>((a<<1)&mask)^(bit(a,n-1)?3:0);
  function mul(a,b){let v=0;while(b){if(b&1)v^=a;b>>>=1;a=shift(a)}return v}
  let a=2,r=1;while(t){if(t&1n)r=mul(r,a);t>>=1n;if(t)a=mul(a,a)}
  return w.enc(mul(w.dec(x),r));
 }
 const T=1000000000039n;
 function run(kind,n,seed,budget=300000){
  const w=world(kind,n,seed),m=new Meter(budget),g=learn(w,m),discoveryWork=m.used;
  if(g.status!=="CLOSED")return {kind,n,discoveryWork,...g};
  // One shared upstream anchor and sixteen distinct low initial states; concealed labels.
  const xs=Array.from({length:16},(_,i)=>w.enc((((0x5a53&((1<<(n-4))-1))<<4)|i)&((1<<n)-1)));
  const nav=navigator(w,g,m),result=nav.batch(xs,T);
  const baselineMeter=new Meter(budget-discoveryWork),base=navigator(w,g,baselineMeter).batch(xs,T,true);
  let verified=0,shortChecks=0;
  // Independent source controls, never used in route selection.
  if(result.status==="EXACT"){
   for(let i=0;i<xs.length;i++){
    const expected=kind==="affine"?affineOracle(w,xs[i],T):rawCycle(w,xs[i],T);
    assert(expected!==null&&result.output[i]===(expected&nav.mask),"long disagreement");verified++;
   }
  }
  if(base.status==="EXACT")for(let i=0;i<xs.length;i++){const expected=kind==="affine"?affineOracle(w,xs[i],T):rawCycle(w,xs[i],T);assert(expected!==null&&base.output[i]===(expected&nav.mask),"baseline disagreement")}
  const shortMeter=new Meter(300000),short=navigator(w,g,shortMeter).batch(xs,257n);
  assert(short.status==="EXACT","short control failed");
  for(let i=0;i<xs.length;i++){let x=xs[i];for(let j=0;j<257;j++)x=w.source(x);assert(short.output[i]===(x&nav.mask),"short disagreement");shortChecks++}
  const clean=({output,...rest})=>rest;
  return {kind,n,budget,discoveryWork,queries:g.queries,checks:g.checks,counterexamples:g.counterexamples,retained:g.coords.length,adaptive:clean(result),baseline:clean(base),totalAdaptive:m.used,totalBaseline:discoveryWork+baselineMeter.used,verified,shortChecks};
 }
 const runs=[run("affine",20,17),run("triangular",20,41),run("triangular",24,93),run("nonlinear",24,39),run("cubic",12,11)];
 assert(runs[0].adaptive.status==="EXACT"&&runs[1].adaptive.status==="EXACT"&&runs[2].adaptive.status==="EXACT","structured routes failed");
 assert(runs[4].status==="UNRESOLVED","hidden cubic not rejected");
 return {time:T.toString(),runs};
}
if(require.main===module)console.log(JSON.stringify(experiment(),null,2));
module.exports={experiment};
