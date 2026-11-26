"use strict";
// CLT-008: persistent exact partial trajectories under a shared budget.
function experiment(){
 const assert=(v,msg)=>{if(!v)throw Error(msg)}, bit=(x,j)=>(x>>>j)&1,pop=x=>{let n=0;while(x){x&=x-1;n++}return n};
 const n=24,N=1<<n,mask=N-1;
 class Meter{constructor(limit){this.limit=limit;this.used=0}pay(k=1){if(this.used+k>this.limit)return false;this.used+=k;return true}}
 function fixture(seed,variant=false){
  const p=Array.from({length:n},(_,i)=>i);let r=seed;
  for(let i=n-1;i>0;i--){r=(Math.imul(r,1664525)+1013904223)>>>0;const j=r%(i+1);[p[i],p[j]]=[p[j],p[i]]}
  const enc=x=>p.reduce((v,j,i)=>v|(bit(x,i)<<j),0),dec=x=>p.reduce((v,j,i)=>v|(bit(x,j)<<i),0);
  const raw=x=>((x<<1)&mask)|((bit(x,n-1)^(bit(x,1)&bit(x,2))^(variant?bit(x,4):0))&1);
  function terms(j){const i=p.indexOf(j);return (i?[1<<p[i-1]]:[1<<p[n-1],(1<<p[1])|(1<<p[2]),...(variant?[1<<p[4]]:[])]).sort((a,b)=>a-b)}
  return {observer:p[0],enc,dec,source:x=>enc(raw(dec(x))),terms,raw};
 }
 // Same local Boolean interpolation contract as CLT-007; source permutation and coefficients withheld.
 function discover(w,m){
  const cache=new Map(),laws=new Map(),todo=[w.observer],known=new Set(todo);let checks=0,counterexamples=0;
  const charge=k=>{if(!m.pay(k))throw Error("discovery budget")};
  const evalp=(ts,x)=>{let v=0;for(const t of ts){charge(1);v^=Number((x&t)===t)}return v};
  const ask=x=>{charge(1);if(!cache.has(x)){charge(1);cache.set(x,w.source(x))}return cache.get(x)};
  const linear=[0,...Array.from({length:n},(_,i)=>1<<i)],quad=linear.slice();
  for(let i=0;i<n;i++)for(let j=i+1;j<n;j++){charge(1);quad.push((1<<i)|(1<<j))}
  for(let i=0;i<todo.length;i++){
   const j=todo[i];let fit=null;
   for(const points of [linear,quad]){
    const t=[];for(const x of points){charge(1);if(evalp(t,x)!==bit(ask(x),j))t.push(x)}
    const diff=new Set(t);for(const k of w.terms(j)){charge(1);diff.has(k)?diff.delete(k):diff.add(k)}checks++;
    if(!diff.size){fit=t;break}counterexamples++;
    const x=[...diff].sort((a,b)=>pop(a)-pop(b)||a-b)[0];assert(evalp(t,x)!==bit(ask(x),j),"invalid counterexample");
   }
   assert(fit!==null,"degree refusal");
   laws.set(j,fit);
   for(const t of fit)for(let k=0;k<n;k++){charge(1);if(bit(t,k)&&!known.has(k)){known.add(k);todo.push(k)}}
  }
  assert(known.size===n,"expected full dependence");
  const ordered=[...laws].sort((a,b)=>a[0]-b[0]);
  const step=x=>ordered.reduce((y,[j,ts])=>y|(ts.reduce((v,t)=>v^Number((x&t)===t),0)<<j),0);
  const cost=ordered.reduce((v,[,ts])=>v+1+ts.length,0);
  return {step,cost,queries:cache.size,checks,counterexamples,signature:JSON.stringify(ordered)};
 }
 // Each stored state has a unique (segment,index) address. Segment tails may
 // connect to prior segments, including cycles across multiple segments.
 class Atlas{
  constructor(g,cap){this.g=g;this.cap=cap;this.index=new Map();this.segs=[];this.edges=0;this.jumps=0;this.skipped=0n}
  run(job,m,quantum){
   let spent=0;
   const pay=k=>{if(spent+k>quantum||!m.pay(k))return false;spent+=k;return true};
   let x=job.x,t=job.t,visited=new Map();
   while(t){
    if(!pay(1)){job.x=x;job.t=t;return "PAUSED"}
    let loc=this.index.get(x);
    if(!loc){
     if(this.index.size>=this.cap){job.x=x;job.t=t;return "MEMORY"}
     if(!pay(2)){job.x=x;job.t=t;return "PAUSED"}
     const id=this.segs.length;this.segs.push({states:[x],link:null});loc=[id,0];this.index.set(x,loc);
    }
    const [id,i]=loc,s=this.segs[id],d=s.states.length-1-i;
    if(d){
     const k=t<BigInt(d)?Number(t):d;
     x=s.states[i+k];t-=BigInt(k);this.jumps++;this.skipped+=BigInt(k);continue;
    }
    if(visited.has(x)){
     const period=visited.get(x)-t;
     assert(period>0n,"zero period");const removed=t-t%period;t%=period;this.skipped+=removed;visited.clear();if(!t)break;
    }
    visited.set(x,t);
    if(s.link!==null){x=s.link;t--;continue}
    if(this.index.size>=this.cap){job.x=x;job.t=t;return "MEMORY"}
    if(!pay(this.g.cost+3)){job.x=x;job.t=t;return "PAUSED"}
    const y=this.g.step(x);this.edges++;
    if(this.index.has(y))s.link=y;
    else{this.index.set(y,[id,s.states.length]);s.states.push(y)}
    x=y;t--;
   }
   job.x=x;job.t=0n;job.result=x;return "EXACT";
  }
  audit(w){
   let count=0;
   for(let id=0;id<this.segs.length;id++){const s=this.segs[id];for(let i=0;i<s.states.length;i++){
    const x=s.states[i],loc=this.index.get(x);assert(loc&&loc[0]===id&&loc[1]===i,"index mismatch");
    const y=i+1<s.states.length?s.states[i+1]:s.link;
    if(y!==null){assert(w.source(x)===y,"edge audit mismatch");count++}
   }}
   assert(count===this.edges,"edge count mismatch");return count;
  }
 }
 // Conventional per-request orbit search, with the SAME round robin scheduling.
 // Each job retains its own exact orbit prefix, but never shares it with other jobs.
 class Independent{
  constructor(g,cap){this.g=g;this.cap=cap;this.states=0;this.edges=0}
  run(job,m,quantum){
   let spent=0;const pay=k=>{if(spent+k>quantum||!m.pay(k))return false;spent+=k;return true};
   if(!job.seen){job.seen=new Map();job.hist=[]}
   while(job.t){
    if(!pay(1))return "PAUSED";
    if(job.seen.has(job.x)){
     const i=job.seen.get(job.x),p=job.hist.length-i;
     job.x=job.hist[i+Number(job.t%BigInt(p))];job.t=0n;break;
    }
    if(this.states>=this.cap)return "MEMORY";
    if(!pay(this.g.cost+3))return "PAUSED";
    job.seen.set(job.x,job.hist.length);job.hist.push(job.x);this.states++;this.edges++;
    job.x=this.g.step(job.x);job.t--;
   }
   job.result=job.x;return "EXACT";
  }
 }
 const T=1000000000039n,limit=300000,cap=6000;
 function jobs(xs,time=T){return xs.map((x,i)=>({id:i,anchor:x,x,t:time,status:"PAUSED"}))}
 function schedule(engine,js,m,quantum=4000){
  let rounds=0;
  while(js.some(j=>j.status==="PAUSED")){
   const before=m.used;rounds++;
   for(const j of js)if(j.status==="PAUSED")j.status=engine.run(j,m,quantum);
   if(m.used-before<50||m.limit-m.used<50)break;
  }
  return {rounds,complete:js.filter(j=>j.status==="EXACT").length,unresolved:js.filter(j=>j.status!=="EXACT").map(j=>j.id)};
 }
 function check(w,js,time){
  let verified=0,steps=0;
  for(const j of js)if(j.status==="EXACT"){
   let s=w.dec(j.anchor),t=time,seen=new Map(),hist=[];
   while(t&&!seen.has(s)){assert(hist.length<2000000,"oracle budget");seen.set(s,hist.length);hist.push(s);s=w.raw(s);steps++;t--}
   if(t){const i=seen.get(s);s=hist[i+Number(t%BigInt(hist.length-i))]}
   assert(w.enc(s)===j.result,"independent source disagreement");verified++;
  }
  return {verified,steps};
 }
 function run(seed){
  const w=fixture(seed),am=new Meter(limit),g=discover(w,am),discovery=am.used,bm=new Meter(limit);assert(bm.pay(discovery),"baseline discovery");
  const a=new Atlas(g,cap),b=new Independent(g,cap);
  const xs=Array.from({length:16},(_,i)=>w.enc((0x5a53<<4)|i));
  const ja=jobs(xs),jb=jobs(xs);
  const ar=schedule(a,ja,am),br=schedule(b,jb,bm);
  const av=check(w,ja,T),bv=check(w,jb,T);
  const failed=ja.find(j=>j.id===8&&j.status!=="EXACT");
  assert(failed,"long orbit expected to remain open");
  const loc=a.index.get(xs[8]);assert(loc,"missing retained failed prefix");
  const prefixStates=a.segs[loc[0]].states.length;
  // New intentions from the failed prefix, generated independently by direct
  // source recurrence; no output cache, new time coordinates and anchors.
  let s=xs[8],held=[];for(let i=0;i<32;i++){if(i%4===0)held.push(s);s=w.source(s)}
  const horizon=64n,ma=new Meter(12000),mb=new Meter(12000),ha=jobs(held,horizon),hb=jobs(held,horizon);
  const beforeEdges=a.edges,sa=schedule(a,ha,ma),sb=schedule(b,hb,mb);
  const va=check(w,ha,horizon),vb=check(w,hb,horizon);
  // Separate epoch: explicitly instantiate new memory for changed transition law.
  const changed=fixture(seed,true),cm=new Meter(300000),cg=discover(changed,cm);
  assert(cg.signature!==g.signature,"changed source identity not detected");
  const fresh=new Atlas(cg,cap);
  const j=jobs([xs[8]],64n),jm=new Meter(12000);schedule(fresh,j,jm);
  const cv=check(changed,j,64n);assert(cv.verified===1,"changed epoch control");
  return {seed,discovery,sourceQueries:g.queries,proofChecks:g.checks,sourceCounterexamples:g.counterexamples,
   first:{atlas:{...ar,work:am.used,verified:av.verified,auditSteps:av.steps},independent:{...br,work:bm.used,verified:bv.verified,auditSteps:bv.steps},failedPrefixStates:prefixStates},
   followup:{atlas:{...sa,work:ma.used,verified:va.verified,newEdges:a.edges-beforeEdges},independent:{...sb,work:mb.used,verified:vb.verified},horizon:horizon.toString()},
   retained:{states:a.index.size,segments:a.segs.length,edges:a.edges,auditedEdges:a.audit(w),jumpCount:a.jumps,skippedTransitions:a.skipped.toString()},changedEpoch:{signatureChanged:true,oldAtlasReused:false,verified:cv.verified}};
 }
 const runs=[run(39),run(93)];
 // Exact cross-segment and non-injective controls.
 const g={step:x=>[1,2,3,4,2,4][x],cost:1,signature:"six-state"};
 const a=new Atlas(g,20),m=new Meter(10000);
 const j1={x:0,t:2n};assert(a.run(j1,m,100)==="EXACT"&&j1.result===2,"prefix");
 const j2={x:5,t:1n};assert(a.run(j2,m,100)==="EXACT"&&j2.result===4,"second prefix");
 const j3={x:0,t:1000000000039n};assert(a.run(j3,m,1000)==="EXACT"&&j3.result===4,"cross-segment cycle");
 const j4={x:5,t:1000000000039n};assert(a.run(j4,m,1000)==="EXACT"&&j4.result===4,"merge path");
 const small=new Atlas(g,2),j5={x:0,t:10n};assert(small.run(j5,m,100)==="MEMORY","memory refusal");
 return {time:T.toString(),limit,cap,runs,controls:{crossSegmentCycle:true,nonInjectiveMerge:true,memoryRefusal:true}};
}
if(require.main===module)console.log(JSON.stringify(experiment(),null,2));
module.exports={experiment};
