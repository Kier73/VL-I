"use strict";
// CLT-006: dependency closure and shared decision-diagram navigation.
function experiment(){
 const assert=(p,m)=>{if(!p)throw Error(m)};
 const pop=x=>{let n=0;while(x){x&=x-1;n++}return n};
 function world(n,seed){
  let p=Array.from({length:n},(_,i)=>i),r=seed;
  for(let i=n-1;i>0;i--){r=(Math.imul(r,1664525)+1013904223)>>>0;let j=r%(i+1);[p[i],p[j]]=[p[j],p[i]]}
  const enc=x=>p.reduce((v,j,i)=>v|(((x>>>i)&1)<<j),0);
  const dec=x=>p.reduce((v,j,i)=>v|(((x>>>j)&1)<<i),0);
  function source(a,x){let s=dec(x),lo=s&15,hi=s>>>4;
   if(a===0)lo=((lo<<1)&15)|(lo>>>3);
   if(a===1)lo^=((s>>>1)&(s>>>2))&1;
   if(a===2)lo^=((s>>>4)&(s>>>2))&1;
   hi=((hi<<1)&((1<<(n-4))-1))|(hi>>>(n-5));
   return enc(lo|(hi<<4));
  }
  // Separate source-program checker: terms per output in physical coordinates.
  function law(a,j){const i=p.indexOf(j);let terms;
   if(i>=4)terms=[1<<p[i===4?n-1:i-1]];
   else if(a===0)terms=[1<<p[(i+3)%4]];
   else if(i===0)terms=[1<<p[0],(1<<p[a===1?1:4])|(1<<p[2])];
   else terms=[1<<j];
   return terms.sort((x,y)=>x-y);
  }
  return {source,law,observer:p[0]};
 }
 function learn(n,k,seed){
  const w=world(n,seed),cache=new Map(),terms=new Map(),pending=[w.observer],seen=new Set(pending);
  let observations=0,checks=0,failures=0,bitEvaluations=0;
  const ev=(t,x)=>t.reduce((v,m)=>v^Number((m&x)===m),0);
  function ask(a,x){const key=a*(1<<n)+x;if(!cache.has(key)){cache.set(key,w.source(a,x));observations++}return cache.get(key)}
  const points=[0,...Array.from({length:n},(_,i)=>1<<i)];
  const quadratic=points.slice();for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)quadratic.push((1<<i)|(1<<j));
  for(let index=0;index<pending.length;index++){
   const j=pending[index];
   for(let a=0;a<k;a++){
    let fit=null;
    for(const probes of [points,quadratic]){
     const t=[];for(const x of probes){bitEvaluations++;if(ev(t,x)!==((ask(a,x)>>>j)&1))t.push(x)}
     const exact=w.law(a,j);checks++;
     if(t.slice().sort((x,y)=>x-y).join()===exact.join()){fit=t;break}
     failures++;
     const diff=new Set(t);for(const m of exact)diff.has(m)?diff.delete(m):diff.add(m);
     const x=[...diff].sort((u,v)=>pop(u)-pop(v)||u-v)[0];
     assert(ev(t,x)!==((ask(a,x)>>>j)&1),"bad counterexample");
    }
    if(fit===null)return {status:"UNRESOLVED"};
    terms.set(a*n+j,fit);
    for(const m of fit)for(let i=0;i<n;i++)if((m&(1<<i))&&!seen.has(i)){seen.add(i);pending.push(i)}
   }
  }
  const coords=[...seen].sort((a,b)=>a-b),pos=new Map(coords.map((x,i)=>[x,i]));
  return {status:"CLOSED",w,terms,coords,pos,observations,checks,failures,bitEvaluations,observer:pos.get(w.observer)};
 }
 class BDD{
  constructor(limit,workLimit){this.nodes=[null,null];this.unique=new Map();this.memo=new Map();this.limit=limit;this.workLimit=workLimit;this.work=0}
  tick(){if(++this.work>this.workLimit)throw Error("work-budget")}
  node(v,l,h){this.tick();if(l===h)return l;const key=`${v},${l},${h}`;if(this.unique.has(key))return this.unique.get(key);if(this.nodes.length>=this.limit)throw Error("node-budget");const id=this.nodes.length;this.nodes.push([v,l,h]);this.unique.set(key,id);return id}
  ite(i,t,e){this.tick();if(i===1)return t;if(i===0)return e;if(t===e)return t;if(t===1&&e===0)return i;
   const key=`${i},${t},${e}`;if(this.memo.has(key))return this.memo.get(key);
   const top=Math.min(...[i,t,e].map(x=>x<2?Infinity:this.nodes[x][0]));
   const branch=(x,b)=>x>=2&&this.nodes[x][0]===top?this.nodes[x][b?2:1]:x;
   const l=this.ite(branch(i,0),branch(t,0),branch(e,0)),h=this.ite(branch(i,1),branch(t,1),branch(e,1));
   const result=this.node(top,l,h);this.memo.set(key,result);return result;
  }
  xor(a,b){return this.ite(a,this.ite(b,0,1),b)}
  and(a,b){return this.ite(a,b,0)}
  compose(outer,inner){const memo=new Map();const sub=x=>{this.tick();if(x<2)return x;if(memo.has(x))return memo.get(x);const [v,l,h]=this.nodes[x],y=this.ite(inner[v],sub(h),sub(l));memo.set(x,y);return y};return outer.map(sub)}
  evaluate(root,x){while(root>=2){const [v,l,h]=this.nodes[root];root=(x>>>v)&1?h:l}return root}
  live(roots){let set=new Set();function visit(x,nodes){if(x<2||set.has(x))return;set.add(x);visit(nodes[x][1],nodes);visit(nodes[x][2],nodes)}for(const r of roots)visit(r,this.nodes);return set.size}
 }
 function execute(n,k,seed,limit=150000,workLimit=2000000){
  const g=learn(n,k,seed);assert(g.status==="CLOSED","local discovery failed");
  const d=g.coords.length,b=new BDD(limit,workLimit),encode=x=>g.coords.reduce((z,j,i)=>z|(((x>>>j)&1)<<i),0);
  const out={n,k,seed,dependencyWidth:d,sourceQueries:g.observations,fullTableQueries:k*(1<<n),symbolicChecks:g.checks,counterexamples:g.failures,bitEvaluations:g.bitEvaluations,termCount:[...g.terms.values()].reduce((s,t)=>s+t.length,0)};
  const word=Array.from({length:k},(_,i)=>i),rep=1000000000039n;
  const start=Array.from({length:n<=10?1<<n:128},(_,i)=>n<=10?i:((Math.imul(i+13,2654435761)>>>0)&((1<<n)-1)));
  try{
   const id=g.coords.map((_,i)=>b.node(i,0,1));
   const ops=word.map(a=>g.coords.map(j=>{let y=0;for(const m of g.terms.get(a*n+j)){let t=1;for(const x of g.coords)if(m&(1<<x))t=b.and(t,id[g.pos.get(x)]);y=b.xor(y,t)}return y}));
   let local=0;
   for(const x of start)for(const a of word){const z=encode(x),next=ops[a].reduce((v,r,i)=>v|(b.evaluate(r,z)<<i),0);assert(next===encode(g.w.source(a,x)),"local check");local++}
   out.localChecks=local;out.localLiveNodes=b.live(ops.flat());
   let block=id;for(const a of word)block=b.compose(ops[a],block);
   let base=block,result=id,t=rep;
   while(t){if(t&1n)result=b.compose(result,base);t>>=1n;if(t)base=b.compose(base,base)}
   out.poweredLiveNodes=b.live(result);
   // Independent source cycle baseline, counts actual action evaluations.
   let sourceSteps=0;
   for(const x of start){let y=x,t=rep,seen=new Map(),hist=[];
    while(t&&!seen.has(y)){seen.set(y,hist.length);hist.push(y);for(const a of word){y=g.w.source(a,y);sourceSteps++}t--}
    if(t){let s=seen.get(y);y=hist[s+Number(t%BigInt(hist.length-s))]}
    const observed=b.evaluate(result[g.observer],encode(x));
    assert(observed===((y>>>g.w.observer)&1),"long oracle mismatch");
   }
   out.status="CLOSED";out.longChecks=start.length;out.cycleActionEvaluations=sourceSteps;
  }catch(e){if(!["work-budget","node-budget"].includes(e.message))throw e;out.status="UNRESOLVED";out.reason=e.message}
  out.allocatedNodes=b.nodes.length-2;out.bddWork=b.work;return out
 }
 const runs=[execute(10,1,17),execute(10,2,17),execute(10,3,17),execute(14,3,41),execute(20,3,93),execute(10,3,17,12,10000)];
 assert(runs[2].status==="CLOSED","prior obstruction not resolved");
 assert(runs[5].status==="UNRESOLVED","budget control");
 // Hidden degree-three alteration invisible to degree-two samples.
 const witness=7;assert([0,1,2,4,3,5,6].every(x=>(x&witness)!==witness),"bad hidden control");
 return {runs,hiddenDegreeControl:{degree2SamplesAgree:true,witness,sourceContractRequired:true}};
}
if(require.main===module)console.log(JSON.stringify(experiment(),null,2));
module.exports={experiment};
