"use strict";
// CLT-005 sparse observation + exact symbolic source verifier.
function experiment(){
 const width=10,N=1<<width,domain=Array.from({length:N},(_,i)=>i);
 const assert=(p,m)=>{if(!p)throw Error(m)};
 const weight=x=>{let n=0;while(x){x&=x-1;n++}return n};
 const xor=(a,b)=>{const c=new Set(a);for(const x of b)c.has(x)?c.delete(x):c.add(x);return c};
 const mul=(a,b,counters)=>{let c=new Set();for(const x of a)for(const y of b){counters.products++;let z=x|y;c.has(z)?c.delete(z):c.add(z)}return c};
 const evalp=(p,x)=>{let v=0;for(const m of p)v^=Number((x&m)===m);return v};
 const equal=(a,b)=>a.size===b.size&&[...a].every(x=>b.has(x));
 // Private source program, independently rendered as Boolean arithmetic and ANF.
 const action=(a,x)=>{
   let lo=x&15, hi=x>>4;
   if(a===0)lo=((lo<<1)&15)|(lo>>3);
   if(a===1)lo^=((x>>1)&(x>>2))&1;
   if(a===2)lo^=((x>>4)&(x>>2))&1;
   // Rotate nuisance bits. Under coupling these are not nuisance anymore.
   return lo|((((hi<<1)&63)|(hi>>5))<<4);
 };
 const program=a=>{
   const p=Array.from({length:width},(_,i)=>new Set([1<<i]));
   for(let i=4;i<width;i++)p[i]=new Set([1<<(i===4?9:i-1)]);
   if(a===0)for(let i=0;i<4;i++)p[i]=new Set([1<<((i+3)%4)]);
   if(a===1)p[0]=new Set([1,6]);
   if(a===2)p[0]=new Set([1,20]);
   return p;
 };
 function form(k,maxDegree,rankLimit){
   const counters={queries:0,challengeQueries:0,cacheHits:0,products:0,verificationChecks:0,failedDegreeAttempts:0,interpolationEvaluations:0,eliminationXors:0};
   const cache=new Map(), evidence=[];
   function ask(a,x,challenge=false){
     const key=a*N+x;
     if(cache.has(key)){counters.cacheHits++;return cache.get(key)}
     const y=action(a,x);cache.set(key,y);counters.queries++;if(challenge)counters.challengeQueries++;
     return y;
   }
   // Verifier holds pinned source program. The learner receives only pass or a separating input.
   function check(p,a,candidate){
     counters.verificationChecks++;
     let exact=new Set(),pr=program(a);
     for(const m of p){let term=new Set([0]);for(let i=0;i<width;i++)if(m&(1<<i))term=mul(term,pr[i],counters);exact=xor(exact,term)}
     const discrepancy=xor(exact,candidate);
     if(!discrepancy.size)return null;
     // A minimal-degree monomial is a witness: no strict subset term exists.
     return [...discrepancy].sort((x,y)=>weight(x)-weight(y)||x-y)[0];
   }
   function acquire(p,a){
     for(let d=1;d<=maxDegree;d++){
       let candidate=new Set();
       const points=domain.filter(x=>weight(x)<=d).sort((x,y)=>weight(x)-weight(y)||x-y);
       for(const x of points){counters.interpolationEvaluations++;if(evalp(p,ask(a,x))!==evalp(candidate,x))candidate.add(x)}
       const witness=check(p,a,candidate);
       if(witness===null){evidence.push({action:a,degree:d,terms:candidate.size});return candidate}
       counters.failedDegreeAttempts++;
       assert(evalp(p,ask(a,witness,true))!==evalp(candidate,witness),"invalid verifier witness");
     }
     return null;
   }
   const basis=[],piv=new Map();
   function add(p){
     let r=new Set(p);
     while(r.size){
       const m=Math.max(...r);
       if(piv.has(m)){r=xor(r,basis[piv.get(m)]);counters.eliminationXors++}
       else{if(basis.length>=rankLimit)return false;piv.set(m,basis.length);basis.push(r);return true}
     }return true;
   }
   function express(p){
     let r=new Set(p),c=0n;
     while(r.size){const m=Math.max(...r),j=piv.get(m);assert(j!==undefined,"unclosed");r=xor(r,basis[j]);c|=1n<<BigInt(j)}return c
   }
   add(new Set([1]));
   const pullbacks=[];
   for(let i=0;i<basis.length;i++){
     const row=[];
     for(let a=0;a<k;a++){
       const p=acquire(basis[i],a);
       if(p===null||!add(p))return {status:"UNRESOLVED",rank:basis.length,counters,evidenceCount:evidence.length,fullTableQueries:N*k};
       row.push(p);
     }
     pullbacks.push(row);
   }
   const ops=Array.from({length:k},(_,a)=>pullbacks.map(r=>express(r[a])));
   const encode=x=>basis.reduce((z,p,i)=>z|(BigInt(evalp(p,x))<<BigInt(i)),0n);
   const parity=z=>{let p=0;while(z){z&=z-1n;p^=1}return p};
   const apply=(m,z)=>m.reduce((v,r,i)=>v|(BigInt(parity(r&z))<<BigInt(i)),0n);
   const compose=(a,b)=>a.map(r=>{let v=0n;while(r){const j=r.toString(2).length-1;r^=1n<<BigInt(j);v^=b[j]}return v});
   // Independent exhaustive validation, separate from discovery query accounting.
   let checked=0;for(const x of domain)for(let a=0;a<k;a++){assert(apply(ops[a],encode(x))===encode(action(a,x)),"independent commutation");checked++}
   let longChecks=0;
   const word=Array.from({length:k},(_,i)=>i);
   let M=basis.map((_,i)=>1n<<BigInt(i));
   for(const a of word)M=compose(ops[a],M);
   let power=M, result=basis.map((_,i)=>1n<<BigInt(i)),t=1000000000039n;
   while(t){if(t&1n)result=compose(result,power);t>>=1n;if(t)power=compose(power,power)}
   const obs=express(new Set([1]));
   for(const x of domain){
     let y=x,t=1000000000039n,seen=new Map(),hist=[];
     while(t&&!seen.has(y)){seen.set(y,hist.length);hist.push(y);for(const a of word)y=action(a,y);t--}
     if(t){const start=seen.get(y);y=hist[start+Number(t%BigInt(hist.length-start))]}
     assert(parity(obs&apply(result,encode(x)))===(y&1),"long horizon");longChecks++
   }
   return {status:"CLOSED",rank:basis.length,classes:new Set(domain.map(x=>encode(x).toString())).size,counters,evidenceCount:evidence.length,fullTableQueries:N*k,checked,longChecks,maxTerms:Math.max(...basis.map(p=>p.size)),maxDegree:Math.max(...basis.flatMap(p=>[...p].map(weight)))};
 }
 const rotation=form(1,4,128),shear=form(2,4,128),coupled=form(3,4,128),budget=form(3,1,128);
 assert(rotation.status==="CLOSED"&&shear.status==="CLOSED","expected closure");assert(budget.status==="UNRESOLVED","expected refusal");
 // Out-of-family hidden monomial: low-degree samples cannot distinguish zero from a high-degree term.
 const hidden=new Set([1023]),sampled=domain.filter(x=>weight(x)<=4);
 assert(sampled.every(x=>evalp(hidden,x)===0)&&evalp(hidden,1023)===1,"hidden control");
 return {width,rotation,shear,coupled,budget,hidden:{samples:sampled.length,witness:1023,zeroPredictionRejected:true}};
}
if(require.main===module)console.log(JSON.stringify(experiment(),null,2));
module.exports={experiment};
