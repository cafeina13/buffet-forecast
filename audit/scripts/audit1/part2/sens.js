// Scenario controls: effect of each control on guests and total kg (risk 10%) on every demo service.
const {execFileSync}=require('child_process');
const run=(sc)=>{execFileSync('node',['harness.js','tmp.json',JSON.stringify(sc)]);delete require.cache[require.resolve('./tmp.json')];return require('./tmp.json').filter(s=>s.risk===0.1);};
const base=run({});
const tot=(s)=>s.dishes.reduce((a,d)=>a+d.recKg,0);
const show=(name,sc,meals)=>{const r=run(sc);let lines=[];r.forEach((s,i)=>{if(meals&&!meals.includes(s.meal))return;const b=base[i];lines.push(`${s.date.slice(5)} ${s.meal[0]} g ${b.guests.toFixed(0)}->${s.guests.toFixed(0)} kg ${tot(b).toFixed(0)}->${tot(s).toFixed(0)}`)});console.log('##',name,'\n '+lines.join('\n '));};
show('child 5%',{child:0.05},['breakfast','dinner']);
show('child 40%',{child:0.40},['breakfast','dinner']);
show('temp -8',{temp:-8},['dinner']);
show('temp +8',{temp:8},['dinner']);
show('occ -25',{occ:-25},['dinner']);
show('match on',{match:true},['dinner']);
show('match off',{match:false},['dinner']);
show('themed on',{themed:true},['dinner']);
