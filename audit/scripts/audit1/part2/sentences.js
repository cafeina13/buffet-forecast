// Classifies every explanation clause on the demo (all services, risk 10%; sentences do not depend on risk except the unit count).
const o=require('./base.json').filter(s=>s.risk===0.1); const sens=require('./sens.json');
const cnt={};const ex={};const add=(k,e)=>{cnt[k]=(cnt[k]||0)+1;if(!ex[k])ex[k]=e;};
let n=0;
for(const s of o)for(const d of s.dishes){n++;
  const parts=d.reason.split('; ').slice(1);
  for(let p of parts){p=p.replace(/\..*$/,'');
    const up=p.endsWith('artırıyor');
    if(p.startsWith('hava')){const sv=sens[d.id];
      const t=+p.match(/hava (\d+)/)[1];
      if(sv===0) add('temp: dish has zero temperature sensitivity in generator',`${s.date} ${s.meal} ${d.name}: ${p}`);
      else { const want=(t>29.96)?(sv>0):(sv<0); add(want===up?'temp: direction matches generator':'temp: direction CONTRADICTS generator',`${s.date} ${s.meal} ${d.name} (sens ${sv}): ${p}`);} }
    else if(p.startsWith('akşam maç')||p.startsWith('tema')||p.startsWith('kongre')) add('event clause (no per-guest effect in generator): '+p.split(',')[0]+(up?' up':' down'),`${s.date} ${s.meal} ${d.name}: ${p}`);
    else if(p.startsWith('çocuklu')) add('child '+(up?'up':'down'),`${s.date} ${s.meal} ${d.name}: ${p}`);
    else if(p.startsWith('otele yeni')) add('first-meal '+s.meal+' '+(up?'up':'down'),`${s.date} ${s.meal} ${d.name}: ${p}`);
    else if(p.startsWith('menüde')) add('n_similar '+s.meal,`${s.date} ${s.meal} ${d.name}: ${p}`);
    else add('other',p);
  }
  if(d.reason.includes('Yeni yemek')) add('new-dish sentence',`${s.date} ${s.meal} ${d.name}`);
  if(d.isNew && !d.reason.includes('Yeni yemek')) add('YENİ badge without new-dish sentence',`${s.date} ${s.meal} ${d.name}`);
}
console.log('dish cards',n);for(const k of Object.keys(cnt).sort())console.log(cnt[k],k,'\n    e.g.',ex[k]);
