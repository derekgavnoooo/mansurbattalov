// Проверочные кадры: node keyframes.js <page.html> <outdir> <t1> <t2> ... [--lang=en]
// Страница должна объявлять window.renderFrame(t) и canvas#c.
const {chromium}=require('playwright');const fs=require('fs');const path=require('path');
const args=process.argv.slice(2),lang=(args.find(a=>a.startsWith('--lang='))||'').slice(7),rest=args.filter(a=>!a.startsWith('--'));
const [page,out,...ts]=rest;
(async()=>{const br=await chromium.launch();const p=await br.newPage();p.on('pageerror',e=>console.log('ERR',e.message));
if(lang)await p.addInitScript(l=>{window.LANG=l},lang);
await p.goto('file://'+path.resolve(page));fs.mkdirSync(out,{recursive:true});
for(let k=0;k<ts.length;k++){const d=await p.evaluate(t=>{renderFrame(t);return document.getElementById('c').toDataURL()},+ts[k]);
fs.writeFileSync(`${out}/k${k}.png`,Buffer.from(d.split(',')[1],'base64'))}
await br.close()})();
