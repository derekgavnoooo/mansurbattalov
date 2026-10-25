// Все кадры ролика: node render.js <page.html> <outdir> [--lang=en]
const {chromium}=require('playwright');const fs=require('fs');const path=require('path');
const args=process.argv.slice(2),lang=(args.find(a=>a.startsWith('--lang='))||'').slice(7),[page,out]=args.filter(a=>!a.startsWith('--'));
(async()=>{const br=await chromium.launch();const p=await br.newPage();p.on('pageerror',e=>console.log('ERR',e.message));
if(lang)await p.addInitScript(l=>{window.LANG=l},lang);
await p.goto('file://'+path.resolve(page));fs.mkdirSync(out,{recursive:true});
const n=Math.floor(await p.evaluate(()=>DUR*FPS));
for(let f=0;f<n;f++){const d=await p.evaluate(f=>{renderFrame(f/FPS);return document.getElementById('c').toDataURL()},f);
fs.writeFileSync(`${out}/${String(f).padStart(5,'0')}.png`,Buffer.from(d.split(',')[1],'base64'))}
await br.close();console.log('frames',n)})();
