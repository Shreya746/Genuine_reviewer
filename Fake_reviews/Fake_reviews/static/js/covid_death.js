var visual_graphs = document.getElementById('visual');
var cla = document.getElementById('classification');
var con = document.getElementById('confusion');
var acc = document.getElementById('accuracy');
var los = document.getElementById('loss');
var tab = document.getElementById('table');
var tab2 = document.getElementById('table2');
var acc2 = document.getElementById('acc');
var pred = document.getElementById('pred');


function visual(){

visual_graphs.style.display='block';
cla.style.display='none';
con.style.display='none';
acc.style.display='none';
los.style.display='none';

}


function classify(){
cla.style.display='block';
con.style.display='none';
visual_graphs.style.display='none';
acc.style.display='none';
los.style.display='none';
}

function confusion(){
cla.style.display='none';
con.style.display='block';
visual_graphs.style.display='none';
acc.style.display='none';
los.style.display='none';
}

function accuracy(){
cla.style.display='none';
con.style.display='none';
acc.style.display='block';
los.style.display='none';
visual_graphs.style.display='none';
}

function loss(){
cla.style.display='none';
con.style.display='none';
acc.style.display='none';
los.style.display='block';
visual_graphs.style.display='none';
}

function standard(){
tab.style.display = 'none'
tab2.style.display = 'block'
acc2.style.display = 'none'
pred.style.display = 'none'
}

function acc_score(){
tab.style.display = 'none'
tab2.style.display = 'none'
acc2.style.display = 'block'
pred.style.display = 'none'
}

function prediction(){
tab.style.display = 'none'
tab2.style.display = 'none'
acc2.style.display = 'none'
pred.style.display = 'block'
}
function features(){
tab.style.display = 'block'
tab2.style.display = 'none'
acc2.style.display = 'none'
pred.style.display = 'none'
}



//var toggle = true;
//function features(){
//if (toggle){
//toggle = false;
//tab.style.display='block';
//return;
//}
//if (!toggle){
//toggle = true;
//tab.style.display='none';
//return;
//}
//}