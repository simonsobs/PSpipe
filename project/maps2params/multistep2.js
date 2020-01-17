function mod(a,b) { var c = a%b; return c<0?c+b:c; }
function ofclass(a,b) { return a.className.split(' ').indexOf(b) >= 0; }
function visible(elem) {
	while(elem) {
		if(elem.style.display == 'none') return false;
		elem = elem.parentElement;
	}
	return true;
}

var steps = {}; // {'name':  { keys: [...], i: index}, ...}
var descs = {}; // {'class': { sels: recursive sels, matcher:, extractor:, formatter: }, ...}
var reset_key = " ".charCodeAt(0);

function match_visible(elem) { return visible(elem); }
function img_base(elem) { return elem.src.split('/').pop().split(".")[0]; }
function img_dir(elem) { return elem.src.split('/').slice(0,-1).pop(); }
function data_desc(elem) { return elem.dataset.desc; }

function add_step(name, keys, i) {tmp = {}; tmp[name] = {keys:keys, i:i}; add_steps(tmp); }
function add_steps(isteps) {
	// Adds steps as defined by {'name': { keys: [...], (i: 0)}, ...}
	for(var name in isteps) {
		var val = isteps[name];
		for(var i = 0; i < val.keys.length; i++)
			if(typeof val.keys[i] == 'string')
				val.keys[i] = val.keys[i].toUpperCase().charCodeAt(0);
		val.i = val.i || 0;
		val.i0 = val.i;
		steps[name] = val;
	}
}

function add_desc(name, props) { tmp = {}; tmp[name] = props; add_descs(tmp); }
function add_descs(idescs) {
	for(var name in idescs) {
		var val = idescs[name];
		val.match   = val.match   || match_visible;
		val.format  = val.format  || img_base;
		descs[name] = val;
	}
}

function set_reset(key) {
	if(key == null || key.length == 0)
		reset_key = 0;
	else
		reset_key = key.toUpperCase().charCodeAt(0);
}

function apply_steps() {
	// For each of our steps, set all but the selected child to hidden
	for(var name in steps) {
		var matches = document.querySelectorAll("." + name);
		for(var i = 0; i < matches.length; i++) {
			var kids = matches[i].children;
			for(var j = 0; j < kids.length; j++)
				kids[j].style.display = j == mod(steps[name].i, kids.length) ? 'initial' : 'none';
		}
	}
}

function apply_descs() {
	// For each of our descs, update its value based on matching elements
	for(var name in descs) {
		var desc = descs[name];
		var candidates = document.querySelectorAll(desc.sel);
		text = "undefined";
		for(var i = 0; i < candidates.length; i++) {
			var cand = candidates[i];
			if(desc.match(cand))
				text = desc.format(cand);
		}
		var targets = document.querySelectorAll("."+name);
		for(var i = 0; i < targets.length; i++)
			targets[i].innerHTML = text;
	}
}

function update() {
	apply_steps();
	apply_descs();
}

function checkKey(e) {
	e = e || window.event;
	var changed = 0;
	for(var name in steps) {
		for(var i = 0; i < steps[name].keys.length; i++)
			if(e.keyCode == steps[name].keys[i]) {
				steps[name].i += 1-2*i;
				changed = 1;
				break;
			}
	}
	if(e.keyCode == reset_key) {
		for(var name in steps) {
			steps[name].i = steps[name].i0;
			changed = 1;
		}
	}
	if(changed) update();
}

document.addEventListener("keydown", checkKey);
window.addEventListener("load", update);
