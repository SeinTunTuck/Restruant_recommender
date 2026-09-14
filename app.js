const FILES = {
  profiles: 'dataset/userprofile.csv', places: 'dataset/geoplaces2.csv', ratings: 'dataset/rating_final.csv',
  userCuisines: 'dataset/usercuisine.csv', userPayments: 'dataset/userpayment.csv',
  cuisines: 'dataset/chefmozcuisine.csv', payments: 'dataset/chefmozaccepts.csv',
  parking: 'dataset/chefmozparking.csv', hours: 'dataset/chefmozhours4.csv'
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const clean = (value = '') => value === '?' ? '' : value.replaceAll('�', 'ó').replaceAll('_', ' ').trim();
const title = (value = '') => clean(value).replace(/\b\w/g, c => c.toUpperCase());
const clamp = (n, min = 0, max = 1) => Math.max(min, Math.min(max, n));
const avg = arr => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

const state = {
  data: null, model: null, ranked: [], shown: 6, selectedCuisines: new Set(), compare: new Set(),
  algorithm: 'hybrid',
  weights: { quality: 34, taste: 28, fit: 24, convenience: 14 },
  session: { occasion: 'family', location: 'slp', cuisines: new Set(['Mexican']), price: 'medium', maxDistance: 8, parking: false, access: false, alcohol: false, matchedUser: '' }
};

const algorithmMeta = {
  item: { label: 'Item-Based CF', title: 'Item-Based Collaborative Filtering', summary: 'Ranks restaurants similar to places preferred by the matched diner.', empty: 'Try a wider radius or a profile with more matching rating history.' },
  location: { label: 'Location-Aware', title: 'Location-Aware Context', summary: 'Prioritizes nearby restaurants that still satisfy the dining brief.', empty: 'Try increasing the travel radius or relaxing a required switch.' },
  hybrid: { label: 'Hybrid', title: 'Hybrid Recommendation', summary: 'Combines rating quality, collaborative taste, preference fit, and distance.', empty: 'Try increasing the travel radius or relaxing one of the switches.' }
};

const occasionProfiles = {
  family: { label: 'Family dinner', ambience: 'family', icon: '🍲' },
  date: { label: 'Date night', ambience: 'solitary', icon: '🥂' },
  friends: { label: 'Dinner with friends', ambience: 'friends', icon: '🍻' },
  solo: { label: 'Solo dinner', ambience: 'solitary', icon: '📖' }
};

const diningAreas = {
  slp: { label: 'San Luis Potosí', latitude: 22.15, longitude: -100.98 },
  cuernavaca: { label: 'Cuernavaca', latitude: 18.92, longitude: -99.23 },
  victoria: { label: 'Ciudad Victoria', latitude: 23.75, longitude: -99.16 }
};

function parseCSV(text) {
  const rows = []; let row = [], field = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const char = text[i], next = text[i + 1];
    if (char === '"' && quoted && next === '"') { field += '"'; i++; }
    else if (char === '"') quoted = !quoted;
    else if (char === ',' && !quoted) { row.push(field); field = ''; }
    else if ((char === '\n' || char === '\r') && !quoted) {
      if (char === '\r' && next === '\n') i++;
      row.push(field); field = '';
      if (row.some(v => v !== '')) rows.push(row);
      row = [];
    } else field += char;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  const headers = rows.shift().map(h => h.trim());
  return rows.map(values => Object.fromEntries(headers.map((h, i) => [h, (values[i] || '').trim()])));
}

async function loadCSV(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Could not load ${path}`);
  const buffer = await response.arrayBuffer();
  return parseCSV(new TextDecoder('utf-8').decode(buffer));
}

const group = (rows, key, value) => rows.reduce((map, row) => {
  const id = row[key];
  if (!map.has(id)) map.set(id, []);
  map.get(id).push(value ? row[value] : row);
  return map;
}, new Map());

function haversine(aLat, aLon, bLat, bLon) {
  const R = 6371, rad = d => d * Math.PI / 180;
  const dLat = rad(bLat - aLat), dLon = rad(bLon - aLon);
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(rad(aLat)) * Math.cos(rad(bLat)) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}

function buildModel(raw) {
  const byUserRatings = group(raw.ratings, 'userID');
  const byPlaceRatings = group(raw.ratings, 'placeID');
  const userCuisines = group(raw.userCuisines, 'userID', 'Rcuisine');
  const userPayments = group(raw.userPayments, 'userID', 'Upayment');
  const cuisines = group(raw.cuisines, 'placeID', 'Rcuisine');
  const payments = group(raw.payments, 'placeID', 'Rpayment');
  const parking = group(raw.parking, 'placeID', 'parking_lot');
  const hours = group(raw.hours, 'placeID');
  const profileMap = new Map(raw.profiles.map(x => [x.userID, x]));
  const placeMap = new Map(raw.places.map(x => [x.placeID, x]));
  const global = avg(raw.ratings.map(r => compositeRating(r)));

  const places = raw.places.map(place => {
    const reviews = byPlaceRatings.get(place.placeID) || [];
    const rawQuality = avg(reviews.map(compositeRating));
    const bayesian = (reviews.length * rawQuality + 7 * global) / (reviews.length + 7);
    return {
      ...place, cuisines: cuisines.get(place.placeID) || [], payments: payments.get(place.placeID) || [],
      parking: parking.get(place.placeID) || ['unknown'], hours: hours.get(place.placeID) || [], reviews,
      quality: bayesian / 2, rawQuality, reviewCount: reviews.length,
      foodAvg: avg(reviews.map(r => +r.food_rating)), serviceAvg: avg(reviews.map(r => +r.service_rating))
    };
  });
  return { byUserRatings, byPlaceRatings, userCuisines, userPayments, profileMap, placeMap, places, global };
}

function compositeRating(r) { return .5 * +r.rating + .3 * +r.food_rating + .2 * +r.service_rating; }

function pearson(userA, userB, byUserRatings) {
  const a = new Map((byUserRatings.get(userA) || []).map(r => [r.placeID, compositeRating(r)]));
  const pairs = (byUserRatings.get(userB) || []).filter(r => a.has(r.placeID)).map(r => [a.get(r.placeID), compositeRating(r)]);
  if (pairs.length < 2) return 0;
  const ax = avg(pairs.map(p => p[0])), bx = avg(pairs.map(p => p[1]));
  const numerator = pairs.reduce((s, p) => s + (p[0] - ax) * (p[1] - bx), 0);
  const denom = Math.sqrt(pairs.reduce((s, p) => s + (p[0] - ax) ** 2, 0) * pairs.reduce((s, p) => s + (p[1] - bx) ** 2, 0));
  return denom ? numerator / denom * Math.min(1, pairs.length / 5) : 0;
}

function collaborativeScores(userID) {
  const model = state.model;
  const own = model.byUserRatings.get(userID) || [];
  const ownMean = avg(own.map(compositeRating));
  const similarities = [...model.byUserRatings.keys()]
    .filter(id => id !== userID).map(id => [id, pearson(userID, id, model.byUserRatings)])
    .filter(([, sim]) => sim > .05).sort((a, b) => b[1] - a[1]).slice(0, 25);
  const result = new Map();
  for (const place of model.places) {
    let weighted = 0, total = 0, evidence = 0;
    for (const [other, sim] of similarities) {
      const review = (model.byUserRatings.get(other) || []).find(r => r.placeID === place.placeID);
      if (!review) continue;
      const mean = avg(model.byUserRatings.get(other).map(compositeRating));
      weighted += sim * (compositeRating(review) - mean); total += Math.abs(sim); evidence++;
    }
    const prediction = total ? clamp((ownMean + weighted / total) / 2) : place.quality;
    result.set(place.placeID, { score: prediction, evidence, neighbors: similarities.length });
  }
  return result;
}

function cosineSimilarity(left, right) {
  let dot = 0, leftNorm = 0, rightNorm = 0;
  for (const [key, value] of left) {
    leftNorm += value ** 2;
    if (right.has(key)) dot += value * right.get(key);
  }
  for (const value of right.values()) rightNorm += value ** 2;
  return leftNorm && rightNorm ? dot / Math.sqrt(leftNorm * rightNorm) : 0;
}

function itemBasedScores(userID) {
  const model = state.model;
  const ownRatings = (model.byUserRatings.get(userID) || [])
    .map(r => ({ placeID: r.placeID, rating: compositeRating(r) }))
    .filter(r => r.rating >= 1);
  const placeVectors = new Map(model.places.map(place => [
    place.placeID,
    new Map((model.byPlaceRatings.get(place.placeID) || []).map(r => [r.userID, compositeRating(r)]))
  ]));
  const result = new Map();

  for (const place of model.places) {
    let weighted = 0, total = 0, evidence = 0;
    const candidateVector = placeVectors.get(place.placeID) || new Map();
    for (const rated of ownRatings) {
      if (rated.placeID === place.placeID) continue;
      const sim = cosineSimilarity(candidateVector, placeVectors.get(rated.placeID) || new Map());
      if (sim <= .05) continue;
      weighted += sim * rated.rating;
      total += sim;
      evidence++;
    }
    result.set(place.placeID, { score: total ? clamp((weighted / total) / 2) : place.quality, evidence });
  }
  return result;
}

function currentBrief() {
  return {
    userID: $('#userSelect').value, price: $('#priceSelect').value, payment: $('#paymentSelect').value,
    maxDistance: +$('#distanceRange').value, parking: $('#parkingToggle').checked,
    access: $('#accessToggle').checked, alcohol: $('#alcoholToggle').checked,
    cuisines: new Set(state.selectedCuisines)
  };
}

function activeProfile() {
  const source = state.model.profileMap.get(state.session.matchedUser);
  const area = diningAreas[state.session.location];
  return { ...source, latitude: area.latitude, longitude: area.longitude, ambience: occasionProfiles[state.session.occasion].ambience, budget: state.session.price };
}

function preferenceScore(place, profile, brief) {
  const signals = [], add = (score, weight, label) => signals.push({ score, weight, label });
  if (brief.cuisines.size) {
    const matches = place.cuisines.filter(c => brief.cuisines.has(c));
    add(matches.length ? 1 : .15, 4, matches.length ? `${title(matches[0])} matches your taste` : 'Cuisine differs from your brief');
  }
  if (brief.price !== 'any') add(place.price === brief.price ? 1 : ({low: 1, medium: 2, high: 3}[place.price] < {low: 1, medium: 2, high: 3}[brief.price] ? .7 : .18), 3, place.price === brief.price ? 'Right on budget' : 'Budget tradeoff');
  if (brief.payment !== 'any') add(place.payments.some(x => normalizePayment(x) === normalizePayment(brief.payment)) ? 1 : 0, 2, 'Preferred payment');
  if (profile?.ambience && profile.ambience !== '?') add((profile.ambience === 'family' && place.Rambience === 'familiar') || (profile.ambience === 'solitary' && place.Rambience === 'quiet') ? 1 : .55, 1.5, 'Ambience fit');
  if (profile?.dress_preference && !['?', 'no preference'].includes(profile.dress_preference)) add(place.dress_code === profile.dress_preference ? 1 : place.dress_code === 'informal' ? .65 : .4, 1, 'Dress preference');
  return { score: signals.length ? signals.reduce((s, x) => s + x.score * x.weight, 0) / signals.reduce((s, x) => s + x.weight, 0) : .65, signals };
}

function normalizePayment(x) { return x.toLowerCase().replace('visa', 'visa').replaceAll('_', '-'); }

function scoreRestaurants() {
  const brief = currentBrief(), model = state.model, profile = activeProfile();
  const userCollab = collaborativeScores(brief.userID);
  const itemCollab = itemBasedScores(brief.userID);
  const seen = new Set((model.byUserRatings.get(brief.userID) || []).map(r => r.placeID));
  const totalWeight = Object.values(state.weights).reduce((a, b) => a + b, 0);
  const w = Object.fromEntries(Object.entries(state.weights).map(([k, v]) => [k, v / totalWeight]));

  state.ranked = model.places.map(place => {
    const distance = haversine(+profile.latitude, +profile.longitude, +place.latitude, +place.longitude);
    const parkingAvailable = !place.parking.some(x => x === 'none' || x === 'unknown');
    const accessAvailable = place.accessibility !== 'no_accessibility';
    const alcoholAvailable = place.alcohol !== 'No_Alcohol_Served';
    const hardPass = distance <= brief.maxDistance && (!brief.parking || parkingAvailable) && (!brief.access || accessAvailable) && (!brief.alcohol || alcoholAvailable);
    const fit = preferenceScore(place, profile, brief);
    const convenience = clamp(1 - distance / Math.max(brief.maxDistance * 1.15, 2));
    const itemTaste = itemCollab.get(place.placeID);
    const userTaste = userCollab.get(place.placeID);
    let score, reason, evidence;
    if (state.algorithm === 'item') {
      score = 100 * (itemTaste.score * .72 + place.quality * .18 + fit.score * .10);
      reason = itemTaste.evidence ? `${itemTaste.evidence} similar restaurant signals` : 'Fallback to quality where item history is sparse';
      evidence = itemTaste.evidence;
    } else if (state.algorithm === 'location') {
      score = 100 * (convenience * .58 + fit.score * .25 + place.quality * .17);
      reason = distance < 2 ? 'Very close to the selected dining area' : `${distance.toFixed(1)} km from the selected dining area`;
      evidence = Math.round(convenience * 10);
    } else {
      const tasteScore = itemTaste.score * .6 + userTaste.score * .4;
      const bestSignal = fit.signals.filter(s => s.score >= .8).sort((a,b) => b.weight - a.weight)[0];
      score = 100 * (place.quality * w.quality + tasteScore * w.taste + fit.score * w.fit + convenience * w.convenience);
      reason = bestSignal?.label;
      evidence = itemTaste.evidence + userTaste.evidence;
    }
    const attributeCoverage = [place.cuisines.length, place.payments.length, place.hours.length, place.parking[0] !== 'unknown'].filter(Boolean).length / 4;
    const confidence = clamp(.35 + Math.min(place.reviewCount, 18) / 36 + attributeCoverage * .15 + Math.min(evidence, 6) * .025);
    return { ...place, distance, hardPass, fit: fit.score, taste: itemTaste.score, collabEvidence: evidence, score, confidence, seen: seen.has(place.placeID), reason: reason || (place.quality > .72 ? 'Consistently strong diner ratings' : distance < 2 ? 'A convenient nearby option' : 'Balanced across your priorities') };
  }).filter(p => p.hardPass).sort((a, b) => b.score - a.score);
  const eligibleIDs = new Set(state.ranked.map(p => p.placeID));
  state.compare = new Set([...state.compare].filter(id => eligibleIDs.has(id)));
  state.shown = 6;
  renderAll();
}

function populateControls() {
  const model = state.model;
  const topCuisines = [...new Set(model.places.flatMap(p => p.cuisines))]
    .map(c => [c, model.places.filter(p => p.cuisines.includes(c)).length]).sort((a,b) => b[1] - a[1]).slice(0, 16).map(x => x[0]);
  $('#cuisineChips').innerHTML = topCuisines.map(c => `<button type="button" class="chip" data-cuisine="${c}">${title(c)}</button>`).join('');
  $('#onboardingCuisines').innerHTML = topCuisines.slice(0, 12).map(c => `<button type="button" class="onboarding-cuisine ${state.session.cuisines.has(c) ? 'selected' : ''}" data-onboarding-cuisine="${c}">${title(c)}</button>`).join('');
  $('#startButton').disabled = false;
  $('#startButton span').textContent = 'Build my dinner shortlist';
}

function matchAnonymousProfile() {
  const targetAmbience = occasionProfiles[state.session.occasion].ambience;
  const candidates = [...state.model.profileMap.entries()].map(([id, profile]) => {
    const history = state.model.byUserRatings.get(id)?.length || 0;
    const tastes = new Set(state.model.userCuisines.get(id) || []);
    const cuisineMatches = [...state.session.cuisines].filter(c => tastes.has(c)).length;
    let score = cuisineMatches * 7 + (profile.budget === state.session.price ? 5 : 0) + (profile.ambience === targetAmbience ? 4 : 0);
    if (state.session.parking && profile.transport === 'car owner') score += 3;
    if (state.session.alcohol && profile.drink_level !== 'abstemious') score += 2;
    const area = diningAreas[state.session.location];
    const locationDistance = haversine(area.latitude, area.longitude, +profile.latitude, +profile.longitude);
    score += Math.max(0, 3 - locationDistance / 5);
    score += Math.min(history, 15) / 15;
    return { id, score, history };
  }).sort((a, b) => b.score - a.score || b.history - a.history);
  state.session.matchedUser = candidates[0].id;
  $('#userSelect').value = candidates[0].id;
}

function applySessionToControls() {
  state.selectedCuisines = new Set(state.session.cuisines);
  $('#priceSelect').value = state.session.price;
  $('#paymentSelect').value = 'any';
  $('#distanceRange').value = state.session.maxDistance;
  $('#distanceOutput').value = `${state.session.maxDistance} km`;
  $('#parkingToggle').checked = state.session.parking;
  $('#accessToggle').checked = state.session.access;
  $('#alcoholToggle').checked = state.session.alcohol;
  updateProfileUI(); updateCuisineUI();
}

function updateProfileUI() {
  const occasion = occasionProfiles[state.session.occasion];
  $('#avatar').textContent = occasion.icon;
  $('#profileName').textContent = occasion.label;
  const cuisines = [...state.selectedCuisines].slice(0, 2).map(title).join(' + ') || 'Open to any cuisine';
  $('#profileMeta').textContent = `${cuisines} · ${diningAreas[state.session.location].label}`;
}

function updateCuisineUI() { $$('.chip').forEach(chip => chip.classList.toggle('selected', state.selectedCuisines.has(chip.dataset.cuisine))); }

const palettes = [['#f1c998','#d66f54'], ['#b9dec4','#5fa186'], ['#e9d879','#e79c4c'], ['#c7c5e8','#7b76ae'], ['#f0b6a6','#cc695d'], ['#c5dfdf','#609a9a']];
const cuisineIcons = { Mexican:'🌮', Pizzeria:'🍕', Italian:'🍝', Japanese:'🍣', Seafood:'🐟', Burgers:'🍔', Cafeteria:'☕', Bar:'🍸', Chinese:'🥢', 'Fast_Food':'🍟', American:'🥪', Bakery:'🥐' };

function renderResults() {
  const query = $('#searchInput').value.toLowerCase().trim(), sort = $('#sortSelect').value;
  const method = algorithmMeta[state.algorithm];
  let results = state.ranked.filter(p => !query || `${p.name} ${p.city} ${p.cuisines.join(' ')}`.toLowerCase().includes(query));
  if (sort === 'rating') results.sort((a,b) => b.quality - a.quality);
  if (sort === 'distance') results.sort((a,b) => a.distance - b.distance);
  if (sort === 'confidence') results.sort((a,b) => b.confidence - a.confidence);
  const shown = results.slice(0, state.shown);
  $('#loadingState').hidden = true;
  $('#resultsGrid').innerHTML = shown.length ? shown.map((p, index) => restaurantCard(p, index)).join('') : `<div class="empty-state"><strong>No venues fit every constraint.</strong><p>${method.empty}</p></div>`;
  $('#showMoreBtn').hidden = results.length <= state.shown;
  $('#resultSummary').textContent = `${results.length} eligible venues · ${method.label} ranking`;
}

function restaurantCard(p, index) {
  const palette = palettes[index % palettes.length], cuisine = p.cuisines[0] || 'Restaurant';
  const icon = cuisineIcons[cuisine] || (p.alcohol === 'Full_Bar' ? '🍹' : '🍽️');
  const stars = '★'.repeat(Math.round(p.rawQuality / 2 * 5)) + '☆'.repeat(5 - Math.round(p.rawQuality / 2 * 5));
  const tags = [p.cuisines[0] && title(p.cuisines[0]), priceLabel(p.price), p.parking[0] !== 'none' && p.parking[0] !== 'unknown' && 'Parking'].filter(Boolean);
  return `<article class="restaurant-card">
    <div class="card-visual" style="--c1:${palette[0]};--c2:${palette[1]}">
      <span class="rank-badge">#${index + 1} match</span><span class="food-icon" aria-hidden="true">${icon}</span>
      <button type="button" class="compare-check ${state.compare.has(p.placeID) ? 'selected' : ''}" data-compare="${p.placeID}">${state.compare.has(p.placeID) ? '✓ Added' : '+ Compare'}</button>
    </div>
    <div class="card-body">
      <div class="card-topline"><div style="min-width:0"><h3 title="${p.name}">${clean(p.name)}</h3><p class="location">${clean(p.city) || clean(p.state) || 'Mexico'} · ${p.distance.toFixed(1)} km away</p></div><div class="score"><strong>${Math.round(p.score)}</strong><span>match</span></div></div>
      <div class="tags">${tags.map((t,i) => `<span class="tag ${i === 0 && state.selectedCuisines.has(p.cuisines[0]) ? 'match' : ''}">${t}</span>`).join('')}</div>
      <div class="rating-row"><span class="stars" aria-label="${p.rawQuality.toFixed(1)} out of 2">${stars}</span><strong>${p.rawQuality.toFixed(1)}/2</strong><span>${p.reviewCount} ratings · ${Math.round(p.confidence*100)}% confidence</span></div>
      <p class="reason"><b>✦</b><span>${p.reason}${p.seen ? ' · You have rated this venue' : ''}</span></p>
    </div>
  </article>`;
}

function priceLabel(price) { return ({ low: '$ · Low', medium: '$$ · Medium', high: '$$$ · High' })[price] || title(price); }

function renderDecision() {
  const method = algorithmMeta[state.algorithm];
  const top = state.ranked[0];
  if (!top) { $('#decisionTitle').textContent = 'No clear recommendation'; $('#decisionText').textContent = 'Relax a constraint to restore the shortlist.'; return; }
  const second = state.ranked[1], gap = second ? top.score - second.score : 100;
  $('#decisionTitle').textContent = gap < 3 ? `${method.label}: a close call` : `${clean(top.name)} leads with ${method.label}`;
  $('#decisionText').textContent = gap < 3 ? `Only ${gap.toFixed(1)} points separate the top two; compare them before deciding.` : `${Math.round(top.score)}% fit, driven by ${top.reason.toLowerCase()}.`;
}

function updateMethodUI() {
  const method = algorithmMeta[state.algorithm];
  $('#algorithmSummary').textContent = method.summary;
  $('#methodTitle').textContent = method.title;
  $('#tuneBtn').disabled = state.algorithm !== 'hybrid';
  $('#tuneBtn').classList.toggle('disabled', state.algorithm !== 'hybrid');
  if (state.algorithm !== 'hybrid') {
    $('#priorityPanel').hidden = true;
    $('#tuneBtn span').textContent = '＋';
  }
}

function renderInsights() {
  renderMap(); renderCuisineChart();
  const top = state.ranked.slice(0, 6), confidence = Math.round(avg(top.map(p => p.confidence)) * 100) || 0;
  $('#confidenceRing').style.setProperty('--pct', confidence);
  $('#confidenceRing strong').textContent = `${confidence}%`;
  $('#confidenceText').textContent = confidence >= 75 ? 'Strong evidence supports this shortlist. The top venues have broad profile and rating coverage.' : 'Treat close scores as directional: some venues have limited ratings or attribute coverage.';
  const attr = top.length ? Math.round(avg(top.map(p => [p.cuisines.length, p.payments.length, p.hours.length, p.parking[0] !== 'unknown'].filter(Boolean).length / 4)) * 100) : 0;
  $('#coverageList').innerHTML = `<div><span>Attribute coverage</span><strong>${attr}%</strong></div><div><span>Median reviews</span><strong>${top.length ? Math.round([...top].sort((a,b)=>a.reviewCount-b.reviewCount)[Math.floor(top.length/2)].reviewCount) : 0}</strong></div><div><span>Similar-diner evidence</span><strong>${top.reduce((s,p)=>s+p.collabEvidence,0)} signals</strong></div>`;
}

function renderMap() {
  const map = $('#miniMap'), top = state.ranked.slice(0, 8), p = activeProfile();
  const points = [{ latitude:+p.latitude, longitude:+p.longitude, name:'Dining area', user:true }, ...top];
  if (!top.length) { map.innerHTML = ''; return; }
  const lats = points.map(x => +x.latitude), lons = points.map(x => +x.longitude), minLat=Math.min(...lats), maxLat=Math.max(...lats), minLon=Math.min(...lons), maxLon=Math.max(...lons);
  const pos = x => ({ left: 7 + 86 * ((+x.longitude-minLon)/(maxLon-minLon||1)), top: 8 + 84 * (1-(+x.latitude-minLat)/(maxLat-minLat||1)) });
  map.innerHTML = `<span class="map-road" style="width:130%;left:-10%;top:45%;transform:rotate(-8deg)"></span><span class="map-road" style="width:110%;left:5%;top:70%;transform:rotate(17deg)"></span>` + points.map(x => { const v=pos(x); return `<span class="map-point ${x.user?'user':''}" data-label="${x.user?'Selected dining area':clean(x.name)}" style="left:${v.left}%;top:${v.top}%"></span>`; }).join('');
}

function renderCuisineChart() {
  const groups = new Map();
  for (const p of state.model.places) for (const c of p.cuisines) {
    if (!groups.has(c)) groups.set(c, []); groups.get(c).push(p);
  }
  const rows = [...groups].filter(([,ps])=>ps.length>=3).map(([c,ps])=>({ c, score:avg(ps.map(p=>p.quality)), n:ps.length })).sort((a,b)=>b.score-a.score).slice(0,7);
  $('#cuisineChart').innerHTML = rows.map(r=>`<div class="bar-item"><span title="${title(r.c)}">${title(r.c)}</span><div class="bar-track"><div class="bar-fill" style="width:${r.score*100}%"></div></div><span class="bar-value">${(r.score*2).toFixed(1)}</span></div>`).join('');
  $('#cuisineScope').textContent = `${groups.size} cuisines`;
}

function renderAll() { renderResults(); renderDecision(); renderInsights(); updateCompareDock(); }

function toggleCompare(id) {
  if (state.compare.has(id)) state.compare.delete(id);
  else if (state.compare.size < 3) state.compare.add(id);
  else return toast('Compare up to three restaurants at a time.');
  renderResults(); updateCompareDock();
}

function updateCompareDock() {
  const places = [...state.compare].map(id => state.model.places.find(p => p.placeID === id)).filter(Boolean);
  $('#compareDock').hidden = !places.length;
  $('#compareCount').textContent = places.length;
  $('#compareNames').innerHTML = places.map(p => `<span>${clean(p.name)}</span>`).join('<span>·</span>');
}

function openComparison() {
  const items = [...state.compare].map(id => state.ranked.find(p=>p.placeID===id) || state.model.places.find(p=>p.placeID===id)).filter(Boolean);
  if (items.length < 2) return toast('Add at least two restaurants to compare.');
  const best = (key, inverse=false) => { const vals=items.map(x=>x[key]); return inverse?Math.min(...vals):Math.max(...vals); };
  const row = (label, fn, key, inverse=false) => `<tr><td>${label}</td>${items.map(x=>`<td class="${x[key]===best(key,inverse)?'winner':''}">${fn(x)}</td>`).join('')}</tr>`;
  $('#compareTable').innerHTML = `<table class="compare-table"><thead><tr><th>Decision factor</th>${items.map(x=>`<th>${clean(x.name)}</th>`).join('')}</tr></thead><tbody>
    ${row('Match score',x=>`${Math.round(x.score)}%`,'score')}${row('Rating',x=>`${x.rawQuality.toFixed(1)}/2`,'rawQuality')}${row('Evidence',x=>`${x.reviewCount} ratings`,'reviewCount')}${row('Distance',x=>`${x.distance.toFixed(1)} km`,'distance',true)}
    <tr><td>Cuisine</td>${items.map(x=>`<td>${x.cuisines.map(title).join(', ')||'Not listed'}</td>`).join('')}</tr><tr><td>Price</td>${items.map(x=>`<td>${priceLabel(x.price)}</td>`).join('')}</tr><tr><td>Parking</td>${items.map(x=>`<td>${x.parking.map(title).join(', ')}</td>`).join('')}</tr><tr><td>Payment</td>${items.map(x=>`<td>${x.payments.slice(0,3).map(title).join(', ')||'Not listed'}</td>`).join('')}</tr>
  </tbody></table>`;
  $('#compareDialog').showModal();
}

function toast(message) { const el=$('#toast'); el.textContent=message; el.classList.add('show'); setTimeout(()=>el.classList.remove('show'),2400); }

function syncOnboardingUI() {
  $$('.occasion-option').forEach(button => button.classList.toggle('selected', button.dataset.occasion === state.session.occasion));
  $$('.onboarding-cuisine').forEach(button => button.classList.toggle('selected', state.session.cuisines.has(button.dataset.onboardingCuisine)));
  $$('#onboardingBudget button').forEach(button => button.classList.toggle('selected', button.dataset.budget === state.session.price));
  $$('#onboardingLocation button').forEach(button => button.classList.toggle('selected', button.dataset.location === state.session.location));
  $('#onboardingDistance').value = state.session.maxDistance;
  $('#onboardingDistanceOut').value = `${state.session.maxDistance} km`;
  $('#onboardingParking').checked = state.session.parking;
  $('#onboardingAccess').checked = state.session.access;
  $('#onboardingAlcohol').checked = state.session.alcohol;
}

function captureOnboarding() {
  state.session.occasion = $('.occasion-option.selected')?.dataset.occasion || 'family';
  state.session.cuisines = new Set($$('.onboarding-cuisine.selected').map(button => button.dataset.onboardingCuisine));
  state.session.price = $('#onboardingBudget button.selected')?.dataset.budget || 'medium';
  state.session.location = $('#onboardingLocation button.selected')?.dataset.location || 'slp';
  state.session.maxDistance = +$('#onboardingDistance').value;
  state.session.parking = $('#onboardingParking').checked;
  state.session.access = $('#onboardingAccess').checked;
  state.session.alcohol = $('#onboardingAlcohol').checked;
}

function syncSessionFromMain() {
  state.session.cuisines = new Set(state.selectedCuisines);
  state.session.price = $('#priceSelect').value === 'any' ? 'medium' : $('#priceSelect').value;
  state.session.maxDistance = +$('#distanceRange').value;
  state.session.parking = $('#parkingToggle').checked;
  state.session.access = $('#accessToggle').checked;
  state.session.alcohol = $('#alcoholToggle').checked;
}

function openOnboarding() {
  if (state.session.matchedUser) syncSessionFromMain();
  syncOnboardingUI();
  $('#onboardingPage').classList.remove('hidden');
  document.body.classList.add('onboarding-open');
  $('#onboardingPage').scrollTop = 0;
}

function bindEvents() {
  $('#occasionOptions').addEventListener('click', e => { const button=e.target.closest('[data-occasion]'); if(!button)return; $$('.occasion-option').forEach(x=>x.classList.remove('selected')); button.classList.add('selected'); });
  $('#onboardingCuisines').addEventListener('click', e => { const button=e.target.closest('[data-onboarding-cuisine]'); if(!button)return; button.classList.toggle('selected'); });
  $('#onboardingBudget').addEventListener('click', e => { const button=e.target.closest('[data-budget]'); if(!button)return; $$('#onboardingBudget button').forEach(x=>x.classList.remove('selected')); button.classList.add('selected'); });
  $('#onboardingLocation').addEventListener('click', e => { const button=e.target.closest('[data-location]'); if(!button)return; $$('#onboardingLocation button').forEach(x=>x.classList.remove('selected')); button.classList.add('selected'); });
  $('#onboardingDistance').addEventListener('input', e => $('#onboardingDistanceOut').value = `${e.target.value} km`);
  $('#onboardingForm').addEventListener('submit', e => {
    e.preventDefault(); captureOnboarding(); matchAnonymousProfile(); applySessionToControls();
    $('#onboardingPage').classList.add('hidden'); document.body.classList.remove('onboarding-open');
    scoreRestaurants(); window.scrollTo({ top: 0, behavior: 'instant' });
  });
  $('#editProfile').addEventListener('click', openOnboarding);
  $('#editProfileSide').addEventListener('click', openOnboarding);
  $('#cuisineChips').addEventListener('click', e => { const c=e.target.dataset.cuisine; if(!c)return; state.selectedCuisines.has(c)?state.selectedCuisines.delete(c):state.selectedCuisines.add(c); updateCuisineUI(); updateProfileUI(); });
  $('#clearCuisines').addEventListener('click', () => { state.selectedCuisines.clear(); updateCuisineUI(); updateProfileUI(); });
  $('#distanceRange').addEventListener('input', e => $('#distanceOutput').value = `${e.target.value} km`);
  $('#recommendBtn').addEventListener('click', () => { syncSessionFromMain(); matchAnonymousProfile(); updateProfileUI(); scoreRestaurants(); toast('Recommendations updated.'); });
  $('#algorithmSelect').addEventListener('change', e => { state.algorithm = e.target.value; updateMethodUI(); scoreRestaurants(); toast(`${algorithmMeta[state.algorithm].label} selected.`); });
  $('#searchInput').addEventListener('input', renderResults);
  $('#sortSelect').addEventListener('change', renderResults);
  $('#showMoreBtn').addEventListener('click', () => { state.shown += 6; renderResults(); });
  $('#resultsGrid').addEventListener('click', e => { const id=e.target.dataset.compare; if(id) toggleCompare(id); });
  $('#tuneBtn').addEventListener('click', () => { const panel=$('#priorityPanel'); panel.hidden=!panel.hidden; $('#tuneBtn span').textContent=panel.hidden?'＋':'−'; });
  for (const key of ['quality','taste','fit','convenience']) $(`#${key}Weight`).addEventListener('input', e => { state.weights[key]=+e.target.value; const total=Object.values(state.weights).reduce((a,b)=>a+b,0); for(const k of Object.keys(state.weights)) $(`#${k}WeightOut`).value=`${Math.round(state.weights[k]/total*100)}%`; scoreRestaurants(); });
  $('#resetBtn').addEventListener('click', () => { applySessionToControls(); scoreRestaurants(); });
  $('#clearCompare').addEventListener('click', () => { state.compare.clear(); renderAll(); });
  $('#openCompare').addEventListener('click', openComparison);
  $('#closeCompare').addEventListener('click', () => $('#compareDialog').close());
  $('#compareDialog').addEventListener('click', e => { if(e.target===$('#compareDialog')) $('#compareDialog').close(); });
}

async function init() {
  try {
    const entries = await Promise.all(Object.entries(FILES).map(async ([key,path]) => [key, await loadCSV(path)]));
    state.data = Object.fromEntries(entries); state.model = buildModel(state.data);
    $('#venueCount').textContent = state.data.places.length;
    $('#ratingCount').textContent = state.data.ratings.length.toLocaleString();
    $('#userCount').textContent = state.data.profiles.length;
    populateControls(); bindEvents(); syncOnboardingUI(); updateMethodUI();
  } catch (error) {
    console.error(error); $('#loadingState').innerHTML = `<p>Could not load the CSV data. Start this page through the local dev server.</p>`;
  }
}

init();
