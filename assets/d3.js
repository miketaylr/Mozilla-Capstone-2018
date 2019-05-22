var checkExist = setInterval(function() {
   if (document.querySelectorAll('#div-d3').length) {
      renderSvg(JSON.parse(document.querySelectorAll('#div-d3')[0].dataset.d3));
      clearInterval(checkExist);
   }
}, 100); // check every 100ms

function renderSvg(rootData) {
  var width = 700;
  var svg = d3.select("svg")
      .style("width", width + 'px')
      .style("height", width + 'px');

  var margin = 20,
      diameter = +width,
      g = svg.append("g").attr("transform", "translate(" + diameter / 2 + "," + diameter / 2 + ")");

  var color = d3.scaleLinear()
      .domain([-1, 5])
      .range(["transparent", "red"])
      .interpolate(d3.interpolateHcl);

  var pack = d3.pack()
      .size([diameter - margin, diameter - margin])
      .padding(2);


  root = d3.hierarchy(rootData)
      .sum(function(d) { return d.value; })
      .sort(function(a, b) { return b.value - a.value; });

  var focus = root,
      nodes = pack(root).descendants(),
      view;

  var circle = g.selectAll("circle")
    .data(nodes)
    .enter().append("circle")
      .attr("class", function(d) { return d.parent ? d.children ? "node" : "node node--leaf" : "node node--root"; })
      .style("fill", function(d) { return d.children ? color(d.depth) : null; })
      .on("click", function(d) { if (focus !== d) zoom(d) });

  var text = g.selectAll("text")
    .data(nodes)
    .enter().append("text")
      .attr("class", "label")
      .style("fill-opacity", function(d) { return d.parent === root ? 1 : 0; })
      .style("fill", '#fff')
      .style('pointer-events', 'auto')
      .style('cursor', 'pointer')
      .style("display", function(d) { return d.parent === root ? "inline" : "none"; })
      .text(function(d) { return d.data.name; })
      .on("click", function(d,evt) {
        console.log('yooo', d);
        d3.event.stopPropagation();
        if (d.children)
          return;
        showOverlay(d);
      });


  var node = g.selectAll("circle,text");

  svg
      .style("background", 'transparent')
      .on("click", function() { zoom(root); });

  zoomTo([root.x, root.y, root.r * 2 + margin]);

  function zoom(d) {
    var focus0 = focus; focus = d;

    var transition = d3.transition()
        .duration(d3.event.altKey ? 7500 : 750)
        .tween("zoom", function(d) {
          var i = d3.interpolateZoom(view, [focus.x, focus.y, focus.r * 2 + margin]);
          return function(t) { zoomTo(i(t)); };
        });

    transition.selectAll("text")
      .filter(function(d) { return d.parent === focus || this.style.display === "inline"; })
        .style("fill-opacity", function(d) { return d.parent === focus ? 1 : 0; })
        .on("start", function(d) { if (d.parent === focus) this.style.display = "inline"; })
        .on("end", function(d) { if (d.parent !== focus) this.style.display = "none"; });
  }

  function zoomTo(v) {
    var k = diameter / v[2]; view = v;
    node.attr("transform", function(d) { return "translate(" + (d.x - v[0]) * k + "," + (d.y - v[1]) * k + ")"; });
    circle.attr("r", function(d) { return d.r * k; });
  }
}

function showOverlay(data) {
  const id = data.data.id;
  console.log(id);
  const clusterEl = document.getElementById(id);
  const clusterParent = clusterEl.parentNode;

  const feedback = clusterParent.getElementsByClassName('clustering-feedback')[0].cloneNode(true);
  const summary = clusterParent.getElementsByClassName('clustering-summary-text')[0];
  const topWords = summary.innerHTML.split(' - ')[1];
  const phrases = clusterParent.getElementsByClassName('clustering-top-text')[0];
  const topPhrases = phrases.innerHTML.split(': ')[1];

  const modal = document.createElement('div');
  modal.setAttribute('id', 'overlay');
  modal.setAttribute('class', 'modal');
  modal.style.display = 'block';

  const modalContent = document.createElement('div');
  modalContent.setAttribute('class', 'modal-content modal-content-fixed-height');

  const buttonContainer = document.createElement('div');
  buttonContainer.setAttribute('class', 'close-button-container');

  const button = document.createElement('button');
  button.setAttribute('class', 'close');
  button.innerText = 'Close';
  button.addEventListener('click', closeOverlay)

  buttonContainer.appendChild(button);

  const headerEl = document.createElement('h2');
  headerEl.setAttribute('class', 'modal-title center');
  headerEl.innerText = 'Cluster Details';

  const textContainer = document.createElement('div');
  textContainer.setAttribute('class', 'modal-text-container');

  const topWordsEl = document.createElement('div');
  topWordsEl.setAttribute('class', 'modal-text modal-top-words');
  topWordsEl.innerHTML = '<b>Top words: </b>' + topWords;

  const topPhrasesEl = document.createElement('div');
  topPhrasesEl.setAttribute('class', 'modal-text modal-top-phrases');
  topPhrasesEl.innerHTML = '<b>Top phrases: </b>' + topPhrases;

  const feedbackEl = document.createElement('div');
  feedbackEl.setAttribute('class', 'modal-text modal-feedback');

  const feedbackChildren = feedback.childNodes;
  feedbackChildren.forEach((item) => {
    item.setAttribute('class', 'modal-feedback-item');
  });

  const feedbackTitleEl = document.createElement('div');
  feedbackTitleEl.innerHTML = `<b> ${feedbackChildren.length} Feedback: </b>`;

  feedbackEl.appendChild(feedbackTitleEl);
  feedbackEl.appendChild(feedback);

  textContainer.appendChild(topWordsEl);
  textContainer.appendChild(topPhrasesEl);
  textContainer.appendChild(feedbackEl);

  modalContent.appendChild(buttonContainer);
  modalContent.appendChild(headerEl);
  modalContent.appendChild(textContainer);
  

  modal.appendChild(modalContent)

  document.body.appendChild(modal);
}

function closeOverlay() {
  const overlay = document.getElementById('overlay');
  overlay.remove();
}