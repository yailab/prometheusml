/* ---- collection of utility functions for the UI ---- */

// The maximum number of columns according to the boostrap grid system
var max_no_cols = 12;

  // max number of input items per column
var max_per_col = 7;

// // check if the carousel item is first or last
// function checkitem(carousel_id) {
//   let $this = $('#' + carousel_id);
//   if ($('.carousel-inner .carousel-item:first').hasClass('active')) {
//     console.log("the fist slide");
//     // Hide left arrow
//     $this.children('.carousel-control-prev').hide();
//     // But show right arrow
//     $this.children('.carousel-control-next').show();
//   } else if ($('.carousel-inner .carousel-item:last').hasClass('active')) {
//     console.log("the last slide");
//     // Hide right arrow
//     $this.children('.right.carousel-control').hide();
//     // But show left arrow
//     $this.children('.left.carousel-control').show();
//   } else {
//     console.log("intermediate slide");
//     $this.children('.carousel-control').show();
//   }
// }


// function to create a carousel with pages
function create_carousel_features (features_list, checked=true, disabled=false) {
  // calculate number of columns
  let no_cols = Math.ceil(features_list.length / max_per_col);
  // the maximum number of cols per slide
  const max_no_col_per_slide = 3;
  // calculate the number of slides in the carousel
  const no_slides = Math.ceil( no_cols/ max_no_col_per_slide);
  // distribute the various features in the slides
  const max_no_features_slide  = max_per_col * max_no_col_per_slide;
  let chunked_features_lists = chunk(features_list, max_no_features_slide);

  // create the carousel
  const carouselDiv = document.createElement("div");
  carouselDiv.classList.add("carousel", "mx-auto");  // do not add `slide` class
  carouselDiv.setAttribute("data-interval", "false");  // stop auto sliding
  let carouselDivID = "carouselWithControls";
  carouselDiv.setAttribute("id", carouselDivID);

  // create the inner carousel where we put the inner content
  const carouselInnerDiv = document.createElement("div");
  carouselInnerDiv.classList.add("carousel-inner", "col-11", "mx-auto", "mb-3");

  // create the carousel indicators
  const carouselIndicatorList = document.createElement("ol");
  carouselIndicatorList.classList.add("carousel-indicators");

  // create the carousel items/slides
  for (let i = 0; i < no_slides; i++) {
    const carouselItem = document.createElement("div");
    carouselItem.classList.add("carousel-item");
    if (i === 0) {
      // the first one always need to be active to display
      carouselItem.classList.add("active");
    }

    // get the feature list for each slide
    let features_list = chunked_features_lists[i]

    // create the radio items for the feature card
    let feature_checkbox_list = [];
    for (let feature in features_list) {
      let checkbox_id = features_list[feature].concat('-id')
      let checkboxDiv = create_checkbox_item(checkbox_id, features_list[feature], checked, disabled)

      // append them to the list
      feature_checkbox_list.push(checkboxDiv);
    }

    // create the grid (row and cols) per slide
    let grid_row_id = "features-checkbox-row-" + i.toString();
    let grid_div = create_checkbox_grid_div(features_list.length, grid_row_id);

    // add the grid div to the slide
    carouselItem.appendChild(grid_div);

    // populate the generated grid div with input items
    populate_grid_div(feature_checkbox_list, grid_div);

    // create the carousel indicators
    const indicatorItem = document.createElement("li");
    indicatorItem.setAttribute("data-target", "#" + carouselDivID);
    indicatorItem.setAttribute("data-slide-to", i.toString());
    if (i === 0) {
      // the first one always need to be active to display
      indicatorItem.classList.add("active");
    }

    // append the carousel item element to inner div
    carouselInnerDiv.appendChild(carouselItem);

    // append the carousel indicators to the list
    carouselIndicatorList.appendChild(indicatorItem);

  }

  // create carousel controls
  const carouseControlPrev = document.createElement("a");
  carouseControlPrev.classList.add("carousel-control-prev");
  carouseControlPrev.setAttribute("href", "#" + carouselDivID);
  carouseControlPrev.setAttribute("role", "button");
  carouseControlPrev.setAttribute("data-slide", "prev");

  const carouseControlPrevIcon = document.createElement("span");   // create the icon
  carouseControlPrevIcon.classList.add("carousel-control-prev-icon");
  carouseControlPrev.appendChild(carouseControlPrevIcon);  // append it to the previous element

  const carouseControlNext = document.createElement("a");
  carouseControlNext.classList.add("carousel-control-next");
  carouseControlNext.setAttribute("href", "#" + carouselDivID);
  carouseControlNext.setAttribute("role", "button");
  carouseControlNext.setAttribute("data-slide", "next");

  const carouseControlNextIcon = document.createElement("span");   // create the icon
  carouseControlNextIcon.classList.add("carousel-control-next-icon");
  carouseControlNext.appendChild(carouseControlNextIcon);  // append it to the previous element

  // append the controls to the carousel div
  if (no_slides > 1) {
    carouselDiv.appendChild(carouseControlPrev);
    carouselDiv.appendChild(carouseControlNext);
  }

  // append the indicators to the carousel div
  carouselDiv.appendChild(carouselIndicatorList);

  // append the carousel-inner div to the carousel div
  carouselDiv.appendChild(carouselInnerDiv);

  // return as an array
  return [carouselDiv, carouselDivID];
}

// function to create a range array --> equivalent to list(range(start, stop)) in Python
const range = (start, end, length = end - start) =>
  Array.from({ length }, (_, i) => start + i)

// function to split an array
const chunk = (arr, size) => arr.reduce((acc, e, i) =>
    (i % size ? acc[acc.length - 1].push(e) : acc.push([e]), acc), []);


// function to allocate the items of a list into groups of maximum space
function group_items (item_list, max_items_per_group) {
  // calculate the number of groups
  let no_groups = Math.ceil(item_list.length / max_items_per_group)

  // placeholder array
  let main_list = [];
  // iterate over the groups and calculate the indexes of each array
  for (let i = 0; i < no_groups; i++) {
    let start_index = max_items_per_group * i
    let end_index = Math.min(max_items_per_group * (i + 1), item_list.length)

    // create a list with the indexes
    let index_list = range(start_index, end_index);

    // concatenate it to the main array
    main_list.push(index_list);
  }

  // return the array of arrays
  return main_list;
}

// function to create range sliders
function create_range_slider (id, label,
                              align="center",
                              description = "test",
                              desc_placement = "bottom") {
  // Create the div to return them
  const rtnDiv = document.createElement("div");
  const alignment_class = "justify-content-" + align;
  rtnDiv.classList.add("row", "mb-3", alignment_class);

  // Create the col divs for label and input slider
  const labelColDiv = document.createElement("div");
  labelColDiv.classList.add("col-md-4", "text-left");
  const inputColDiv = document.createElement("div");
  inputColDiv.classList.add("col-md-4");

  // Create the label element
  const input_label = document.createElement("label");
  input_label.setAttribute("for", id);
  input_label.textContent = 'Select ' + label + ': ';

  // add the popover attributes to label
  input_label.setAttribute("data-trigger", "hover");
  input_label.setAttribute("data-content", description);
  input_label.setAttribute("data-placement", desc_placement);

  // Create the input element
  let slider = document.createElement("input");
  slider.type = "text";
  slider.setAttribute("id", id);
  slider.classList.add("js-range-slider")

  // Append the input element to column div
  inputColDiv.appendChild(slider);

  // Append the label element to the column div
  labelColDiv.appendChild(input_label);

  // Append the col div to the return div
  rtnDiv.appendChild(labelColDiv);
  rtnDiv.appendChild(inputColDiv);

  return rtnDiv;
}

// function to create new sliders
function create_slider (id, min, max, step, label) {
  // create a new div element
  const newDiv = document.createElement("div");

  // add an input element (content)
  let slider = document.createElement("input");
  slider.type = "range";
  slider.classList.add("custom-range")
  slider.setAttribute("id", id);
  slider.setAttribute("min", min);
  slider.setAttribute("max", max);
  slider.setAttribute("step", step);


  // Add a span
  let slider_value = document.createElement("span");

  // Add the label element
  const input_label = document.createElement("label");
  input_label.setAttribute("for", id);
  input_label.textContent = 'Select ' + label + ': ';
  input_label.appendChild(slider_value);

  slider_value.innerHTML = slider.value; // Display the default slider value

  // Update the current slider value (each time you drag the slider handle)
  slider.oninput = function() {
    slider_value.innerHTML = this.value;
    slider_value.style.color = '#ff7F2a';
  }

  // add the elements to the created div
  newDiv.appendChild(input_label);
  newDiv.appendChild(slider);
  // newDiv.appendChild(slider_value);

  return newDiv;
}

// function to create new radio items
function create_radio_item (id, label, name,
                            checked = false,
                            disabled = true,
                            description = "",
                            desc_placement = "top",
                            data_html = "false",
                            href_info = "",
                            class_name= "test") {
  // create a new div element
  const newDiv = document.createElement("div");
  // add the classes to the div element
  newDiv.classList.add("custom-control", "custom-radio");

  // add an input element (content)
  const input = document.createElement("input");
  input.type = "radio";
  input.classList.add("custom-control-input", class_name);
  input.id = id;
  input.name = name;
  input.checked = checked;
  input.disabled = disabled;

  // Add the label element
  const input_label = document.createElement("label");
  input_label.classList.add("custom-control-label", class_name);
  input_label.setAttribute("for", id);
  input_label.textContent = label;

  let pop_desc =""
  if (data_html === "true") {
    const wiki_elem = ". <a href='" + href_info + "' target='_blank' rel='noopener noreferrer'>wiki</a>"
    pop_desc = description + wiki_elem
  } else {
    pop_desc = description
  }

  // add the popover attributes to label
  input_label.setAttribute("data-trigger", "hover");
  input_label.setAttribute("data-html", data_html);
  input_label.setAttribute("data-content", pop_desc);
  input_label.setAttribute("data-placement", desc_placement);

  // add the elements to the created div
  newDiv.appendChild(input);
  newDiv.appendChild(input_label);

  return newDiv;
}


// function to create new checkbox items
function create_checkbox_item (id, label,
                               checked = true,
                               disabled = false,
                               class_name= "test",
                               description = "",
                               desc_placement = "top") {
  // create a new div element
  const newDiv = document.createElement("div");
  // add the classes to the div element
  newDiv.classList.add("custom-control", "custom-checkbox");

  // add an input element (content)
  const input = document.createElement("input");
  input.type = "checkbox";
  input.classList.add("custom-control-input", class_name)
  input.setAttribute("id", id);
  input.checked = checked;
  input.disabled = disabled;

  // Add the label element
  const input_label = document.createElement("label");
  input_label.classList.add("custom-control-label");
  input_label.setAttribute("for", id);
  input_label.textContent = label;

  // add the popover attributes to label
  input_label.setAttribute("data-trigger", "hover");
  input_label.setAttribute("data-content", description);
  input_label.setAttribute("data-placement", desc_placement);

  // add the elements to the created div
  newDiv.appendChild(input);
  newDiv.appendChild(input_label);

  return newDiv;
}


// function to create the grid of columns for the input items
function create_checkbox_grid_div (no_input_items, row_div_id) {
  // calculate number of columns
  let no_cols = Math.ceil(no_input_items / max_per_col);

  // throw an error if it has reached the maximum number of columns allowed
  if (no_cols > max_no_cols) {
    // show only the maximum number of columns
    throw 'Reached maximum number of columns!';
  }


  // add the row div -- function-scoped
  let rowDiv = document.createElement("div");
  rowDiv.classList.add("row", "px-3", "py-0");
  // add a specific id to the div
  rowDiv.setAttribute("id", row_div_id);


  // calculate the bootstrap class of each col (all of them are equal)
  let col_class_no = max_no_cols/no_cols;
  let col_class = "col-".concat(col_class_no.toString());

  // iterate over the col divs
  for (let i = 0; i < no_cols; i++) {
    // create a new column div
    const colDiv = document.createElement("div");
    colDiv.classList.add(col_class, "text-truncate");

    // append it to the row div
    rowDiv.appendChild(colDiv);
  }

  return rowDiv;
}


// function to populate the grid div with input items
function populate_grid_div (input_items_list, rowDiv) {
  // get the child column elements of the row div
  let childrenCols = rowDiv.children;
  // get the number of columns
  let no_cols = rowDiv.children.length;

  // get the input items indexes per column index
  let col_item_index = group_items(input_items_list, max_per_col)

  // iterate over indices of the cols
  for (let col_index = 0; col_index < no_cols; col_index++) {
    // iterate over the indices of the items
    for (let item_index of col_item_index[col_index]) {
      // append the items from the list each col
      childrenCols[col_index].appendChild(input_items_list[item_index]);
    }
  }
}

// keep track of running intervals
let intervals = [];

// function for incremental update of the progressbar
function update_bar(value, stop = false) {
  let i = 0;
  // This block will be executed 100 times.
  let interval = setInterval(function() {
    if (i === value+1 || stop) clearInterval(this);
    else set_bar(i++);
  }, 200);
  intervals.push(interval);
} // End


// function to set the progressbar at a specific value
function set_bar(value) {
  // progressbar.setAttribute("aria-valuenow", value);
  progressbar.css({ "width": value + '%', "background-color": "#ff7F2a" });  // the level of the progressbar
  progressbar.text(value + '%');   // the label of the progressbar
}

function set_bar_stop(value) {
  // progressbar.setAttribute("aria-valuenow", value);
  progressbar.css({ "width": value + '%', "background-color": "#ff7F2a" });  // the level of the progressbar
  progressbar.text(value + '%');   // the label of the progressbar
  intervals.forEach(clearInterval);  // stop running interval for progress bar
}
