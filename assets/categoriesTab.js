// Test if file has loaded
//alert('If you see this alert, then your custom JavaScript script has run!')

// Close modal
$('body').on('click', '#close-modal-comp-issue',function () {
    $('#modal-comp-issue').css('display', 'none');
});

// $('body').on('click', '#close-modal-site',function () {
//     $('#modal-site').css('display', 'none');
// });

// === SEARCH TAB ===
$("body").on('keyup', '#searchrequest', function (e) {
    if (e.keyCode == 13 && $('#search-table-container').css('display') == 'none') {
        $('#search-table-container').fadeIn();
    }
});

//$('body').on('change', $('#search-count-reveal'), function (e) {
//    alert('Hi');
//});