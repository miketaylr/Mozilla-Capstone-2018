$("body").on('keyup', '#searchrequest', function (e) {
    if (e.keyCode == 13) {
        $('#search-loading').css('display', 'block');
    }
});