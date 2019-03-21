// IN PROGRESS; NOT CURRENTLY WORKING

$(window).on('hashchange', function(e){
 // Your Code goes here
 console.log('hihihihiiii')
    if (window.location.href.includes('clustering')) {
        console.log('clustering js');
        $('body').append('<img src="https://loading.io/assets/img/loader/msg.gif" id="search-loading"/>')
    }
});