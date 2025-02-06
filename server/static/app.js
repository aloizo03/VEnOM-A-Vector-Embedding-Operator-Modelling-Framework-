$(function(){
    $("#do_something").click(function(){
        var item_name = prompt("enter item name")
        $.get("/add_item?name="+item_name,function(result_data){
            $('#items_holder').append(result_data)
        })
    })
})