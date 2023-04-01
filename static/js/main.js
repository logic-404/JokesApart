$(document).ready(function () {
  $("#req-form").on("submit", function (event) {
    event.preventDefault(); // prevent the default form submit action
    $("#query").prop("disabled", true);
    $.ajax({
      type: "POST",
      url: "/predict",
      data: $("#query").val(),
      async: true,
      success: function (data) {
        var joke = data.replace(/ {2}/g, "<br>"); // replace double spaces with a newline
        $("#para").html(joke);
        $("#query").prop("disabled", false);
      },
    });
  });
});
