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
        console.log("Result: " + data);
        $("#query").prop("disabled", false);
      },
    });
  });
});
