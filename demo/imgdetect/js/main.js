var deepdetect = new DeepDetect.Service('http://localhost/api');

$(document).ready(function() {

  $('#service_select').select2();

  var listServices = function() {

    var service_url = "/api/info";

    $.ajax({
      type: "GET",
      url: service_url,
      dataType: 'json',
      success: function(data) {

        if(data.head.services && data.head.services.length > 0) {
          $('#serviceForm').removeClass('bg-info')
            .find('span').html('Select a service: ');

          $('#service_select').select2({
            data: data.head.services.map(function(item) {
              return {id: item.name, text: item.name};
            })
          });
          $('#service_select').show();
          $('#uploadForm').removeClass('hidden');
        } else {
          $('#serviceForm').removeClass('bg-info')
            .addClass('bg-danger')
            .html('No service found, set a service on DeepDetect server.');
        }
      }
    });
  };

  $('a#urlSubmit').click(function() {

    $('.loading').removeClass('hidden');

    var predict_data = {
      service: $('#service_select').val(),
      parameters: {
        mllib: {gpu: true},
        output: {best: 3}
      },
      data: [$('input#url').val()]
    };

    deepdetect.predict.post(predict_data).then(
      function(data) {
        var prediction = data.body.predictions[0];
        $('#emptyImage img').attr('src', prediction.uri);
        $('#emptyImage ul').html('');
        $.each(prediction.classes, function() {
          var percent = parseInt(this.prob * 100);

          var style = 'success';

          if(percent < 60) {
            style='warning';
          }

          if(percent < 25) {
            style = 'danger';
          }

          var predictionHtml = '<div class="row"><div class="col-lg-4">';
          predictionHtml    += '<div class="progress">';
          predictionHtml    += '<div class="progress-bar ';
          predictionHtml    += 'progress-bar-' + style + '" ';
          predictionHtml    += 'role="progressbar" ';
          predictionHtml    += 'aria-valuenow="' + percent + '" ';
          predictionHtml    += 'aria-valuemin="0" ';
          predictionHtml    += 'aria-valuemax="100" ';
          predictionHtml    += 'style="width: ' + percent + '%;">';
          predictionHtml    += percent + '%</div></div></div>';
          predictionHtml    += '<div class="col-lg-8">';
          predictionHtml    += this.cat + '</div></div>';
          $('#emptyImage .predictions').append(predictionHtml);
        });

        $('#imageList').prepend('<hr>');
        $('#imageList').prepend(
          $('#emptyImage').clone().attr('id', '').removeClass('hidden')
        );
        $('.loading').addClass('hidden');

        $('#emptyImage .predictions').html('');
        $('input#url').val('');
      },
      function(jqXHR, textStatus, errorThrown) {
        $('input#url').val('');
        $('.loading').addClass('hidden');
        $('#submitAlert').removeClass('hidden')
          .find('.error')
          .html(errorThrown);
        window.setTimeout(function() {
          $("#submitAlert").addClass('hidden');
        }, 4000);
      }
    );

  });

  $('input#url').val('');

  listServices();
});
