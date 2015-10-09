$(document).ready(function() {

  $('a#urlSubmit').click(function() {

    $('.loading').removeClass('hidden');

    var post_url = "/api/predict";
    var post_data = {
      service: "imageserv",
      parameters: {
        input: {width: 224, height: 224},
	mllib: {gpu: true},
        output: {best: 3}
      },
      data: [$('input#url').val()]
    };

    $.ajax({
      type: "POST",
      url: post_url,
      data: JSON.stringify(post_data),
      dataType: 'json',
      success: function(data) {
        $('#emptyImage img').attr('src', post_data.data[0]);
        $('#emptyImage ul').html('');
        $.each(data.body.predictions.classes, function() {
          var percent = parseInt(this.prob * 100);

          var style = 'success';

          if(percent < 60) {
            style='warning';
          }

          if(percent < 25) {
            style = 'danger';
          }

          var re = /(n\d+) (.*)/;
          var md = this.cat.match(re);
          if(md.length == 3) {
            var nid = md[1];
            var name = md[2];

            var predictionHtml = '<div class="row"><div class="col-lg-4"><div class="progress">';
            predictionHtml += '<div class="progress-bar progress-bar-' + style + '" role="progressbar" ';
            predictionHtml += 'aria-valuenow="' + percent + '" aria-valuemin="0" aria-valuemax="100" ';
            predictionHtml += 'style="width: ' + percent + '%;">' + percent + '%</div></div></div>';
            predictionHtml += '<div class="col-lg-8"><a target="_blank" href="http://www.image-net.org/synset?wnid=' + nid + '">' + name + '</a></div></div>';
            $('#emptyImage .predictions').append(predictionHtml);
          }
        });

        $('#imageList').prepend('<hr>');
        $('#imageList').prepend($('#emptyImage').clone().attr('id', '').removeClass('hidden'));
        $('.loading').addClass('hidden');

        $('#emptyImage .predictions').html('');
        $('input#url').val('');
      },
      error: function(jqXHR, textStatus, errorThrown) {
        $('input#url').val('');
        $('.loading').addClass('hidden');
        $('#submitAlert').removeClass('hidden').find('.error').html(errorThrown);
        window.setTimeout(function() { $("#submitAlert").addClass('hidden'); }, 4000);
      }
    });

  });

  $('input#url').val('');
});
