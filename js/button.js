

document.getElementById('sendRequest').addEventListener('click', function() {
    const value1 = parseFloat(document.getElementById('value1').value);
    const value2 = parseFloat(document.getElementById('value2').value);
    

    if (isNaN(value1) || isNaN(value2)) {
        alert('Please enter valid floating point numbers');
        return;
    }

    document.getElementById('loading').src = 'https://wallpapers.com/images/featured/funny-anime-9vnia7uc4fa7w7x2.jpg';
    fetch('https://animechan.xyz/api/random')
        .then(response => response.json())
        .then(quote => {
            console.log(quote);
            document.getElementById('output').innerHTML = quote.quote;
        });

});

