document.addEventListener('DOMContentLoaded', function () {
    // Initialize the map with a default view
    const map = L.map('map').setView([31.5497, 74.3436], 10); // Default to Lahore
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 18,
    }).addTo(map);

    // Custom icons
    const redIcon = L.icon({
        iconUrl: 'https://cdn.rawgit.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
        shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
        iconSize: [25, 41],
        iconAnchor: [12, 41],
        popupAnchor: [1, -34],
        shadowSize: [41, 41]
    });

    const blueIcon = L.icon({
        iconUrl: 'https://cdn.rawgit.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-blue.png',
        shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
        iconSize: [25, 41],
        iconAnchor: [12, 41],
        popupAnchor: [1, -34],
        shadowSize: [41, 41]
    });

    // Function to fetch transactions for a city
    function fetchTransactions(city) {
        fetch('http://127.0.0.1:5000/get_city_transactions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ city: city }),
        })
            .then((response) => response.json())
            .then((data) => {
                if (data.transactions) {
                    updateTable(data.transactions);
                    updateMap(data.transactions);
                    if (data.transactions.length > 0) {
                        const firstTransaction = data.transactions[0];
                        map.setView([firstTransaction.Latitude, firstTransaction.Longitude], 12);
                    }
                } else {
                    alert(data.message || "No transactions found.");
                }
            })
            .catch((error) => console.error('Error:', error));
    }

    // Function to update the table with transaction data
    function updateTable(transactions) {
        const tableBody = document.querySelector('#dataTable tbody');
        tableBody.innerHTML = ''; // Clear existing rows
        transactions.forEach((tx) => {
            const row = `<tr>
                <td>${tx.Transaction_Date}</td>
                <td>${tx.City}</td>
                <td>${tx.Card_Type}</td>
                <td>${tx.Merchant_Type}</td>
                <td>$${tx.Amount.toFixed(2)}</td>
                <td>${tx.Is_Fraudulent ? '✅ Yes' : '❌ No'}</td>
            </tr>`;
            tableBody.innerHTML += row;
        });
    }

    // Function to update the map with transaction markers
    function updateMap(transactions) {
        map.eachLayer((layer) => {
            if (layer instanceof L.Marker) {
                map.removeLayer(layer); // Clear existing markers
            }
        });

        transactions.forEach((tx) => {
            const marker = L.marker([tx.Latitude, tx.Longitude], {
                icon: tx.Is_Fraudulent ? redIcon : blueIcon
            }).addTo(map)
                .bindPopup(`<b>Amount:</b> $${tx.Amount.toFixed(2)}<br><b>Fraudulent:</b> ${tx.Is_Fraudulent ? 'Yes' : 'No'}`);
        });
    }

    // Event listener for the search button
    document.getElementById('searchBtn').addEventListener('click', () => {
        const city = document.getElementById('cityInput').value.trim();
        if (city) {
            fetchTransactions(city);
        } else {
            alert("Please enter a city name.");
        }
    });

    // Fetch default transactions for Lahore on page load
    fetchTransactions("Lahore");
});