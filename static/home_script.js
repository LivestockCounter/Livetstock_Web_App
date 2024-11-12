const dropArea = document.querySelector('.drop-section');
const listSection = document.querySelector('.list-section');
const listContainer = document.querySelector('.list');
const fileSelector = document.querySelector('.file-selector');
const fileSelectorInput = document.querySelector('.file-selector-input');
const customChangeEvent = new Event('change');


// Remove previous file if new one is uploaded
function clearPreviousUploads() {
    listContainer.innerHTML = '';
}
function trysystem() {
 fileSelectorInput.click();
}

// upload files with browse button
fileSelector.onclick = () => fileSelectorInput.click();
fileSelectorInput.onchange = () => {
    [...fileSelectorInput.files].forEach((file) => {
        if (typeValidation(file.type)) {
            clearPreviousUploads(); 
            uploadFile(file);
        }
    });
};

// check the file type (only images and videos)
function typeValidation(type) {
    var splitType = type.split('/')[0];
    return (splitType == 'image' || splitType == 'video'); // Allow only images and videos
}

// when file is over the drag area
dropArea.ondragover = (e) => {
    e.preventDefault();
    [...e.dataTransfer.items].forEach((item) => {
        if (typeValidation(item.type)) {
            dropArea.classList.add('drag-over-effect');
        }
    });
}

// when file leave the drag area
dropArea.ondragleave = () => {
    dropArea.classList.remove('drag-over-effect');
}

// when file drop on the drag area
dropArea.ondrop = (e) => {
    e.preventDefault();
    dropArea.classList.remove('drag-over-effect');
    if (e.dataTransfer.items) {
        [...e.dataTransfer.items].forEach((item) => {
            if (item.kind === 'file') {
                const file = item.getAsFile();
                if (typeValidation(file.type)) {
                    clearPreviousUploads();
                    uploadFile(file);
                }
            }
        });
    } else {
        [...e.dataTransfer.files].forEach((file) => {
            if (typeValidation(file.type)) {
                uploadFile(file);
            }
        });
    }
}

// Trigger file input when the upload box is clicked
document.addEventListener("DOMContentLoaded", function () {
    document.getElementById('upload-box').addEventListener('click', function() {
    document.getElementById('file-input').click();
    });
});

// upload file function
function uploadFile(file) {
    listSection.style.opacity = '1';
    listSection.style.maxHeight = '500px';
    var li = document.createElement('li');
    li.classList.remove('in-prog');
    li.innerHTML = `
        <div class="col">
            <img src="static/icons/${iconSelector(file.type)}" alt="">
        </div>
        <div class="col">
            <div class="file-name">
                <div class="name">${file.name}</div>
                <span>0%</span>
            </div>
            <div class="file-progress">
                <span></span>
            </div>
            <div class="file-size">${(file.size / (1024 * 1024)).toFixed(2)} MB</div>
        </div>
        <div class="col">
            <svg xmlns="http://www.w3.org/2000/svg" class="cross" height="20" width="20"><path d="m5.979 14.917-.854-.896 4-4.021-4-4.062.854-.896 4.042 4.062 4-4.062.854.896-4 4.062 4 4.021-.854.896-4-4.063Z"/></svg>
            <svg xmlns="http://www.w3.org/2000/svg" class="tick" height="20" width="20"></svg>
        </div>
    `;
    listContainer.prepend(li);

    var http = new XMLHttpRequest();
    var data = new FormData();
    data.append('file', file);

    // Send file to Flask endpoint (assuming '/upload')
    http.open('POST', true);

    // Track progress
    http.upload.onprogress = (e) => {
        var percent_complete = (e.loaded / e.total) * 100;
        li.querySelectorAll('span')[0].innerHTML = Math.round(percent_complete) + '%';
        li.querySelectorAll('span')[1].style.width = percent_complete + '%';
    };

    // On successful upload
    http.onload = () => {
        if (http.status === 200) {
            li.classList.add('complete');
            li.classList.remove('in-prog');
        } else {
            console.error('Upload failed', http.status, http.statusText);
        }
    };

// If user clicks cancel or selects a new file
li.querySelector('.cross').onclick = () => {
    fileSelectorInput.value = '';
    listSection.style.opacity = '0';
    listSection.style.maxHeight = '0';
};

    // Send the request
    http.send(data);
}

// Find icon for file
function iconSelector(type) {
    var splitType = (type.split('/')[0] == 'application') ? type.split('/')[1] : type.split('/')[0];
    return splitType + '.png';
}

// NAVBAR
drop_menu = document.querySelector(".drop_menu");
drop_menu.onclick = function(){
    navBar = document.querySelector(".navbar");
    navBar.classList.toggle("active");
}


// TRANSLATION
const originaltext = {
        "intro_text": "Say goodbye to time-consuming manual counting and experience a faster, more reliable way to manage your livestock with precision and efficiency. Our automated Livestock Counting System is designed to make counting animals easier than ever. Integrating advance image recognition and powerful YOLOv8 framework, our system identifies and counts livestock—including cattle, chickens, and goats—directly from images or videos.",
        "upload_h1": "Upload a file to Count for Livestocks.",
        "upload_p": "Only images & videos are accepted.",
        "upload_span": "Drag & Drop file here",
        "file-selector": "Browse Files",
        "drop-here": "Drop Here",
        "list-title": "Uploaded File",
        "process_info": "This may take a while. If you'r process a video, this can take<br> 30 minutes or more depending on the length of the video.",
        "tip_1": "<strong>1. Increase the Accuracy:</strong> Use images and videos that show the entire animal clearly.",
        "tip_2": "<strong>2. Faster Video Processing:</strong> For faster processing, use videos with 30 FPS and a lower bitrate.",
        "tip_3": "<strong>3. File Format:</strong> Use landscape-oriented images and videos. Portrait formats may be cropped during processing.",
        "tip_4": "<strong>4. Remember to Download:</strong> Don’t forget to download your processed results!",
        "tip_5": "<strong>5. If Unresponsive:</strong> Click the Home button if the page becomes unresponsive."
    };

const translations = {
    "intro_text": "Magpaalam na sa mabagal at nakakaubos ng oras na manu-manong pagbibilang at makaranas ng mas mabilis, mas maaasahang paraan upang i-manage ang iyong mga alagang hayop.Ang aming awtomatikong Livestock Counting System ay dinisenyo upang gawing mas madali ang pagbibilang ng mga hayop. Gamit ang pinagsasamang advance image recognition at  YOLOv8 framework, ang aming system ay kayang tumukoy at magbilang ng mga hayop—kabilang ang baka, manok, at kambing—direkta mula sa mga larawan o video.",
    "upload_h1": "Mag-upload ng file para mabilang ang mga hayop.",
    "upload_p": "Larawan at video lang ang pwedeng gamitin.",
    "upload_span": "Ilagay ang file dito",
    "file-selector": "Maghanap",
    "drop-here": "Ilagay Dito",
    "list-title": "Na-upload na File",
    "process_info": "Maaring magtagal ito. Kung nagproposeso ka ng video, maari <br>itong tumagal ng 30 minuto o higit pa depende sa haba ng video.",
    "tip_1": "<strong>1. Increase the Accuracy:</strong> Gumamit ng mga larawan at video kung saan makikita ang buong hayop.",
    "tip_2": "<strong>2. Faster Video Processing:</strong> Para sa mas mabilis na pagproseso, gumamit ng video na may 30 FPS at mababang bitrate.",
    "tip_3": "<strong>3. File Format:</strong> Gumamit ng landscape-oriented na mga larawan at video. Maaaring ma-crop ang mga file na nakaformat ng portrait.",
    "tip_4": "<strong>4. Remember to Download:</strong> Huwag kalimutang i-download ang iyong prinosesong resulta!",
    "tip_5": "<strong>5. If Unresponsive:</strong> Kapag hindi gumagana and page, i-click ang button na Home."
    };

    let isTranslated = false;

    function toggleTranslation() {
        if (isTranslated) {
        document.querySelector(".intro_container p:nth-child(2)").innerHTML = originaltext["intro_text"];
        document.querySelector(".upload_h1").innerHTML = originaltext["upload_h1"];
        document.querySelector(".upload_p").innerHTML = originaltext["upload_p"];
        document.querySelector(".upload_span").innerHTML = originaltext["upload_span"];
        document.querySelector(".file-selector").innerHTML = originaltext["file-selector"];
        document.querySelector(".drop-here").innerHTML = originaltext["drop-here"];
        document.querySelector(".list-title").innerHTML = originaltext["list-title"];
        document.querySelector(".process_info").innerHTML = originaltext["process_info"];
        document.querySelector(".tip_1").innerHTML = originaltext["tip_1"];
        document.querySelector(".tip_2").innerHTML = originaltext["tip_2"];
        document.querySelector(".tip_3").innerHTML = originaltext["tip_3"];
        document.querySelector(".tip_4").innerHTML = originaltext["tip_4"];
        document.querySelector(".tip_5").innerHTML = originaltext["tip_5"];
    } else {
        document.querySelector(".intro_container p:nth-child(2)").innerHTML = translations["intro_text"];
        document.querySelector(".upload_h1").innerHTML = translations["upload_h1"];
        document.querySelector(".upload_p").innerHTML = translations["upload_p"];
        document.querySelector(".upload_span").innerHTML = translations["upload_span"];
        document.querySelector(".file-selector").innerHTML = translations["file-selector"];
        document.querySelector(".drop-here").innerHTML = translations["drop-here"];
        document.querySelector(".process_info").innerHTML = translations["process_info"];
        document.querySelector(".tip_1").innerHTML = translations["tip_1"];
        document.querySelector(".tip_2").innerHTML = translations["tip_2"];
        document.querySelector(".tip_3").innerHTML = translations["tip_3"];
        document.querySelector(".tip_4").innerHTML = translations["tip_4"];
        document.querySelector(".tip_5").innerHTML = translations["tip_5"];
    }

    isTranslated = !isTranslated;
}

// TIPS & GUIDES
document.querySelectorAll('.tips_guides').forEach(item => {
    item.addEventListener('click', function() {
        document.querySelector('.tips_pop_up').style.display = 'flex';
    });
});
document.querySelector('.tips_pop_up').onclick = () =>{
    document.querySelector('.tips_pop_up').style.display = 'none';
}



// LOADER
const loader = document.querySelector(".loader");
const process = document.querySelector(".process");

loader.classList.add("loader_invisible");
process.classList.remove("process_visible");
localStorage.setItem("processing_Inpogress", "False");

let processing_Inpogress = localStorage.getItem("processing_Inpogress");
if (processing_Inpogress === "True") {
    window.addEventListener("load", () =>{
        loader.classList.remove("loader_invisible");
        process.classList.add("process_visible");
    })
}

function file_processing() {
    loader.classList.remove("loader_invisible");
    process.classList.add("process_visible");

    localStorage.setItem("processing_Inpogress", "True");
}


// CANCEL
function restartApp() {
    if (confirm("Are you sure you want to cancel processing?")) {
        fetch('/restart', { method: 'POST' })
            .then(response => {
                if (response.ok) {
                    window.location.href = '/';
                    localStorage.removeItem("processing_Inpogress");
                }
            });
    }
}

function refreshApp() {
    fetch('/restart', { method: 'POST' })
    window.location.href = '/';
    localStorage.removeItem("processing_Inpogress");
}