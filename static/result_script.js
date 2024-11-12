
// NAVBAR
drop_menu = document.querySelector(".drop_menu");
drop_menu.onclick = function(){
    navBar = document.querySelector(".navbar");
    navBar.classList.toggle("active");
}

// IMAGE/VIDEO RESIZE
document.querySelectorAll('.output_result img, .output_result video').forEach(item => {
    item.addEventListener('click', function() {
        document.querySelector('.pop_up').style.display = 'block';
    });
});
document.querySelector('.pop_up span').onclick = () =>{
    document.querySelector('.pop_up').style.display = 'none';
}


// LINK
document.querySelector(".back_button a").addEventListener("mouseover", function() {
    document.querySelector(".download a").classList.add("hover-download");
});
document.querySelector(".back_button a").addEventListener("mouseout", function() {
    document.querySelector(".download a").classList.remove("hover-download");
});

document.querySelector(".download a").addEventListener("mouseover", function() {
    document.querySelector(".back_button a").classList.add("hover-back_button");
    document.querySelector(".download a").classList.add("hover-download");
});
document.querySelector(".download a").addEventListener("mouseout", function() {
    document.querySelector(".back_button a").classList.remove("hover-back_button");
    document.querySelector(".download a").classList.remove("hover-download");
});


// TIPS & GUIDES
document.querySelectorAll('.tips_guides').forEach(item => {
    item.addEventListener('click', function() {
        document.querySelector('.tips_pop_up').style.display = 'flex';
    });
});
document.querySelector('.tips_pop_up').onclick = () =>{
    document.querySelector('.tips_pop_up').style.display = 'none';
}

// TRANSLATION
const originaltext = {
    "result_info": "This version of Livestock Counting System still needs improvement. Results may sometimes include incorrect detections and inaccurate counts. <strong><br>Thank you For using our system!</strong>",
    "tip_1": "<strong>1. Increase the Accuracy:</strong> Use images and videos that show the entire animal clearly.",
    "tip_2": "<strong>2. Faster Video Processing:</strong> For faster processing, use videos with 30 FPS and a lower bitrate.",
    "tip_3": "<strong>3. File Format:</strong> Use landscape-oriented images and videos. Portrait formats may be cropped during processing.",
    "tip_4": "<strong>4. Remember to Download:</strong> Don’t forget to download your processed results!",
    "tip_5": "<strong>5. If Unresponsive:</strong> Click the Home button if the page becomes unresponsive."
};

const translations = {
"result_info": "Ang Livestock Counting System na ito ay nangangailangan pa ng kaunting kaayusan. Ang mga resulta ay maaaring magkaroon ng mga maling detection at hindi tamang bilang. <strong><br>Salamat sa paggamit ng aming system!<strong>",
"tip_1": "<strong>1. Increase the Accuracy:</strong> Gumamit ng mga larawan at video kung saan makikita ang buong hayop.",
"tip_2": "<strong>2. Faster Video Processing:</strong> Para sa mas mabilis na pagproseso, gumamit ng video na may 30 FPS at mababang bitrate.",
"tip_3": "<strong>3. File Format:</strong> Gumamit ng landscape-oriented na mga larawan at video. Maaaring ma-crop ang mga file na nakaformat ng portrait.",
"tip_4": "<strong>4. Remember to Download:</strong> Huwag kalimutang i-download ang iyong prinosesong resulta!",
"tip_5": "<strong>5. If Unresponsive:</strong> Kapag hindi gumagana and page, i-click ang button na Home."
};

let isTranslated = false;

function toggleTranslation() {
    if (isTranslated) {
    document.querySelector(".result_info").innerHTML = originaltext["result_info"];
    document.querySelector(".tip_1").innerHTML = originaltext["tip_1"];
    document.querySelector(".tip_2").innerHTML = originaltext["tip_2"];
    document.querySelector(".tip_3").innerHTML = originaltext["tip_3"];
    document.querySelector(".tip_4").innerHTML = originaltext["tip_4"];
    document.querySelector(".tip_5").innerHTML = originaltext["tip_5"];
} else {
    document.querySelector(".result_info").innerHTML = translations["result_info"];
    document.querySelector(".tip_1").innerHTML = translations["tip_1"];
    document.querySelector(".tip_2").innerHTML = translations["tip_2"];
    document.querySelector(".tip_3").innerHTML = translations["tip_3"];
    document.querySelector(".tip_4").innerHTML = translations["tip_4"];
    document.querySelector(".tip_5").innerHTML = translations["tip_5"];
}

isTranslated = !isTranslated;
}

// HOME     
function refreshApp() {
    window.location.href = '/';
    localStorage.removeItem("processing_Inpogress");
    fetch('/restart', { method: 'POST' })
}



// old
document.querySelectorAll('.video_result img, .video_result video').forEach(item => {
    item.addEventListener('click', function() {
        this.classList.toggle('expand');
    });
});