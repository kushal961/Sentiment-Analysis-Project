document.addEventListener("DOMContentLoaded", function () {
    // Add hover effect to the button
    const button = document.querySelector(".btn");
    
    button.addEventListener("mouseover", function () {
        button.style.transform = "scale(1.1)";
    });

    button.addEventListener("mouseleave", function () {
        button.style.transform = "scale(1)";
    });
});

