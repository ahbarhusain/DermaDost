
        function speakResponse(response) {
        const synth = window.speechSynthesis;
        const utterance = new SpeechSynthesisUtterance(response);

        synth.speak(utterance);
        }