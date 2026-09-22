class MultimodalFusion:
    def __init__(self):
        # Weights representing the confidence in each modality
        # Video (facial micro-expressions) usually yields highest reliability
        # Audio (stress) is also strong, text usually helps as a contextual backup
        self.weights = {
            'video': 0.50,
            'audio': 0.35,
            'text': 0.15
        }

    def predict_fusion(self, video_score, audio_score, text_score, available=None, **kwargs):
        """
        Aggregates predictions from available modalities.
        Input scores represent the probability of DECEPTION (0.0 to 1.0)
        """
        if available is None:
            available = {'video': True, 'audio': True, 'text': True}

        used_weights = {
            key: self.weights[key]
            for key, use in available.items()
            if use
        }
        if not used_weights:
            used_weights = self.weights.copy()

        total_weight = sum(used_weights.values())
        normalized_weights = {
            key: (weight / total_weight)
            for key, weight in used_weights.items()
        }

        final_score = 0.0
        final_score += video_score * normalized_weights.get('video', 0.0)
        final_score += audio_score * normalized_weights.get('audio', 0.0)
        final_score += text_score * normalized_weights.get('text', 0.0)

        if final_score >= 0.5:
            prediction = "Lie"
            confidence = final_score
        else:
            prediction = "Truth"
            confidence = 1.0 - final_score

        return prediction, confidence, final_score
