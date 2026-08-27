from app.config.settings import NGUONG_COSINE_DISTANCE


class DichVuNhanDang:

    def __init__(self, recognizer):

        self.recognizer = recognizer


    def nhan_dang(
        self,
        anh_bgr,
        threshold=NGUONG_COSINE_DISTANCE
    ):

        embedding = (

            self.recognizer
            .embedder
            .trich_xuat(
                anh_bgr
            )
        )


        if embedding is None:

            return {

                "success": False,

                "message":
                "Không phát hiện được khuôn mặt.",

                "results": []
            }


        return (

            self.recognizer.identify(

                embedding,

                threshold
            )
        )