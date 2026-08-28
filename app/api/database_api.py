# app/api/database_api.py
# ================================================================
# DATABASE API
# ================================================================

class DatabaseAPI:

    def __init__(self, database):
        self.database = database

    def get_all_users(self):
        return self.database.lay_tat_ca_nguoi()

    def get_user(self, user_id):
        return self.database.lay_nguoi_theo_id(user_id)

    def delete_user(self, user_id):
        self.database.xoa_nguoi(user_id)