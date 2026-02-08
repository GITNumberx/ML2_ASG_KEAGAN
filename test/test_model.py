# ===============================
# Improved Model: Random Forest (Depth-Constrained)
# ===============================

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,          # depth constraint (key improvement)
    random_state=RANDOM_STATE
)

# Train model
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluation metrics
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest RMSE: {rmse_rf:.2f}")
print(f"Random Forest MAE: {mae_rf:.2f}")
print(f"Random Forest R²: {r2_rf:.4f}")