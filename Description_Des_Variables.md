Voici une description claire et concise des **21 variables** du jeu de données *Telecom Customer Churn* 📊 :

---

### 🧾 **Variables démographiques**
- **customerID** : Identifiant unique du client (type chaîne de caractères)
- **gender** : Sexe du client (`Male` ou `Female`)
- **SeniorCitizen** : Indique si le client est un senior (`0` = non, `1` = oui)
- **Partner** : Le client vit-il avec un(e) partenaire (`Yes` ou `No`)
- **Dependents** : Le client a-t-il des personnes à charge (`Yes` ou `No`)

---

### 📆 **Informations sur l’abonnement**
- **tenure** : Nombre de mois depuis l’abonnement du client
- **Contract** : Type de contrat (`Month-to-month`, `One year`, `Two year`)
- **PaperlessBilling** : Facturation sans papier (`Yes` ou `No`)
- **PaymentMethod** : Mode de paiement (`Electronic check`, `Mailed check`, `Bank transfer`, `Credit card`)

---

### 📞 **Services souscrits**
- **PhoneService** : Le client a-t-il un service téléphonique (`Yes` ou `No`)
- **MultipleLines** : Le client a-t-il plusieurs lignes (`Yes`, `No`, `No phone service`)
- **InternetService** : Type d’accès Internet (`DSL`, `Fiber optic`, `No`)
- **OnlineSecurity** : Sécurité en ligne (`Yes`, `No`, `No internet service`)
- **OnlineBackup** : Sauvegarde en ligne (`Yes`, `No`, `No internet service`)
- **DeviceProtection** : Protection des appareils (`Yes`, `No`, `No internet service`)
- **TechSupport** : Assistance technique (`Yes`, `No`, `No internet service`)
- **StreamingTV** : Accès à la télévision en streaming (`Yes`, `No`, `No internet service`)
- **StreamingMovies** : Accès aux films en streaming (`Yes`, `No`, `No internet service`)

---

### 💰 **Variables financières**
- **MonthlyCharges** : Montant mensuel facturé au client
- **TotalCharges** : Montant total facturé depuis le début de l’abonnement

---

### 🎯 **Variable cible**
- **Churn** : Le client a-t-il quitté l’entreprise (`Yes` ou `No`)

---