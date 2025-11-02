# Ανάλυση και Σύγκριση Αλγορίθμων Multi-Armed Bandit: ε-greedy και Softmax

## 1. Εισαγωγή

### 1.1 Ορισμός του Προβλήματος

Το πρόβλημα του Multi-Armed Bandit (MAB) αποτελεί θεμελιώδες πρόβλημα στη θεωρία λήψης αποφάσεων υπό αβεβαιότητα. Στο κλασικό πλαίσιο, ένας πράκτορας (agent) αντιμετωπίζει n διαφορετικούς μοχλούς (arms), καθένας από τους οποίους παρέχει ανταμοιβές από μια άγνωστη στάσιμη κατανομή πιθανότητας. Ο στόχος του πράκτορα είναι η μεγιστοποίηση της αθροιστικής ανταμοιβής σε ορίζοντα Τ χρονικών βημάτων.

Μαθηματικά, κάθε μοχλός i ∈ {1, 2, ..., n} χαρακτηρίζεται από μια κατανομή πιθανότητας P_i με μέση τιμή μ_i. Όταν ο πράκτορας επιλέγει μοχλό i στο χρόνο t, λαμβάνει ανταμοιβή r_t ~ P_i. Η βέλτιστη στρατηγική θα ήταν η συνεχής επιλογή του μοχλού με τη μέγιστη μέση ανταμοιβή μ* = max_i μ_i.

### 1.2 Το Δίλημμα Exploration-Exploitation

Το κεντρικό δίλημμα του MAB είναι η ισορροπία μεταξύ:
- **Exploration (Εξερεύνηση)**: Δοκιμή υποδεέστερων μοχλών για απόκτηση πληροφορίας
- **Exploitation (Εκμετάλλευση)**: Επιλογή του τρέχοντος καλύτερου μοχλού για μεγιστοποίηση άμεσης ανταμοιβής

Το regret ορίζεται ως: R_T = Τμ* - Σ_{t=1}^{T} r_t, όπου μ* είναι η μέση ανταμοιβή του βέλτιστου μοχλού.

---

## 2. Μεθοδολογία

### 2.1 Πειραματική Διάταξη

**Παράμετροι Πειράματος:**
- Αριθμός μοχλών: n = 5
- Αριθμός παιχνιδιών ανά πείραμα: T = 1000
- Αριθμός επαναλήψεων: 100 (για στατιστική σημαντικότητα)
- Κατανομή ανταμοιβών: Κανονική N(μ_i, σ²) με σ = 1.0
- Πραγματικές μέσες τιμές: Τυχαίες από U(0, 2)

### 2.2 Αλγόριθμος ε-greedy

Ο αλγόριθμος ε-greedy είναι μια απλή στοχαστική στρατηγική που ισορροπεί exploration και exploitation:

**Πολιτική Επιλογής:**
```
Με πιθανότητα ε:
    Επίλεξε τυχαία μοχλό (uniform exploration)
Με πιθανότητα 1-ε:
    Επίλεξε arg max_i Q_t(i) (greedy exploitation)
```

**Ενημέρωση Εκτιμήσεων:**
Οι εκτιμήσεις Q_t(i) ενημερώνονται με incremental averaging:

Q_{t+1}(i) = Q_t(i) + (1/N_t(i)) * (r_t - Q_t(i))

όπου N_t(i) είναι ο αριθμός φορών που επιλέχθηκε ο μοχλός i μέχρι το χρόνο t.

**Θεωρητικές Ιδιότητες:**
- Για σταθερό ε > 0: Sublinear regret O(T^{2/3})
- Για ε_t → 0: Ασυμπτωτική σύγκλιση στη βέλτιστη πολιτική
- Απλότητα υλοποίησης και υπολογιστική αποδοτικότητα

### 2.3 Αλγόριθμος Softmax (Boltzmann Exploration)

Ο αλγόριθμος Softmax χρησιμοποιεί την κατανομή Boltzmann για πιθανοτική επιλογή βάσει των εκτιμώμενων αξιών:

**Πολιτική Επιλογής:**
```
P(a_t = i) = exp(Q_t(i)/τ) / Σ_j exp(Q_t(j)/τ)
```

όπου τ > 0 είναι η παράμετρος θερμοκρασίας που ελέγχει το επίπεδο exploration:
- τ → 0: Καθαρή εκμετάλλευση (greedy)
- τ → ∞: Uniform τυχαία επιλογή
- τ = 1: Balanced exploration-exploitation

**Ενημέρωση Εκτιμήσεων:**
Όπως στον ε-greedy: Q_{t+1}(i) = Q_t(i) + (1/N_t(i)) * (r_t - Q_t(i))

**Θεωρητικές Ιδιότητες:**
- Smoother exploration: Υψηλότερες πιθανότητες σε πιο υποσχόμενους μοχλούς
- Adaptive exploration: Η exploration μειώνεται φυσικά καθώς οι διαφορές στις εκτιμήσεις μεγαλώνουν
- Υπολογιστικά πιο ακριβής λόγω του υπολογισμού exponentials

---

## 3. Αρχιτεκτονική Υλοποίησης

### 3.1 Υψηλού Επιπέδου Αρχιτεκτονική

```
┌─────────────────────────────────────────────────────┐
│              Επίπεδο Πειραματικής Εκτέλεσης         │
│  - Διαχείριση επαναλήψεων                           │
│  - Συγκέντρωση αποτελεσμάτων                        │
│  - Στατιστική ανάλυση                               │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│              Επίπεδο Αλγορίθμου                     │
│  ┌──────────────────┐    ┌──────────────────┐       │
│  │  epsilon_greedy  │    │  softmax_bandit  │       │
│  │  - Επιλογή arm   │    │  - Επιλογή arm   │       │
│  │  - Ενημέρωση Q   │    │  - Ενημέρωση Q   │       │
│  └──────────────────┘    └──────────────────┘       │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│              Επίπεδο Περιβάλλοντος                  │
│  - Γεννήτρια ανταμοιβών (Κανονική κατανομή)         │
│  - Διατήρηση πραγματικών παραμέτρων                 │
└─────────────────────────────────────────────────────┘
```

### 3.2 Ανάλυση Κώδικα

#### 3.2.1 Αρχικοποίηση Περιβάλλοντος

```python
n_arms = 5
true_means = np.random.uniform(0, 2, n_arms)
optimal_arm = np.argmax(true_means)
```

**Σχεδιαστικές Επιλογές:**
- Χρήση Uniform(0,2) για τις μέσες τιμές εξασφαλίζει επαρκή διαφοροποίηση
- Η τυπική απόκλιση σ=1.0 δημιουργεί υπερκάλυψη στις κατανομές (σημαντικό για exploration)

#### 3.2.2 Συνάρτηση ε-greedy

```python
def epsilon_greedy(epsilon=0.1):
    for _ in range(n_experiments):
        counts = np.zeros(n_arms)      # N_t(i): Πλήθος επιλογών
        values = np.zeros(n_arms)      # Q_t(i): Εκτιμήσεις
        
        for _ in range(n_plays):
            # Exploration vs Exploitation
            if np.random.random() < epsilon:
                arm = np.random.randint(n_arms)
            else:
                arm = np.argmax(values)
            
            # Sampling από περιβάλλον
            reward = np.random.normal(true_means[arm], 1.0)
            
            # Incremental update
            counts[arm] += 1
            values[arm] += (reward - values[arm]) / counts[arm]
```

**Πολυπλοκότητα:**
- Χωρική: O(n) για αποθήκευση counts και values
- Χρονική: O(n) ανά βήμα για argmax, O(1) για update
- Συνολική: O(Tn) για T παιχνίδια

**Αριθμητική Σταθερότητα:**
Το incremental averaging αποφεύγει overflow και διατηρεί ακρίβεια:
- Αποφυγή αποθήκευσης άθροισμα/count ξεχωριστά
- Κάθε update είναι O(1) χωρίς επανυπολογισμό όλων των rewards

#### 3.2.3 Συνάρτηση Softmax

```python
from scipy.special import softmax

def softmax_bandit(temperature=0.5):
    for _ in range(n_experiments):
        counts = np.zeros(n_arms)
        values = np.zeros(n_arms)
        
        for _ in range(n_plays):
            # Boltzmann distribution
            probs = softmax(values / temperature)
            arm = np.random.choice(n_arms, p=probs)
            
            reward = np.random.normal(true_means[arm], 1.0)
            
            counts[arm] += 1
            values[arm] += (reward - values[arm]) / counts[arm]
```

**Υλοποίηση scipy.special.softmax:**
Η συνάρτηση χρησιμοποιεί numerically stable implementation:
```
softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
```
Το αφαίρεση του max(x) αποτρέπει overflow σε μεγάλες τιμές.

**Πολυπλοκότητα:**
- Χωρική: O(n)
- Χρονική: O(n) ανά βήμα για softmax και sampling
- Συνολική: O(Tn)

#### 3.2.4 Στατιστική Συγκέντρωση

```python
total_rewards = []
optimal_actions = []

for _ in range(n_experiments):
    # ... εκτέλεση αλγορίθμου ...
    total_rewards.append(rewards)
    optimal_actions.append(optimal)

return np.mean(total_rewards, axis=0), np.mean(optimal_actions, axis=0)
```

**Στατιστική Αξιοπιστία:**
- n=100 επαναλήψεις παρέχουν 95% CI με σφάλμα ~±2σ/√100
- Averaging κατά μήκος axis=0 διατηρεί χρονική δυναμική

---

## 4. Ανάλυση Αποτελεσμάτων

### 4.1 Γράφημα 1: Μέση Ανταμοιβή ανά Παιχνίδι

**Παρατηρήσεις:**

1. **Αρχική Φάση (t < 100)**:
   - Και οι δύο αλγόριθμοι ξεκινούν από χαμηλή απόδοση (~0.9)
   - Ταχεία αύξηση καθώς συγκεντρώνουν πληροφορία
   - Υψηλή διακύμανση λόγω exploration

2. **ε-greedy Συμπεριφορά**:
   - Φθάνει υψηλότερες μέσες ανταμοιβές (~1.9-2.1)
   - Εμφανίζει μεγαλύτερη διακύμανση (variance)
   - Συνεχής exploration (10%) δημιουργεί θόρυβο

3. **Softmax Συμπεριφορά**:
   - Σταθεροποιείται σε χαμηλότερη ανταμοιβή (~1.4-1.8)
   - Μικρότερη διακύμανση (smoother curve)
   - Με τ=0.5 κάνει υπερβολική εκμετάλλευση νωρίς

**Αιτιολόγηση:**
Το ε-greedy με ε=0.1 εξασφαλίζει επαρκή exploration σε όλη τη διάρκεια, επιτρέποντας διόρθωση λανθασμένων αρχικών εκτιμήσεων. Το Softmax με τ=0.5 "κλειδώνει" νωρίς σε suboptimal μοχλούς, καθώς οι πιθανότητες γίνονται πολύ ανισομερείς.

### 4.2 Γράφημα 2: Αθροιστική Μέση Ανταμοιβή

**Παρατηρήσεις:**

1. **Μακροπρόθεσμη Απόδοση**:
   - ε-greedy συγκλίνει σε ~1.75
   - Softmax συγκλίνει σε ~1.63
   - Διαφορά ~7.5% υπέρ του ε-greedy

2. **Ρυθμός Σύγκλισης**:
   - Και οι δύο αλγόριθμοι σταθεροποιούνται στα t≈200
   - ε-greedy έχει μικρή upward τάση (συνεχής exploration)
   - Softmax plateau-αρει νωρίτερα

3. **Regret Ανάλυση**:
   - Αν μ* ≈ 1.8, τότε:
     - ε-greedy: R_1000 ≈ 50 (linear regret λόγω σταθερού ε)
     - Softmax: R_1000 ≈ 170 (μεγαλύτερο λόγω suboptimal convergence)

**Θεωρητική Ερμηνεία:**
Το σταθερό ε οδηγεί σε O(T) regret, αλλά με μικρή σταθερά. Το Softmax με σταθερό τ έχει παρόμοιο asymptotic regret αλλά χειρότερη σταθερά για αυτή την επιλογή παραμέτρων.

### 4.3 Γράφημα 3: Ποσοστό Βέλτιστων Ενεργειών

**Παρατηρήσεις:**

1. **ε-greedy**:
   - Φθάνει ~93% βέλτιστες ενέργειες
   - Θεωρητικό όριο: 100% - ε = 90% (για uniform exploration)
   - Το 93% > 90% υποδηλώνει ότι μερικές φορές η τυχαία επιλογή τυχαίνει ο βέλτιστος

2. **Softmax**:
   - Σταθεροποιείται στο ~57% βέλτιστες ενέργειες
   - Σημαντική υστέρηση υποδηλώνει premature convergence
   - Το σφάλμα διατηρείται σταθερό (δεν autocorrect)

3. **Δυναμική Εξέλιξη**:
   - ε-greedy: Γρήγορη άνοδος (0 → 80% στα πρώτα 100 παιχνίδια)
   - Softmax: Πιο αργή και χαμηλότερη σύγκλιση

**Αιτία Υποδόσης Softmax:**
Με τ=0.5 και κανονική κατανομή σ=1.0, οι αρχικές εκτιμήσεις έχουν μεγάλο θόρυβο. Αν ένας suboptimal μοχλός τύχει να έχει καλά δείγματα αρχικά, το Softmax του δίνει πολύ υψηλή πιθανότητα, οδηγώντας σε positive feedback loop.

---

## 5. Συγκριτική Αξιολόγηση

### 5.1 Πίνακας Σύγκρισης

| Κριτήριο | ε-greedy (ε=0.1) | Softmax (τ=0.5) |
|----------|------------------|-----------------|
| Μέση Αθροιστική Ανταμοιβή | 1750 | 1630 |
| Ποσοστό Βέλτιστων | 93% | 57% |
| Variance | Υψηλή | Χαμηλή |
| Υπολογιστική Πολυπλοκότητα | O(n) | O(n) |
| Απλότητα | Υψηλή | Μέτρια |
| Ευαισθησία σε Παραμέτρους | Χαμηλή (ε robust) | Υψηλή (τ critical) |

### 5.2 Θεωρητικές Εξηγήσεις

**Γιατί το ε-greedy υπεραπέδωσε:**

1. **Uniform Exploration Guarantee**: Το 10% τυχαία exploration εξασφαλίζει ότι όλοι οι μοχλοί δοκιμάζονται επαρκώς
2. **Robustness σε Early Bias**: Μπορεί να διορθώσει λανθασμένες αρχικές εκτιμήσεις
3. **Asymptotic Optimality**: Για ε_t = c/t, ε-greedy είναι asymptotically optimal

**Γιατί το Softmax υποαπέδωσε:**

1. **Temperature Mismatch**: Το τ=0.5 είναι πολύ μικρό για σ=1.0 (ideally τ ≈ σ)
2. **Premature Exploitation**: Οι πιθανότητες γίνονται πολύ ακραίες πολύ γρήγορα
3. **Lack of Hard Exploration**: Δεν υπάρχει εγγύηση δειγματοληψίας όλων των μοχλών

### 5.3 Βελτιώσεις και Επεκτάσεις

**Για ε-greedy:**
- **Decaying ε**: ε_t = min(1, c/(d + t)) για sublinear regret
- **Optimistic Initialization**: Q_0(i) = μ_max για early exploration
- **UCB Combination**: Hybrid ε-greedy + upper confidence bounds

**Για Softmax:**
- **Temperature Scheduling**: τ_t = τ_0 / log(t+2) για adaptive exploration
- **Auto-tuning τ**: Βασισμένο στην εμπειρική διακύμανση των ανταμοιβών
- **Gradient Bandit**: Παραμετρικός Softmax με gradient ascent

---

## 6. Συμπεράσματα

### 6.1 Κύρια Ευρήματα

1. **Απόδοση**: Το ε-greedy με ε=0.1 υπερτερεί κατά 7.5% σε αθροιστική ανταμοιβή
2. **Σταθερότητα**: Το Softmax παρέχει πιο σταθερές, προβλέψιμες ανταμοιβές
3. **Παράμετροι**: Η απόδοση του Softmax είναι πολύ ευαίσθητη στην επιλογή του τ
4. **Complexity Trade-off**: Και οι δύο έχουν O(Tn), αλλά ε-greedy είναι απλούστερος

### 6.2 Πρακτικές Συστάσεις

**Χρησιμοποιήστε ε-greedy όταν:**
- Χρειάζεστε απλότητα και robustness
- Το regret είναι κρίσιμο (e.g., clinical trials)
- Δεν έχετε χρόνο για parameter tuning

**Χρησιμοποιήστε Softmax όταν:**
- Χρειάζεστε smooth, differentiable πολιτικές (για gradient methods)
- Μπορείτε να κάνετε calibration του τ
- Θέλετε σταδιακή μετάβαση exploration → exploitation

### 6.3 Μελλοντική Εργασία

1. **Adaptive Algorithms**: Thompson Sampling, UCB1, Exp3
2. **Contextual Bandits**: Χρήση features για την επιλογή μοχλών
3. **Non-stationary Environments**: Sliding windows, discounted updates
4. **Multi-objective**: Pareto-optimal exploration-exploitation

---

## Βιβλιογραφία

[1] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

[2] Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. *Machine Learning*, 47(2-3), 235-256.

[3] Lattimore, T., & Szepesvári, C. (2020). *Bandit Algorithms*. Cambridge University Press.

[4] Vermorel, J., & Mohri, M. (2005). Multi-armed bandit algorithms and empirical evaluation. *European Conference on Machine Learning*, 437-448.

[5] Garivier, A., & Moulines, E. (2011). On upper-confidence bound policies for switching bandit problems. *International Conference on Algorithmic Learning Theory*, 174-188.
