import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ↑↓ to select a card, ←→ to select a territory. Press Space to play the selected card."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Conquer the Card Kingdom! Play cards to capture territories on the map. Control 7 of the 10 territories to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # Fonts
        try:
            self.font_main = pygame.font.SysFont("Arial", 24, bold=True)
            self.font_small = pygame.font.SysFont("Arial", 16)
            self.font_card = pygame.font.SysFont("Arial", 18, bold=True)
            self.font_card_small = pygame.font.SysFont("Arial", 12)
        except pygame.error:
            self.font_main = pygame.font.Font(None, 30)
            self.font_small = pygame.font.Font(None, 22)
            self.font_card = pygame.font.Font(None, 24)
            self.font_card_small = pygame.font.Font(None, 18)

        # Colors
        self.COLOR_BG = (40, 42, 54)  # dracula bg
        self.COLOR_PLAYER = (80, 250, 123)  # dracula green
        self.COLOR_OPPONENT = (255, 85, 85)  # dracula red
        self.COLOR_NEUTRAL = (98, 114, 164)  # dracula comment
        self.COLOR_GRID = (68, 71, 90)  # dracula current line
        self.COLOR_TEXT = (248, 248, 242)  # dracula foreground
        self.COLOR_CURSOR = (255, 184, 108)  # custom orange
        self.COLOR_INVALID = (255, 121, 198)  # dracula pink
        self.CARD_COLORS = {
            "Attack": (255, 85, 85),
            "Reinforce": (189, 147, 249),  # dracula purple
        }

        # Game constants
        self.MAX_STEPS = 500
        self.HAND_SIZE = 4
        self.NUM_TERRITORIES = 10
        self.WIN_CONDITION = 7

        # Initialize state variables
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.territories = []
        self.all_cards = []
        self.player_hand = []
        self.opponent_hand = []
        self.player_card_cursor = 0
        self.player_territory_cursor = 0
        self.animation = None
        self.particles = []
        self.feedback_message = ""
        self.feedback_timer = 0

        self._initialize_territories()
        self.reset()

        self.validate_implementation()

    def _initialize_territories(self):
        # Create territory layout (2x5 grid) - deterministic part
        grid_w, grid_h = 110, 70
        start_x, start_y = (self.screen_width - 5 * (grid_w + 10) + 10) // 2, 80
        for i in range(self.NUM_TERRITORIES):
            row, col = divmod(i, 5)
            x = start_x + col * (grid_w + 10)
            y = start_y + row * (grid_h + 10)
            adj = []
            if col > 0:
                adj.append(i - 1)  # Left
            if col < 4:
                adj.append(i + 1)  # Right
            if row > 0:
                adj.append(i - 5)  # Up
            if row < 1:
                adj.append(i + 5)  # Down
            self.territories.append(
                {
                    "id": i,
                    "owner": 0,
                    "strength": 0,
                    "pos": (x, y),
                    "size": (grid_w, grid_h),
                    "adj": adj,
                }
            )

    def _create_card_deck(self):
        # Create card deck - randomized part, requires self.np_random
        self.all_cards = []
        card_id = 0
        for _ in range(10):  # More low-strength cards
            self.all_cards.append(
                {
                    "id": card_id,
                    "name": "Attack",
                    "strength": self.np_random.integers(1, 3),
                    "ability": None,
                }
            )
            card_id += 1
        for _ in range(7):  # Fewer mid-strength cards
            self.all_cards.append(
                {
                    "id": card_id,
                    "name": "Attack",
                    "strength": self.np_random.integers(3, 5),
                    "ability": None,
                }
            )
            card_id += 1
        for _ in range(3):  # Few high-strength cards
            self.all_cards.append(
                {"id": card_id, "name": "Attack", "strength": 5, "ability": None}
            )
            card_id += 1
        for _ in range(5):  # Some special cards
            self.all_cards.append(
                {
                    "id": card_id,
                    "name": "Reinforce",
                    "strength": self.np_random.integers(1, 3),
                    "ability": "reinforce",
                }
            )
            card_id += 1

    def _draw_new_cards(self, hand, count=1):
        for _ in range(count):
            if len(hand) < self.HAND_SIZE:
                new_card = self.np_random.choice(self.all_cards).copy()
                hand.append(new_card)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Create the deck of cards using the seeded RNG
        self._create_card_deck()

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.animation = None
        self.particles = []
        self.feedback_message = ""
        self.feedback_timer = 0

        for t in self.territories:
            t["owner"] = 0  # 0: Neutral, 1: Player, 2: Opponent
            t["strength"] = 0

        # Initial territory ownership
        player_start = self.np_random.integers(0, self.NUM_TERRITORIES)
        self.territories[player_start]["owner"] = 1
        self.territories[player_start]["strength"] = 1

        opponent_start_candidates = [
            i for i, t in enumerate(self.territories) if t["owner"] == 0
        ]
        opponent_start = self.np_random.choice(opponent_start_candidates)
        self.territories[opponent_start]["owner"] = 2
        self.territories[opponent_start]["strength"] = 1

        self.player_hand = []
        self.opponent_hand = []
        self._draw_new_cards(self.player_hand, self.HAND_SIZE)
        self._draw_new_cards(self.opponent_hand, self.HAND_SIZE)

        self.player_card_cursor = 0
        self.player_territory_cursor = 0

        self._update_score()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1  # Boolean, we only care about the press event
        shift_held = action[2] == 1  # Boolean

        reward = 0
        terminated = False

        # --- Handle Cursor Movement ---
        if self.player_hand: # Prevent crash if hand is empty
            if movement == 1:  # Up
                self.player_card_cursor = (self.player_card_cursor - 1 + len(self.player_hand)) % len(self.player_hand)
            elif movement == 2:  # Down
                self.player_card_cursor = (self.player_card_cursor + 1) % len(self.player_hand)
        if movement == 3:  # Left
            self.player_territory_cursor = (self.player_territory_cursor - 1 + self.NUM_TERRITORIES) % self.NUM_TERRITORIES
        elif movement == 4:  # Right
            self.player_territory_cursor = (self.player_territory_cursor + 1) % self.NUM_TERRITORIES

        # --- Handle Player Action ---
        if space_pressed and self.player_hand:
            self.steps += 1
            card_idx = self.player_card_cursor
            terr_idx = self.player_territory_cursor

            card = self.player_hand[card_idx]
            territory = self.territories[terr_idx]

            # Invalid move: trying to play on own territory
            if territory["owner"] == 1:
                reward = -0.1
                self._set_feedback("Cannot attack your own territory!", self.COLOR_INVALID)
            else:
                # --- Player's Turn ---
                # SFX: Card play
                old_owner = territory["owner"]
                if card["strength"] > territory["strength"]:
                    # Capture
                    territory["owner"] = 1
                    territory["strength"] = card["strength"]
                    if old_owner != 1:
                        reward += 1.0  # Capture bonus
                        self._set_feedback("Territory Captured!", self.COLOR_PLAYER)
                    self._create_particles(territory, self.COLOR_PLAYER)
                    # SFX: Capture success

                    # Handle special ability
                    if card.get("ability") == "reinforce":
                        for adj_idx in territory["adj"]:
                            if self.territories[adj_idx]["owner"] == 1:
                                self.territories[adj_idx]["strength"] += 1
                                self._create_particles(
                                    self.territories[adj_idx],
                                    self.COLOR_CURSOR,
                                    count=10,
                                )
                else:
                    # Defended
                    reward += 0.0  # No change
                    self._set_feedback("Attack Defended!", self.COLOR_OPPONENT)
                    self._create_particles(territory, self.COLOR_NEUTRAL)
                    # SFX: Defend

                self.player_hand.pop(card_idx)
                self._draw_new_cards(self.player_hand, 1)
                if self.player_hand:
                    self.player_card_cursor = min(self.player_card_cursor, len(self.player_hand) - 1)
                else:
                    self.player_card_cursor = 0


                # --- AI's Turn ---
                ai_reward = self._ai_turn()
                reward -= ai_reward  # Subtract reward if AI captures player territory

            # --- Resolution ---
            self._update_score()

            if self.score >= self.WIN_CONDITION:
                reward += 100
                terminated = True
                self.game_over = True
                self._set_feedback("VICTORY!", self.COLOR_PLAYER)
            elif self.score == 0:
                reward -= 100
                terminated = True
                self.game_over = True
                self._set_feedback("DEFEAT!", self.COLOR_OPPONENT)

        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self._set_feedback("Time Limit Reached.", self.COLOR_NEUTRAL)

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _ai_turn(self):
        if not self.opponent_hand:
            return 0

        # AI difficulty scaling
        difficulty_bonus = int(self.score * 0.2)

        # AI chooses best card
        best_card = max(self.opponent_hand, key=lambda c: c["strength"])
        ai_strength = best_card["strength"] + difficulty_bonus

        # AI chooses target
        player_territories = [t for t in self.territories if t["owner"] == 1]
        neutral_territories = [t for t in self.territories if t["owner"] == 0]

        target = None
        # Priority 1: Capture weakest player territory
        weakest_player_territories = sorted(
            [t for t in player_territories if ai_strength > t["strength"]],
            key=lambda t: t["strength"],
        )
        if weakest_player_territories:
            target = weakest_player_territories[0]
        # Priority 2: Capture any neutral territory
        elif neutral_territories:
            target = self.np_random.choice(neutral_territories)
        # Priority 3: Attack strongest player territory it can't beat (to weaken for later)
        elif player_territories:
            target = max(player_territories, key=lambda t: t["strength"])
        else:  # No valid targets
            return 0

        # Execute AI move
        loss_penalty = 0
        if ai_strength > target["strength"]:
            if target["owner"] == 1:
                loss_penalty = 1.0  # Player lost a territory
            target["owner"] = 2
            target["strength"] = ai_strength
            self._create_particles(target, self.COLOR_OPPONENT)
            # SFX: Enemy capture
        else:
            self._create_particles(target, self.COLOR_NEUTRAL)
            # SFX: Player defend

        self.opponent_hand.remove(best_card)
        self._draw_new_cards(self.opponent_hand, 1)

        return loss_penalty

    def _update_score(self):
        self.score = sum(1 for t in self.territories if t["owner"] == 1)
        self.opponent_score = sum(1 for t in self.territories if t["owner"] == 2)

    def _set_feedback(self, message, color):
        self.feedback_message = message
        self.feedback_color = color
        self.feedback_timer = 60  # frames

    def _create_particles(self, territory, color, count=30):
        cx = territory["pos"][0] + territory["size"][0] // 2
        cy = territory["pos"][1] + territory["size"][1] // 2
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append(
                {
                    "pos": [cx, cy],
                    "vel": vel,
                    "lifespan": self.np_random.integers(20, 40),
                    "color": color,
                    "radius": self.np_random.uniform(2, 5),
                }
            )

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Update and draw particles
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1  # gravity
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p["lifespan"] / 40))
                pygame.gfxdraw.filled_circle(
                    self.screen,
                    int(p["pos"][0]),
                    int(p["pos"][1]),
                    int(p["radius"]),
                    (*p["color"], alpha),
                )

        # Draw territories
        for i, t in enumerate(self.territories):
            color = [self.COLOR_NEUTRAL, self.COLOR_PLAYER, self.COLOR_OPPONENT][
                t["owner"]
            ]
            rect = pygame.Rect(t["pos"], t["size"])

            # Draw selector pulse
            if i == self.player_territory_cursor:
                pulse = abs(math.sin(pygame.time.get_ticks() * 0.005))
                sel_color = self.COLOR_CURSOR
                if t["owner"] == 1:  # Invalid target
                    sel_color = self.COLOR_INVALID

                # Draw glowing outline
                for offset in range(1, 5):
                    alpha = 150 - offset * 30
                    pygame.gfxdraw.rectangle(
                        self.screen,
                        rect.inflate(offset * 2, offset * 2),
                        (*sel_color, alpha),
                    )

            pygame.draw.rect(self.screen, color, rect, border_radius=5)
            pygame.draw.rect(self.screen, self.COLOR_GRID, rect, width=2, border_radius=5)

            if t["strength"] > 0:
                strength_text = self.font_main.render(
                    str(t["strength"]), True, self.COLOR_TEXT
                )
                text_rect = strength_text.get_rect(center=rect.center)
                self.screen.blit(strength_text, text_rect)

    def _render_ui(self):
        # Draw player hand
        card_w, card_h = 80, 110
        hand_y = self.screen_height - card_h - 10
        if not self.player_hand:
            total_hand_w = 0
        else:
            total_hand_w = len(self.player_hand) * (card_w + 10) - 10
        hand_x_start = (self.screen_width - total_hand_w) // 2

        for i, card in enumerate(self.player_hand):
            card_x = hand_x_start + i * (card_w + 10)
            rect = pygame.Rect(card_x, hand_y, card_w, card_h)

            is_selected = i == self.player_card_cursor

            # Draw selector
            if is_selected:
                sel_rect = rect.inflate(10, 10)
                pulse = abs(math.sin(pygame.time.get_ticks() * 0.005))
                sel_color = (*self.COLOR_CURSOR, 100 + pulse * 100)
                pygame.draw.rect(self.screen, sel_color, sel_rect, border_radius=8)

            # Card body
            card_type = card["name"] # FIX: Use 'name' field directly
            border_color = self.CARD_COLORS.get(card_type, self.COLOR_GRID)
            pygame.draw.rect(self.screen, self.COLOR_GRID, rect, border_radius=8)
            pygame.draw.rect(
                self.screen, border_color, rect.inflate(-4, -4), border_radius=6
            )
            pygame.draw.rect(
                self.screen, self.COLOR_BG, rect.inflate(-8, -8), border_radius=4
            )

            # Card text
            strength_text = self.font_main.render(
                str(card["strength"]), True, self.COLOR_TEXT
            )
            self.screen.blit(
                strength_text,
                strength_text.get_rect(center=(rect.centerx, rect.centery - 10)),
            )

            name_text = self.font_card_small.render(card_type, True, self.COLOR_TEXT)
            self.screen.blit(
                name_text, name_text.get_rect(center=(rect.centerx, rect.centery + 25))
            )

        # Draw Score and Turn
        score_text = self.font_main.render(
            f"Territories: {self.score}", True, self.COLOR_TEXT
        )
        self.screen.blit(score_text, (10, 10))

        turn_text = self.font_main.render(
            f"Turn: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT
        )
        self.screen.blit(
            turn_text, turn_text.get_rect(topright=(self.screen_width - 10, 10))
        )

        # Draw feedback message
        if self.feedback_timer > 0:
            self.feedback_timer -= 1
            alpha = min(255, self.feedback_timer * 5)
            feedback_surf = self.font_main.render(
                self.feedback_message, True, self.feedback_color
            )
            feedback_surf.set_alpha(alpha)
            self.screen.blit(
                feedback_surf,
                feedback_surf.get_rect(center=(self.screen_width / 2, 40)),
            )

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_territories": self.score,
            "opponent_territories": self.opponent_score,
            "player_hand_size": len(self.player_hand),
            "game_over": self.game_over,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation:
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8

        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Setup Pygame window for human play
    pygame.display.set_caption("Card Kingdom Conquest")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    clock = pygame.time.Clock()

    running = True
    while running:
        # --- Create an action from keyboard input ---
        # Default action is no-op
        movement, space, shift = 0, 0, 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                elif event.key == pygame.K_SPACE:
                    space = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift = 1
                elif event.key == pygame.K_r:  # Reset on 'r' key
                    print("--- RESETTING ENVIRONMENT ---")
                    obs, info = env.reset()
                elif event.key == pygame.K_ESCAPE:
                    running = False

        action = np.array([movement, space, shift])

        # --- Step the environment ---
        # For turn-based games, we only step when an action is taken
        if not np.array_equal(action, [0, 0, 0]):
            obs, reward, terminated, truncated, info = env.step(action)
            print(
                f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}"
            )
            if terminated:
                print("--- GAME OVER ---")
                # Optional: auto-reset after a short delay
                # pygame.time.wait(2000)
                # obs, info = env.reset()

        # --- Render the current state ---
        # Get the observation from the environment (which is already a rendered frame)
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))  # Transpose back for pygame
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))

        pygame.display.flip()
        clock.tick(30)  # Limit to 30 FPS

    env.close()