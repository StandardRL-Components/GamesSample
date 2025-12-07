import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import numpy as np
import os
import pygame


# Set headless mode for Pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A simulation game where the player cultivates a beneficial microorganism colony.
    The goal is to help the beneficial colony take over 80% of the grid.
    The player must fight against harmful bacteria that spread and convert tiles.
    The player uses cards with special abilities to terraform, purify, and seed the grid.
    """

    metadata = {"render_modes": ["rgb_array"]}

    # --- User-facing metadata ---
    game_description = (
        "Cultivate a beneficial microorganism colony to dominate the grid while fighting off "
        "spreading harmful bacteria using special abilities."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor. Press space to use the selected ability card "
        "and shift to cycle through your hand."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 32, 20
    TILE_WIDTH = SCREEN_WIDTH // GRID_COLS
    TILE_HEIGHT = SCREEN_HEIGHT // GRID_ROWS
    UI_HEIGHT = 80
    GAME_AREA_HEIGHT = SCREEN_HEIGHT - UI_HEIGHT

    # Colors
    COLOR_BG = (10, 5, 20)
    COLOR_NEUTRAL = (40, 30, 60)
    COLOR_BENEFICIAL = (20, 80, 50)
    COLOR_BENEFICIAL_GLOW = (50, 220, 150)
    COLOR_HARMFUL = (100, 20, 30)
    COLOR_HARMFUL_GLOW = (255, 50, 80)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_TEXT = (220, 220, 240)
    COLOR_UI_BG = (20, 15, 35)
    COLOR_UI_BORDER = (80, 70, 110)
    COLOR_BAR_GOOD = (50, 220, 150)
    COLOR_BAR_BAD = (255, 50, 80)
    COLOR_BAR_EMPTY = (40, 30, 60)

    # Game parameters
    MAX_STEPS = 2000
    INITIAL_HARMFUL_BACTERIA = 3
    VICTORY_THRESHOLD = 0.8
    FAILURE_THRESHOLD = 0.5
    HAND_SIZE = 3

    # Tile types
    TILE_TYPE_NEUTRAL = 0
    TILE_TYPE_BENEFICIAL = 1
    TILE_TYPE_HARMFUL = 2

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 14)
        self.font_medium = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        # --- Game State Variables ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.grid = None
        self.harmful_bacteria = []
        self.harmful_bacteria_speed = 0.0
        self.cursor_pos = None
        self.hand = []
        self.selected_card_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []
        self.background_shapes = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False

        # Initialize grid
        self.grid = np.full((self.GRID_COLS, self.GRID_ROWS), self.TILE_TYPE_NEUTRAL, dtype=np.uint8)

        # Initial beneficial patch
        start_x, start_y = self.GRID_COLS // 4, self.GRID_ROWS // 2
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if 0 <= start_x + dx < self.GRID_COLS and 0 <= start_y + dy < self.GRID_ROWS:
                    self.grid[start_x + dx, start_y + dy] = self.TILE_TYPE_BENEFICIAL

        # Initialize harmful bacteria
        self.harmful_bacteria_speed = 0.5
        self.harmful_bacteria = []
        for _ in range(self.INITIAL_HARMFUL_BACTERIA):
            self._spawn_harmful_bacterium()

        # Player state
        self.cursor_pos = pygame.Vector2(self.GRID_COLS // 2, self.GRID_ROWS // 2)
        self.selected_card_idx = 0
        self.prev_space_held = True  # Prevent action on first frame
        self.prev_shift_held = True

        # Cards
        self._create_card_deck()
        self.hand = [self._draw_card() for _ in range(self.HAND_SIZE)]

        self.particles = []
        self._generate_background()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        self.steps += 1

        # --- 1. Handle Player Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held

        # Move cursor
        if movement == 1: self.cursor_pos.y = max(0, self.cursor_pos.y - 1)  # Up
        elif movement == 2: self.cursor_pos.y = min(self.GRID_ROWS - 1, self.cursor_pos.y + 1)  # Down
        elif movement == 3: self.cursor_pos.x = max(0, self.cursor_pos.x - 1)  # Left
        elif movement == 4: self.cursor_pos.x = min(self.GRID_COLS - 1, self.cursor_pos.x + 1)  # Right

        # Cycle card
        if shift_pressed and self.hand:
            self.selected_card_idx = (self.selected_card_idx + 1) % len(self.hand)

        # Play card
        if space_pressed and self.hand:
            card = self.hand[self.selected_card_idx]
            play_reward = self._apply_card_effect(card, self.cursor_pos)
            reward += play_reward
            reward += 1.0  # Reward for taking an action

            self.hand.pop(self.selected_card_idx)
            self.hand.append(self._draw_card())
            if self.selected_card_idx >= len(self.hand):
                self.selected_card_idx = max(0, len(self.hand) - 1)

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- 2. Update Game World ---
        old_beneficial_pct, old_harmful_pct = self._get_biome_balance()

        self._update_difficulty()
        self._update_harmful_bacteria()
        self._update_beneficial_spread()

        # --- 3. Calculate Rewards & Termination ---
        new_beneficial_pct, new_harmful_pct = self._get_biome_balance()

        if new_beneficial_pct > old_beneficial_pct:
            reward += (new_beneficial_pct - old_beneficial_pct) * 10.0
        if new_harmful_pct > old_harmful_pct:
            reward -= (new_harmful_pct - old_harmful_pct) * 20.0

        self.score += reward
        terminated = False
        truncated = False

        if new_beneficial_pct >= self.VICTORY_THRESHOLD:
            terminated = True
            self.score += 100
            reward += 100
        elif new_harmful_pct > self.FAILURE_THRESHOLD:
            terminated = True
            self.score -= 100
            reward -= 100

        if self.steps >= self.MAX_STEPS:
            truncated = True

        self.game_over = terminated or truncated

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        ben_pct, harm_pct = self._get_biome_balance()
        return {
            "score": self.score,
            "steps": self.steps,
            "beneficial_percentage": ben_pct,
            "harmful_percentage": harm_pct,
            "colony_size": np.sum(self.grid == self.TILE_TYPE_BENEFICIAL),
        }

    def _spawn_harmful_bacterium(self):
        x = self.np_random.integers(self.GRID_COLS * 3 // 4, self.GRID_COLS)
        y = self.np_random.integers(0, self.GRID_ROWS)
        path_type = self.np_random.choice(["horizontal", "vertical", "box"])
        path = []
        if path_type == "horizontal":
            x1 = self.np_random.integers(self.GRID_COLS // 2, self.GRID_COLS)
            x2 = self.np_random.integers(self.GRID_COLS // 2, self.GRID_COLS)
            y_path = self.np_random.integers(0, self.GRID_ROWS)
            path = [pygame.Vector2(x1, y_path), pygame.Vector2(x2, y_path)]
        elif path_type == "vertical":
            y1 = self.np_random.integers(0, self.GRID_ROWS)
            y2 = self.np_random.integers(0, self.GRID_ROWS)
            x_path = self.np_random.integers(self.GRID_COLS // 2, self.GRID_COLS)
            path = [pygame.Vector2(x_path, y1), pygame.Vector2(x_path, y2)]
        else:  # box
            x1 = self.np_random.integers(self.GRID_COLS // 2, self.GRID_COLS - 2)
            y1 = self.np_random.integers(0, self.GRID_ROWS - 2)
            path = [pygame.Vector2(x1, y1), pygame.Vector2(x1 + 2, y1), pygame.Vector2(x1 + 2, y1 + 2), pygame.Vector2(x1, y1 + 2)]
        if not path: path = [pygame.Vector2(x, y), pygame.Vector2(x, y)]
        self.harmful_bacteria.append({"pos": pygame.Vector2(x, y), "path": path, "path_idx": 0, "progress": 0.0})

    def _create_card_deck(self):
        self.deck = [
            {"name": "Terraform", "desc": "Convert a 3x3 area to beneficial.", "cost": 0, "type": "terraform"},
            {"name": "Purify", "desc": "Cleanse a harmful tile.", "cost": 0, "type": "purify"},
            {"name": "Seed Bomb", "desc": "Place a new beneficial tile.", "cost": 0, "type": "seed"},
            {"name": "Barrier", "desc": "Make a tile immune to harm for 50 steps.", "cost": 0, "type": "barrier"},
        ]

    def _draw_card(self):
        idx = self.np_random.integers(0, len(self.deck))
        return self.deck[idx].copy()

    def _apply_card_effect(self, card, pos):
        reward = 0
        cx, cy = int(pos.x), int(pos.y)
        if card["type"] == "terraform":
            self._create_particles(pos, self.COLOR_BENEFICIAL_GLOW, 30, 3.0)
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS:
                        if self.grid[nx, ny] == self.TILE_TYPE_NEUTRAL:
                            self.grid[nx, ny] = self.TILE_TYPE_BENEFICIAL
                            reward += 5.0
        elif card["type"] == "purify":
            self._create_particles(pos, (200, 200, 255), 15, 2.0)
            if self.grid[cx, cy] == self.TILE_TYPE_HARMFUL:
                self.grid[cx, cy] = self.TILE_TYPE_NEUTRAL
                reward += 5.0
        elif card["type"] == "seed":
            self._create_particles(pos, self.COLOR_BENEFICIAL, 10, 1.0)
            if self.grid[cx, cy] == self.TILE_TYPE_NEUTRAL:
                self.grid[cx, cy] = self.TILE_TYPE_BENEFICIAL
                reward += 5.0
        return reward

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 200 == 0:
            self.harmful_bacteria_speed += 0.05

    def _update_harmful_bacteria(self):
        for bacterium in self.harmful_bacteria:
            if not bacterium["path"]: continue
            bacterium["progress"] += self.harmful_bacteria_speed / 30.0
            current_target_idx = bacterium["path_idx"]
            next_target_idx = (current_target_idx + 1) % len(bacterium["path"])
            start_pos = bacterium["path"][current_target_idx]
            end_pos = bacterium["path"][next_target_idx]
            
            t = min(1.0, bacterium["progress"])
            bacterium["pos"] = start_pos.lerp(end_pos, t)
            
            if bacterium["progress"] >= 1.0:
                bacterium["progress"] -= 1.0
                bacterium["path_idx"] = next_target_idx
                bacterium["pos"] = end_pos
            
            gx, gy = int(bacterium["pos"].x), int(bacterium["pos"].y)
            if 0 <= gx < self.GRID_COLS and 0 <= gy < self.GRID_ROWS:
                if self.grid[gx, gy] != self.TILE_TYPE_HARMFUL:
                    self.grid[gx, gy] = self.TILE_TYPE_HARMFUL

    def _update_beneficial_spread(self):
        if self.steps % 10 != 0: return
        newly_beneficial = []
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS):
                if self.grid[x, y] == self.TILE_TYPE_BENEFICIAL:
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS:
                            if self.grid[nx, ny] == self.TILE_TYPE_NEUTRAL:
                                if self.np_random.random() < 0.25:
                                    newly_beneficial.append((nx, ny))
        for x, y in newly_beneficial:
            self.grid[x, y] = self.TILE_TYPE_BENEFICIAL

    def _get_biome_balance(self):
        total_tiles = self.GRID_COLS * self.GRID_ROWS
        if total_tiles == 0: return 0.0, 0.0
        beneficial_count = np.sum(self.grid == self.TILE_TYPE_BENEFICIAL)
        harmful_count = np.sum(self.grid == self.TILE_TYPE_HARMFUL)
        return beneficial_count / total_tiles, harmful_count / total_tiles

    def _render_all(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_grid()
        self._render_harmful_bacteria()
        self._update_and_render_particles()
        self._render_cursor()
        self._render_ui()

    def _generate_background(self):
        self.background_shapes = []
        for _ in range(30):
            r = self.np_random.integers(50, 150)
            pos = (self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.GAME_AREA_HEIGHT))
            color = (self.np_random.integers(15, 30), self.np_random.integers(10, 25), self.np_random.integers(25, 40))
            self.background_shapes.append({"pos": pos, "radius": r, "color": color})

    def _render_background(self):
        for shape in self.background_shapes:
            pygame.gfxdraw.filled_circle(self.screen, shape["pos"][0], shape["pos"][1], shape["radius"], shape["color"])

    def _render_grid(self):
        pulse = (math.sin(self.steps * 0.1) + 1) / 2
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS):
                rect = pygame.Rect(x * self.TILE_WIDTH, y * self.TILE_HEIGHT, self.TILE_WIDTH, self.TILE_HEIGHT)
                tile_type = self.grid[x, y]
                color = self.COLOR_NEUTRAL
                if tile_type == self.TILE_TYPE_BENEFICIAL: color = self.COLOR_BENEFICIAL
                elif tile_type == self.TILE_TYPE_HARMFUL: color = self.COLOR_HARMFUL
                pygame.draw.rect(self.screen, color, rect)
                if tile_type == self.TILE_TYPE_BENEFICIAL:
                    glow_radius = int(self.TILE_WIDTH / 4 + pulse * 3)
                    pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, glow_radius, (*self.COLOR_BENEFICIAL_GLOW, 60))
                elif tile_type == self.TILE_TYPE_HARMFUL:
                    glow_radius = int(self.TILE_WIDTH / 3)
                    pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, glow_radius, (*self.COLOR_HARMFUL_GLOW, 80))

    def _render_harmful_bacteria(self):
        for bacterium in self.harmful_bacteria:
            pixel_pos = (int(bacterium["pos"].x * self.TILE_WIDTH + self.TILE_WIDTH / 2), int(bacterium["pos"].y * self.TILE_HEIGHT + self.TILE_HEIGHT / 2))
            radius = self.TILE_WIDTH / 2.5
            points = []
            for i in range(8):
                angle = i * math.pi / 4 + self.steps * 0.05
                r = radius if i % 2 == 0 else radius * 0.6
                points.append((pixel_pos[0] + r * math.cos(angle), pixel_pos[1] + r * math.sin(angle)))
            pygame.draw.polygon(self.screen, self.COLOR_HARMFUL_GLOW, points)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_HARMFUL_GLOW)

    def _render_cursor(self):
        if self.game_over: return
        cx, cy = int(self.cursor_pos.x), int(self.cursor_pos.y)
        rect = pygame.Rect(cx * self.TILE_WIDTH, cy * self.TILE_HEIGHT, self.TILE_WIDTH, self.TILE_HEIGHT)
        for i in range(4):
            glow_rect = rect.inflate(i * 2, i * 2)
            alpha = 100 - i * 25
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, (*self.COLOR_CURSOR, alpha), s.get_rect(), border_radius=3)
            self.screen.blit(s, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2, border_radius=3)

    def _create_particles(self, grid_pos, color, count, max_speed):
        pixel_pos = ((grid_pos.x + 0.5) * self.TILE_WIDTH, (grid_pos.y + 0.5) * self.TILE_HEIGHT)
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * max_speed
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({"pos": pygame.Vector2(pixel_pos), "vel": vel, "lifespan": self.np_random.integers(20, 40), "color": color})

    def _update_and_render_particles(self):
        active_particles = []
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] > 0:
                active_particles.append(p)
                alpha = int(255 * (p["lifespan"] / 40))
                s = pygame.Surface((4, 4), pygame.SRCALPHA)
                pygame.draw.circle(s, (*p["color"], alpha), (2, 2), 2)
                self.screen.blit(s, (p["pos"].x - 2, p["pos"].y - 2))
        self.particles = active_particles

    def _render_ui(self):
        ui_rect = pygame.Rect(0, self.GAME_AREA_HEIGHT, self.SCREEN_WIDTH, self.UI_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_rect)
        pygame.draw.line(self.screen, self.COLOR_UI_BORDER, (0, self.GAME_AREA_HEIGHT), (self.SCREEN_WIDTH, self.GAME_AREA_HEIGHT), 2)
        ben_pct, harm_pct = self._get_biome_balance()
        bar_width = 200
        bar_x, bar_y = 10, self.GAME_AREA_HEIGHT + 15
        ben_width = int(bar_width * ben_pct)
        harm_width = int(bar_width * harm_pct)
        pygame.draw.rect(self.screen, self.COLOR_BAR_EMPTY, (bar_x, bar_y, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_BAR_GOOD, (bar_x, bar_y, ben_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_BAR_BAD, (bar_x + bar_width - harm_width, bar_y, harm_width, 20))
        balance_text = self.font_medium.render("Biome Balance", True, self.COLOR_TEXT)
        self.screen.blit(balance_text, (bar_x, bar_y + 25))
        score_text = self.font_medium.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (230, self.GAME_AREA_HEIGHT + 15))
        steps_text = self.font_medium.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (230, self.GAME_AREA_HEIGHT + 40))
        card_area_x = 380
        for i, card in enumerate(self.hand):
            is_selected = i == self.selected_card_idx
            card_rect = pygame.Rect(card_area_x + i * 85, self.GAME_AREA_HEIGHT + 10, 80, self.UI_HEIGHT - 20)
            border_color = self.COLOR_CURSOR if is_selected else self.COLOR_UI_BORDER
            bg_color = (40, 35, 60) if is_selected else self.COLOR_UI_BG
            pygame.draw.rect(self.screen, bg_color, card_rect, border_radius=5)
            pygame.draw.rect(self.screen, border_color, card_rect, 2, border_radius=5)
            card_name = self.font_medium.render(card["name"], True, self.COLOR_TEXT)
            self.screen.blit(card_name, (card_rect.x + 5, card_rect.y + 5))
            line_y = card_rect.y + 25
            current_line = ""
            for word in card["desc"].split(" "):
                test_line = current_line + word + " "
                if self.font_small.size(test_line)[0] > card_rect.width - 10:
                    line_surf = self.font_small.render(current_line, True, self.COLOR_TEXT)
                    self.screen.blit(line_surf, (card_rect.x + 5, line_y))
                    line_y += 12
                    current_line = word + " "
                else:
                    current_line = test_line
            line_surf = self.font_small.render(current_line, True, self.COLOR_TEXT)
            self.screen.blit(line_surf, (card_rect.x + 5, line_y))

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to run the file directly to play the game
    # It will create a window and render the game, listening for keyboard input.
    os.environ["SDL_VIDEODRIVER"] = "x11"  # Or "windows", "macOS"
    env = GameEnv(render_mode="human_playable")
    obs, info = env.reset()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Gut Biome Simulator")
    clock = pygame.time.Clock()
    done = False
    total_reward = 0
    while not done:
        action = [0, 0, 0]
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: action[0] = 4
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            done = True
        clock.tick(30)
    pygame.quit()