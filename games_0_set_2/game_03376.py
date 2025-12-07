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
        "Controls: Arrow keys to move cursor. Space to place tower. Shift to cycle tower type."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Side-view tower defense. Place towers to defend your base from waves of enemies."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 30 * 60  # 60 seconds at 30fps

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_PATH = (40, 50, 60)
    COLOR_BASE = (60, 180, 75)
    COLOR_ENEMY = (210, 60, 60)
    COLOR_UI_TEXT = (230, 230, 230)
    COLOR_UI_HEALTH = (60, 180, 75)
    COLOR_UI_HEALTH_BG = (90, 30, 30)
    COLOR_CURSOR_VALID = (255, 255, 255, 100)
    COLOR_CURSOR_INVALID = (255, 0, 0, 100)

    # Game parameters
    BASE_HEALTH_MAX = 100
    STARTING_CURRENCY = 250
    WAVE_COUNT_TOTAL = 15
    WAVE_COOLDOWN = 5 * FPS  # 5 seconds
    CURSOR_SPEED = 6

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # Define tower types
        self.TOWER_TYPES = [
            {
                "name": "Gatling", "cost": 100, "range": 80, "damage": 4,
                "fire_rate": 0.2 * self.FPS, "color": (0, 150, 255), "projectile_speed": 10
            },
            {
                "name": "Cannon", "cost": 250, "range": 120, "damage": 25,
                "fire_rate": 1.5 * self.FPS, "color": (255, 150, 0), "projectile_speed": 7
            },
        ]

        # Define enemy path
        self.path = [pygame.Vector2(p) for p in [
            (-20, 300), (80, 300), (80, 150), (250, 150), (250, 300),
            (450, 300), (450, 100), (580, 100), (580, 220), (self.SCREEN_WIDTH + 20, 220)
        ]]
        self.base_pos = pygame.Vector2(self.SCREEN_WIDTH - 20, 220)

        # Initialize state variables (will be properly set in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.reward_this_step = 0
        self.base_health = 0
        self.currency = 0
        self.current_wave = 0
        self.wave_in_progress = False
        self.wave_timer = 0
        self.enemies = []
        self.enemies_to_spawn = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = pygame.Vector2(0, 0)
        self.selected_tower_type_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        
        # self.reset() is called by the environment wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.reward_this_step = 0

        self.base_health = self.BASE_HEALTH_MAX
        self.currency = self.STARTING_CURRENCY

        self.current_wave = 0
        self.wave_in_progress = False
        self.wave_timer = self.WAVE_COOLDOWN // 2  # Start first wave sooner

        self.enemies = []
        self.enemies_to_spawn = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = pygame.Vector2(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        self.selected_tower_type_idx = 0
        self.last_space_held = False
        self.last_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = 0
        self.steps += 1
        terminated = False

        if not self.game_over:
            self._handle_input(action)
            self._update_game_state()

        terminated = self._check_termination()
        if terminated and not self.game_over:
            if self.game_won:
                self.reward_this_step += 100
                self.score += 1000
            else:  # Lost
                self.reward_this_step -= 100
            self.game_over = True

        self.score += self.reward_this_step

        return (
            self._get_observation(),
            float(self.reward_this_step),
            terminated,
            self.steps >= self.MAX_STEPS,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Movement
        if movement == 1: self.cursor_pos.y -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos.y += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos.x -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos.x += self.CURSOR_SPEED
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.SCREEN_WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.SCREEN_HEIGHT)

        # Cycle tower type
        if shift_held and not self.last_shift_held:
            self.selected_tower_type_idx = (self.selected_tower_type_idx + 1) % len(self.TOWER_TYPES)

        # Place tower
        if space_held and not self.last_space_held and self._is_valid_placement(self.cursor_pos):
            tower_type = self.TOWER_TYPES[self.selected_tower_type_idx]
            if self.currency >= tower_type["cost"]:
                self.currency -= tower_type["cost"]
                self.towers.append({
                    "pos": pygame.Vector2(self.cursor_pos),
                    "type_idx": self.selected_tower_type_idx,
                    "cooldown": 0,
                })
                # sfx: place_tower.wav
                self._create_particles(self.cursor_pos, 20, tower_type["color"], 2, 4)

        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _update_game_state(self):
        self._update_waves()
        self._update_towers()
        self._update_projectiles()
        self._update_enemies()
        self._update_particles()

    def _update_waves(self):
        if not self.wave_in_progress and self.current_wave < self.WAVE_COUNT_TOTAL:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self._start_next_wave()

        if self.wave_in_progress and not self.enemies and not self.enemies_to_spawn:
            self.wave_in_progress = False
            self.wave_timer = self.WAVE_COOLDOWN
            self.reward_this_step += 1
            self.currency += 100 + self.current_wave * 10

        if self.enemies_to_spawn and self.steps % 15 == 0:  # Spawn one enemy every 0.5s
            self.enemies.append(self.enemies_to_spawn.pop(0))

    def _start_next_wave(self):
        self.current_wave += 1
        self.wave_in_progress = True

        num_enemies = 3 + self.current_wave * 2
        base_health = 20 * (1.05 ** (self.current_wave - 1))
        base_speed = 1.0 * (1.02 ** (self.current_wave - 1))

        for _ in range(num_enemies):
            self.enemies_to_spawn.append({
                "pos": pygame.Vector2(self.path[0]),
                "health": base_health,
                "max_health": base_health,
                "speed": base_speed + self.np_random.uniform(-0.1, 0.1),
                "path_idx": 1,
                "value": 5 + self.current_wave,
            })

        if self.current_wave == self.WAVE_COUNT_TOTAL and not self.enemies and not self.enemies_to_spawn:
            self.game_won = True

    def _update_towers(self):
        for tower in self.towers:
            tower_spec = self.TOWER_TYPES[tower["type_idx"]]
            if tower["cooldown"] > 0:
                tower["cooldown"] -= 1
                continue

            target = None
            min_dist = tower_spec["range"]
            for enemy in self.enemies:
                dist = tower["pos"].distance_to(enemy["pos"])
                if dist < min_dist:
                    min_dist = dist
                    target = enemy

            if target:
                tower["cooldown"] = tower_spec["fire_rate"]
                self.projectiles.append({
                    "pos": pygame.Vector2(tower["pos"]),
                    "target": target,
                    "speed": tower_spec["projectile_speed"],
                    "damage": tower_spec["damage"],
                    "color": (255, 255, 100)
                })
                # sfx: shoot.wav

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            if proj["target"] not in self.enemies:
                self.projectiles.remove(proj)
                continue

            direction = (proj["target"]["pos"] - proj["pos"]).normalize()
            proj["pos"] += direction * proj["speed"]

            if proj["pos"].distance_to(proj["target"]["pos"]) < 5:
                proj["target"]["health"] -= proj["damage"]
                self.projectiles.remove(proj)
                # sfx: hit.wav
                self._create_particles(proj["pos"], 5, proj["color"], 1, 2)

    def _update_enemies(self):
        for enemy in self.enemies[:]:
            if enemy["health"] <= 0:
                self.reward_this_step += 0.1
                self.currency += enemy["value"]
                self.enemies.remove(enemy)
                # sfx: enemy_die.wav
                self._create_particles(enemy["pos"], 15, self.COLOR_ENEMY, 1, 3)
                continue

            target_pos = self.path[enemy["path_idx"]]
            direction = (target_pos - enemy["pos"])
            dist = direction.length()

            if dist < enemy["speed"]:
                enemy["path_idx"] += 1
                if enemy["path_idx"] >= len(self.path):
                    self.base_health = max(0, self.base_health - 10)
                    self.reward_this_step -= 1
                    self.enemies.remove(enemy)
                    # sfx: base_damage.wav
                    self._create_particles(self.base_pos, 30, (255, 100, 100), 3, 5)
                    continue
            else:
                enemy["pos"] += direction.normalize() * enemy["speed"]

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifetime"] -= 1
            p["radius"] -= 0.05
            if p["lifetime"] <= 0 or p["radius"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.base_health <= 0:
            return True
        if self.current_wave > self.WAVE_COUNT_TOTAL and not self.enemies and not self.enemies_to_spawn:
            self.game_won = True
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Path
        for i in range(len(self.path) - 1):
            pygame.draw.line(self.screen, self.COLOR_PATH, self.path[i], self.path[i + 1], 40)

        # Base
        pygame.draw.circle(self.screen, self.COLOR_BASE, (int(self.base_pos.x), int(self.base_pos.y)), 15)

        # Towers
        for tower in self.towers:
            spec = self.TOWER_TYPES[tower["type_idx"]]
            pos = (int(tower["pos"].x), int(tower["pos"].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, spec["color"])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, spec["color"])

        # Enemies
        for enemy in self.enemies:
            pos = (int(enemy["pos"].x), int(enemy["pos"].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 6, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 6, self.COLOR_ENEMY)
            # Health bar
            health_pct = enemy["health"] / enemy["max_health"]
            bar_w = 12
            pygame.draw.rect(self.screen, (50, 0, 0), (pos[0] - bar_w / 2, pos[1] - 15, bar_w, 3))
            pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH, (pos[0] - bar_w / 2, pos[1] - 15, bar_w * health_pct, 3))

        # Projectiles
        for proj in self.projectiles:
            pos = (int(proj["pos"].x), int(proj["pos"].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, proj["color"])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, proj["color"])

        # Particles
        for p in self.particles:
            pos = (int(p["pos"].x), int(p["pos"].y))
            radius = int(p["radius"])
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, p["color"])

        # Cursor
        self._render_cursor()

    def _render_cursor(self):
        pos = (int(self.cursor_pos.x), int(self.cursor_pos.y))
        spec = self.TOWER_TYPES[self.selected_tower_type_idx]

        is_valid = self._is_valid_placement(self.cursor_pos)
        cursor_color = self.COLOR_CURSOR_VALID if is_valid else self.COLOR_CURSOR_INVALID

        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], spec["range"], cursor_color)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, cursor_color)

    def _render_ui(self):
        # Base Health
        health_text = self.font_small.render(f"Base HP: {self.base_health}/{self.BASE_HEALTH_MAX}", True,
                                             self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (10, 10))
        health_pct = self.base_health / self.BASE_HEALTH_MAX
        pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH_BG, (10, 30, 150, 10))
        pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH, (10, 30, 150 * health_pct, 10))

        # Currency
        currency_text = self.font_small.render(f"Funds: ${self.currency}", True, (255, 223, 0))
        self.screen.blit(currency_text, (10, 50))

        # Wave Info
        if self.wave_in_progress:
            wave_text_str = f"Wave: {self.current_wave}/{self.WAVE_COUNT_TOTAL}"
        else:
            if self.current_wave < self.WAVE_COUNT_TOTAL:
                next_wave_in = self.wave_timer / self.FPS
                wave_text_str = f"Next wave in: {next_wave_in:.1f}s"
            else:
                wave_text_str = "All waves cleared!"
        wave_text = self.font_small.render(wave_text_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))

        # Selected Tower
        spec = self.TOWER_TYPES[self.selected_tower_type_idx]
        tower_select_text = self.font_small.render(
            f"Selected: {spec['name']} (Cost: ${spec['cost']})", True, spec["color"]
        )
        self.screen.blit(tower_select_text, (10, self.SCREEN_HEIGHT - 30))

        # Game Over / Win message
        if self.game_over:
            msg_str = "YOU WIN!" if self.game_won else "GAME OVER"
            color = self.COLOR_BASE if self.game_won else self.COLOR_ENEMY
            msg_surf = self.font_large.render(msg_str, True, color)
            pos = (self.SCREEN_WIDTH / 2 - msg_surf.get_width() / 2, self.SCREEN_HEIGHT / 2 - msg_surf.get_height() / 2)
            self.screen.blit(msg_surf, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "currency": self.currency,
            "base_health": self.base_health,
            "current_wave": self.current_wave,
        }

    def _is_valid_placement(self, pos):
        # Check cost
        if self.currency < self.TOWER_TYPES[self.selected_tower_type_idx]["cost"]:
            return False

        # Check proximity to path
        for i in range(len(self.path) - 1):
            p1 = self.path[i]
            p2 = self.path[i + 1]
            # Simple bounding box check on path segment
            min_x, max_x = min(p1.x, p2.x), max(p1.x, p2.x)
            min_y, max_y = min(p1.y, p2.y), max(p1.y, p2.y)
            if (min_x - 25 < pos.x < max_x + 25) and \
                    (min_y - 25 < pos.y < max_y + 25):
                return False

        # Check proximity to other towers
        for tower in self.towers:
            if pos.distance_to(tower["pos"]) < 20:
                return False

        return True

    def _create_particles(self, pos, count, color, min_speed, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(min_speed, max_speed)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "color": color,
                "lifetime": self.np_random.integers(10, 20),
                "radius": self.np_random.uniform(2, 4)
            })

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block is for human play and is not used by the evaluation system.
    # It allows you to test the environment with keyboard controls.
    # Set the video driver to something other than "dummy" to see the window.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset(seed=42)
    done = False

    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()

    action = np.array([0, 0, 0])  # Start with no-op

    print("Controls:")
    print("- Arrow Keys: Move cursor")
    print("- Space: Place tower")
    print("- L/R Shift: Cycle tower type")
    print("- Q or close window: Quit")

    while not done:
        # Human controls
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                done = True

        keys = pygame.key.get_pressed()

        # Reset action
        action.fill(0)

        # Movement
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4

        # Place tower
        if keys[pygame.K_SPACE]: action[1] = 1

        # Cycle tower
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if reward != 0:
            print(
                f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Base HP: {info['base_health']}")

        # Render to screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(GameEnv.FPS)

    print(f"Game Over. Final Score: {info['score']}")
    env.close()
    pygame.quit()