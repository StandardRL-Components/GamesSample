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
        "Controls: Arrows to move cursor. Space to place selected tower. Shift to cycle tower types."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of invading aliens. Place towers strategically on the grid to survive."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.CELL_SIZE = 30
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_SIZE * self.CELL_SIZE) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_SIZE * self.CELL_SIZE) // 2 + 20
        self.MAX_STEPS = 2000
        self.MAX_WAVES = 10

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_PATH = (60, 60, 80)
        self.COLOR_BASE = (255, 200, 0)
        self.COLOR_ALIEN = (220, 50, 50)
        self.COLOR_TEXT = (220, 220, 240)

        self.TOWER_SPECS = [
            {
                "name": "Machine Gun", "color": (0, 255, 150), "range": 80,
                "damage": 3, "cooldown": 15, "cost": 0
            },
            {
                "name": "Sniper", "color": (0, 200, 255), "range": 160,
                "damage": 15, "cooldown": 60, "cost": 0
            }
        ]

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        # Path definition (list of grid cell coordinates)
        self.path_coords = [
            (-1, 5), (0, 5), (1, 5), (2, 5), (2, 4), (2, 3), (3, 3), (4, 3), (5, 3),
            (5, 4), (5, 5), (5, 6), (5, 7), (6, 7), (7, 7), (7, 6), (7, 5), (7, 4),
            (7, 3), (7, 2), (8, 2), (9, 2), (10, 2)
        ]
        self.path_pixels = [self._grid_to_pixel(x, y) for x, y in self.path_coords]

        # Initialize state variables
        self.np_random = None
        self.grid = None
        self.aliens = None
        self.towers = None
        self.particles = None
        self.projectiles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.game_won = None
        self.base_health = None
        self.wave_number = None
        self.cursor_pos = None
        self.selected_tower_type = None
        self.prev_space_held = None
        self.prev_shift_held = None

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()

        # Initialize all game state
        self.grid = [[0 for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        self.aliens = []
        self.towers = []
        self.particles = []
        self.projectiles = []

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.base_health = 100
        self.wave_number = 0

        self.cursor_pos = (self.GRID_SIZE // 2, self.GRID_SIZE // 2)
        self.selected_tower_type = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self._start_next_wave()

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01  # Small time penalty
        self.steps += 1

        # Unpack factorized action and handle logic
        self._handle_actions(action)
        reward += self._update_game_state()

        self._check_wave_completion()
        terminated = self._check_termination()

        if terminated:
            if self.game_won:
                reward += 100
            elif self.base_health <= 0:
                reward -= 100

        # Update previous button states for press detection
        self.prev_space_held = (action[1] == 1)
        self.prev_shift_held = (action[2] == 1)

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_actions(self, action):
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1

        # Move cursor
        if movement == 1: self.cursor_pos = (self.cursor_pos[0], max(0, self.cursor_pos[1] - 1))
        elif movement == 2: self.cursor_pos = (self.cursor_pos[0], min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1))
        elif movement == 3: self.cursor_pos = (max(0, self.cursor_pos[0] - 1), self.cursor_pos[1])
        elif movement == 4: self.cursor_pos = (min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1), self.cursor_pos[1])

        # Cycle tower type on press (0 -> 1 transition)
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
            # sfx: UI_CYCLE

        # Place tower on press
        if space_held and not self.prev_space_held:
            cx, cy = self.cursor_pos
            is_on_path = any(p == (cx, cy) for p in self.path_coords[1:-1])
            if self.grid[cy][cx] == 0 and not is_on_path:
                spec = self.TOWER_SPECS[self.selected_tower_type]
                px, py = self._grid_to_pixel(cx, cy)
                self.towers.append({
                    "x": px, "y": py, "grid_x": cx, "grid_y": cy,
                    "spec": spec, "cooldown": 0, "type": self.selected_tower_type + 1
                })
                self.grid[cy][cx] = self.selected_tower_type + 1
                # sfx: TOWER_PLACE

    def _update_game_state(self):
        step_reward = 0

        # Update towers
        for tower in self.towers:
            if tower["cooldown"] > 0:
                tower["cooldown"] -= 1
                continue

            target = None
            max_dist_on_path = -1
            for alien in self.aliens:
                dist = math.hypot(alien["x"] - tower["x"], alien["y"] - tower["y"])
                if dist <= tower["spec"]["range"]:
                    if alien["path_progress"] > max_dist_on_path:
                        max_dist_on_path = alien["path_progress"]
                        target = alien

            if target:
                # sfx: TOWER_SHOOT
                tower["cooldown"] = tower["spec"]["cooldown"]
                target["health"] -= tower["spec"]["damage"]
                step_reward += 1  # Reward for hitting
                self.projectiles.append(
                    {"start": (tower["x"], tower["y"]), "end": (target["x"], target["y"]), "color": tower["spec"]["color"]})

        # Update aliens
        aliens_to_remove = []
        for i, alien in enumerate(self.aliens):
            if alien["health"] <= 0:
                # sfx: ALIEN_EXPLODE
                self.score += 10
                step_reward += 10  # Reward for destroying
                self._create_explosion(alien["x"], alien["y"])
                aliens_to_remove.append(i)
                continue

            current_target_waypoint_idx = alien["target_waypoint"]
            target_px, target_py = self.path_pixels[current_target_waypoint_idx]

            dx, dy = target_px - alien["x"], target_py - alien["y"]
            dist_to_target = math.hypot(dx, dy)

            if dist_to_target < alien["speed"]:
                alien["target_waypoint"] += 1
                if alien["target_waypoint"] >= len(self.path_pixels):
                    self.base_health = max(0, self.base_health - alien["damage"])
                    aliens_to_remove.append(i)
                    self._create_explosion(alien["x"], alien["y"], self.COLOR_BASE)
                    # sfx: BASE_DAMAGE
                    continue
            else:
                alien["x"] += (dx / dist_to_target) * alien["speed"]
                alien["y"] += (dy / dist_to_target) * alien["speed"]
            alien["path_progress"] += alien["speed"]

        for i in sorted(aliens_to_remove, reverse=True): del self.aliens[i]

        # Update particles and projectiles
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["life"] -= 1
            p["radius"] += p["expansion_rate"]
        self.projectiles.clear()

        return step_reward

    def _start_next_wave(self):
        self.wave_number += 1
        if self.wave_number > self.MAX_WAVES: return

        num_aliens = 2 + self.wave_number * 2
        alien_health = 10 + (self.wave_number - 1) * 5
        alien_speed = 0.8 + (self.wave_number - 1) * 0.1
        spawn_delay = 40

        start_x, start_y = self.path_pixels[0]
        for i in range(num_aliens):
            self.aliens.append({
                "x": start_x - i * spawn_delay, "y": start_y,
                "health": alien_health, "max_health": alien_health,
                "speed": alien_speed, "damage": 10,
                "target_waypoint": 1, "path_progress": -i * spawn_delay
            })

    def _check_wave_completion(self):
        if not self.aliens and not self.game_over:
            if self.wave_number >= self.MAX_WAVES:
                self.game_won = True
                self.game_over = True
            else:
                self._start_next_wave()

    def _check_termination(self):
        if self.base_health <= 0: self.game_over = True
        if self.steps >= self.MAX_STEPS: self.game_over = True
        self.base_health = min(100, self.base_health)  # Assert health doesn't exceed 100
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        if len(self.path_pixels) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_pixels, width=self.CELL_SIZE)

        for x in range(self.GRID_SIZE + 1):
            s = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y)
            e = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_SIZE * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, s, e)
        for y in range(self.GRID_SIZE + 1):
            s = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            e = (self.GRID_OFFSET_X + self.GRID_SIZE * self.CELL_SIZE, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, s, e)

        base_x, base_y = self.path_pixels[-1]
        pygame.draw.rect(self.screen, self.COLOR_BASE,
                         (base_x - self.CELL_SIZE // 2, base_y - self.CELL_SIZE // 2, self.CELL_SIZE, self.CELL_SIZE))

        for tower in self.towers:
            spec, color = tower["spec"], tower["spec"]["color"]
            if spec['name'] == "Machine Gun":
                pygame.draw.rect(self.screen, color, (tower["x"] - 10, tower["y"] - 10, 20, 20))
                pygame.draw.rect(self.screen, self.COLOR_BG, (tower["x"] - 6, tower["y"] - 6, 12, 12))
            elif spec['name'] == "Sniper":
                pts = [(tower['x'], tower['y'] - 10), (tower['x'] - 10, tower['y'] + 8), (tower['x'] + 10, tower['y'] + 8)]
                pygame.draw.polygon(self.screen, color, pts)

        for alien in self.aliens:
            pos = (int(alien["x"]), int(alien["y"]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_ALIEN)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.COLOR_ALIEN)
            if alien["health"] < alien["max_health"]:
                hp = alien["health"] / alien["max_health"]
                pygame.draw.rect(self.screen, (100, 0, 0), (pos[0] - 10, pos[1] - 15, 20, 3))
                pygame.draw.rect(self.screen, (0, 255, 0), (pos[0] - 10, pos[1] - 15, 20 * hp, 3))

        for p in self.projectiles: pygame.draw.aaline(self.screen, p["color"], p["start"], p["end"], 2)
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            # p['radius'] is a float, but Pygame drawing functions require integers.
            int_radius = int(p["radius"])
            if int_radius <= 0:
                continue
            s = pygame.Surface((int_radius * 2, int_radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(s, int_radius, int_radius, int_radius, p["color"] + (alpha,))
            self.screen.blit(s, (int(p["x"] - int_radius), int(p["y"] - int_radius)))

        cursor_px, cursor_py = self._grid_to_pixel(self.cursor_pos[0], self.cursor_pos[1])
        tower_range = self.TOWER_SPECS[self.selected_tower_type]["range"]
        range_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(range_surface, cursor_px, cursor_py, tower_range, (255, 255, 255, 30))
        pygame.gfxdraw.aacircle(range_surface, cursor_px, cursor_py, tower_range, (255, 255, 255, 80))
        self.screen.blit(range_surface, (0, 0))

        cursor_rect = pygame.Rect(cursor_px - self.CELL_SIZE // 2, cursor_py - self.CELL_SIZE // 2, self.CELL_SIZE,
                                  self.CELL_SIZE)
        pygame.draw.rect(self.screen, (255, 255, 255), cursor_rect, 2)

    def _render_ui(self):
        pygame.draw.rect(self.screen, (10, 10, 20), (0, 0, self.WIDTH, 40))

        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 8))

        wave_text = self.font_large.render(f"WAVE: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, wave_text.get_rect(center=(self.WIDTH // 2, 20)))

        self.screen.blit(self.font_small.render("BASE HEALTH", True, self.COLOR_TEXT), (self.WIDTH - 160, 5))
        pygame.draw.rect(self.screen, (100, 0, 0), (self.WIDTH - 160, 22, 150, 12))
        pygame.draw.rect(self.screen, self.COLOR_BASE, (self.WIDTH - 160, 22, 150 * (self.base_health / 100), 12))

        tower_spec = self.TOWER_SPECS[self.selected_tower_type]
        tower_name_text = self.font_small.render(f"Selected: {tower_spec['name']}", True, tower_spec['color'])
        self.screen.blit(tower_name_text, (10, self.HEIGHT - 25))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg, color = ("YOU WIN!", (0, 255, 0)) if self.game_won else ("GAME OVER", (255, 0, 0))
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2)))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _grid_to_pixel(self, grid_x, grid_y):
        px = self.GRID_OFFSET_X + grid_x * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.GRID_OFFSET_Y + grid_y * self.CELL_SIZE + self.CELL_SIZE // 2
        return px, py

    def _create_explosion(self, x, y, color=(255, 255, 200)):
        self.particles.append({
            "x": x, "y": y, "radius": 5, "life": 20, "max_life": 20,
            "color": color, "expansion_rate": 1.5
        })

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False

    # For human play, we need a real display
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.display.init()
    pygame.font.init()

    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    # Re-initialize fonts after changing display mode
    env.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
    env.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

    while not done:
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        action = [movement, 1 if keys[pygame.K_SPACE] else 0,
                  1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0]
        action_taken = any(action)

        for event in pygame.event.get():
            if event.type == pygame.QUIT: done = True
            if event.type == pygame.KEYDOWN: action_taken = True

        if action_taken or env.game_over:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30)

    env.close()