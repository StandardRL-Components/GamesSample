import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the placement cursor. Space to build a tower."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of invading enemies by strategically placing towers."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_SIZE = 12
    TILE_W_ISO, TILE_H_ISO = 40, 20
    ORIGIN_X, ORIGIN_Y = WIDTH // 2, 80
    MAX_STEPS = 3000
    MAX_WAVES = 10

    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_PATH = (40, 44, 56)
    COLOR_GRID = (55, 60, 74)
    COLOR_BASE = (50, 200, 100)
    COLOR_BASE_DMG = (255, 100, 100)
    COLOR_ENEMY = (220, 50, 50)
    COLOR_ENEMY_HEALTH = (100, 255, 100)
    COLOR_TOWER = (60, 150, 255)
    COLOR_PROJECTILE = (100, 200, 255)
    COLOR_CURSOR_VALID = (255, 255, 0, 150)
    COLOR_CURSOR_INVALID = (255, 0, 0, 150)
    COLOR_TEXT = (240, 240, 240)
    COLOR_GOLD = (255, 223, 0)

    # Game Parameters
    INITIAL_BASE_HEALTH = 100
    INITIAL_GOLD = 150
    TOWER_COST = 50
    TOWER_RANGE = 2.5  # In grid units
    TOWER_COOLDOWN = 30  # In frames
    ENEMY_DAMAGE = 10
    WAVE_INTERMISSION_TIME = 150  # 5 seconds at 30fps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        # FIX: Initialize a display to allow .convert_alpha() to work in headless mode.
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # Path definition (grid coordinates)
        self.path = [
            (-1, 5), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5),
            (4, 4), (4, 3), (4, 2),
            (5, 2), (6, 2), (7, 2), (8, 2),
            (8, 3), (8, 4), (8, 5), (8, 6), (8, 7),
            (7, 7), (6, 7), (5, 7), (4, 7), (3, 7), (2, 7),
            (2, 6), (2, 5), (2, 4), (2, 3), (2, 2), (2, 1),
            (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1),
        ]
        self.base_pos = self.path[-1]

        # Valid tower placement spots
        self.buildable_tiles = set()
        path_set = set(self.path)
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if (r, c) not in path_set:
                    self.buildable_tiles.add((r, c))

        # Initialize state variables
        # self.reset() is called by the wrapper/verifier, no need to call it here.

    def _world_to_screen(self, x, y):
        """Converts grid coordinates to screen coordinates."""
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_W_ISO / 2
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_H_ISO / 2
        return int(screen_x), int(screen_y)

    def _draw_iso_poly(self, surface, color, grid_pos, z=0):
        """Draws a single isometric tile polygon."""
        x, y = grid_pos
        sx, sy = self._world_to_screen(x, y)
        sy -= z * self.TILE_H_ISO
        points = [
            (sx, sy - self.TILE_H_ISO / 2),
            (sx + self.TILE_W_ISO / 2, sy),
            (sx, sy + self.TILE_H_ISO / 2),
            (sx - self.TILE_W_ISO / 2, sy),
        ]
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def _draw_iso_cube(self, surface, color, grid_pos, height=1.0):
        """Draws a 3D-like isometric cube."""
        x, y = grid_pos
        sx, sy = self._world_to_screen(x, y)

        top_color = color
        side_color_1 = tuple(max(0, c - 30) for c in color)
        side_color_2 = tuple(max(0, c - 60) for c in color)

        z_offset = height * self.TILE_H_ISO

        # Top face
        top_points = [
            (sx, sy - z_offset),
            (sx + self.TILE_W_ISO / 2, sy - z_offset + self.TILE_H_ISO / 2),
            (sx, sy - z_offset + self.TILE_H_ISO),
            (sx - self.TILE_W_ISO / 2, sy - z_offset + self.TILE_H_ISO / 2),
        ]
        pygame.draw.polygon(surface, top_color, top_points)

        # Right face
        right_points = [
            (sx, sy - z_offset + self.TILE_H_ISO),
            (sx + self.TILE_W_ISO / 2, sy - z_offset + self.TILE_H_ISO / 2),
            (sx + self.TILE_W_ISO / 2, sy + self.TILE_H_ISO / 2),
            (sx, sy + self.TILE_H_ISO),
        ]
        pygame.draw.polygon(surface, side_color_1, right_points)

        # Left face
        left_points = [
            (sx, sy - z_offset + self.TILE_H_ISO),
            (sx - self.TILE_W_ISO / 2, sy - z_offset + self.TILE_H_ISO / 2),
            (sx - self.TILE_W_ISO / 2, sy + self.TILE_H_ISO / 2),
            (sx, sy + self.TILE_H_ISO),
        ]
        pygame.draw.polygon(surface, side_color_2, left_points)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False

        self.base_health = self.INITIAL_BASE_HEALTH
        self.gold = self.INITIAL_GOLD

        self.wave = 0
        self.wave_state = "intermission"  # "intermission" or "active"
        self.wave_timer = self.WAVE_INTERMISSION_TIME

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.prev_space_held = False
        self.prev_movement_action = 0
        self.action_cooldown = 0

        self.pending_reward = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        self.pending_reward = -0.01  # Time penalty
        self.steps += 1

        if not self.game_over:
            self._handle_input(action)
            self._update_wave_manager()
            self._update_towers()
            self._update_projectiles()
            self._update_enemies()
            self._update_particles()

        reward = self.pending_reward
        self.score += reward

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated and not self.game_over:
            self.game_over = True
            if self.victory:
                self.pending_reward += 100
                self.score += 100
            else: # Lost due to health
                self.pending_reward -= 100
                self.score -= 100
            reward = self.pending_reward
        
        # The environment is only truncated if it's not already terminated
        truncated = truncated and not terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Cursor movement with cooldown to prevent zipping across the screen
        if self.action_cooldown > 0:
            self.action_cooldown -= 1

        if movement != 0 and self.action_cooldown == 0:
            if movement == 1: self.cursor_pos[1] -= 1  # Up
            elif movement == 2: self.cursor_pos[1] += 1  # Down
            elif movement == 3: self.cursor_pos[0] -= 1  # Left
            elif movement == 4: self.cursor_pos[0] += 1  # Right
            self.action_cooldown = 4  # 4-frame cooldown for movement

        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_SIZE - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_SIZE - 1)

        # Place tower on space press (not hold)
        if space_held and not self.prev_space_held:
            self._place_tower()

        self.prev_space_held = space_held

    def _place_tower(self):
        pos = tuple(self.cursor_pos)
        is_buildable = pos in self.buildable_tiles
        is_occupied = any(t["pos"] == pos for t in self.towers)

        if self.gold >= self.TOWER_COST and is_buildable and not is_occupied:
            self.gold -= self.TOWER_COST
            self.towers.append(
                {
                    "pos": pos,
                    "cooldown": 0,
                }
            )
            # sfx: build_tower.wav

    def _update_wave_manager(self):
        if self.wave_state == "intermission":
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self.wave += 1
                if self.wave > self.MAX_WAVES:
                    self.victory = True
                    return
                self.wave_state = "active"
                self._spawn_wave()
        elif self.wave_state == "active":
            if not self.enemies:
                self.wave_state = "intermission"
                self.wave_timer = self.WAVE_INTERMISSION_TIME
                self.pending_reward += 1.0  # Wave clear bonus

    def _spawn_wave(self):
        # sfx: wave_start.wav
        num_enemies = 3 + self.wave * 2
        health = 10 + self.wave * 5
        speed = 0.03 + self.wave * 0.005

        for i in range(num_enemies):
            start_pos = self._get_path_position(0)
            self.enemies.append(
                {
                    "progress": -i * 0.2,  # Stagger enemies
                    "pos": start_pos,
                    "health": health,
                    "max_health": health,
                    "speed": speed,
                }
            )

    def _get_path_position(self, progress):
        if progress <= 0: return self.path[0]
        if progress >= len(self.path) - 1: return self.path[-1]

        idx = int(progress)
        p1 = self.path[idx]
        p2 = self.path[idx + 1]
        interp = progress - idx

        return (p1[0] + (p2[0] - p1[0]) * interp, p1[1] + (p2[1] - p1[1]) * interp)

    def _update_enemies(self):
        for enemy in self.enemies[:]:
            enemy["progress"] += enemy["speed"]
            enemy["pos"] = self._get_path_position(enemy["progress"])

            if enemy["progress"] >= len(self.path) - 1:
                self.base_health -= self.ENEMY_DAMAGE
                self.base_health = max(0, self.base_health)
                self.enemies.remove(enemy)
                self.pending_reward -= 5.0  # Penalty for leak
                # sfx: base_damage.wav
                # Spawn damage particle at base
                for _ in range(20):
                    self.particles.append(
                        self._create_particle(self.base_pos, self.COLOR_BASE_DMG)
                    )

    def _update_towers(self):
        for tower in self.towers:
            if tower["cooldown"] > 0:
                tower["cooldown"] -= 1
                continue

            target = None
            min_dist = self.TOWER_RANGE ** 2

            # Find closest enemy in range
            for enemy in self.enemies:
                dist_sq = (tower["pos"][0] - enemy["pos"][0]) ** 2 + (
                    tower["pos"][1] - enemy["pos"][1]
                ) ** 2
                if dist_sq < min_dist:
                    min_dist = dist_sq
                    target = enemy

            if target:
                tower["cooldown"] = self.TOWER_COOLDOWN
                self.projectiles.append(
                    {
                        "start_pos": tower["pos"],
                        "end_pos": target["pos"],
                        "target": target,
                        "progress": 0.0,
                    }
                )
                # sfx: tower_shoot.wav

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p["progress"] += 0.1  # Projectile speed

            # Simple linear interpolation for projectile motion
            start_sx, start_sy = self._world_to_screen(p["start_pos"][0], p["start_pos"][1])
            target_sx, target_sy = self._world_to_screen(p["target"]["pos"][0], p["target"]["pos"][1])

            px = start_sx + (target_sx - start_sx) * p["progress"]
            py = start_sy + (target_sy - start_sy) * p["progress"]
            p["screen_pos"] = (px, py)

            if p["progress"] >= 1.0:
                if p["target"] in self.enemies:
                    p["target"]["health"] -= 10  # Projectile damage
                    # sfx: enemy_hit.wav
                    for _ in range(5):
                        self.particles.append(
                            self._create_particle(p["target"]["pos"], self.COLOR_PROJECTILE)
                        )

                    if p["target"]["health"] <= 0:
                        # sfx: enemy_destroy.wav
                        for _ in range(30):
                            self.particles.append(
                                self._create_particle(p["target"]["pos"], self.COLOR_ENEMY)
                            )
                        self.enemies.remove(p["target"])
                        self.pending_reward += 0.1  # Kill reward
                        self.gold += 5  # Gold for kill
                self.projectiles.remove(p)

    def _create_particle(self, grid_pos, color):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 4)
        return {
            "screen_pos": self._world_to_screen(grid_pos[0], grid_pos[1]),
            "vel": (math.cos(angle) * speed, math.sin(angle) * speed),
            "lifespan": self.np_random.integers(15, 30),
            "color": color,
            "size": self.np_random.uniform(2, 5),
        }

    def _update_particles(self):
        for particle in self.particles[:]:
            px, py = particle["screen_pos"]
            vx, vy = particle["vel"]
            particle["screen_pos"] = (px + vx, py + vy)
            particle["lifespan"] -= 1
            particle["size"] *= 0.95
            if particle["lifespan"] <= 0 or particle["size"] < 0.5:
                self.particles.remove(particle)

    def _check_termination(self):
        return self.base_health <= 0 or self.victory

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.gold,
            "wave": self.wave,
            "base_health": self.base_health,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render path and buildable grid
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color = self.COLOR_PATH if (r, c) in self.path else self.COLOR_GRID
                self._draw_iso_poly(self.screen, color, (r, c))

        # Render base
        self._draw_iso_cube(self.screen, self.COLOR_BASE, self.base_pos, height=1.5)

        # Render towers
        for tower in self.towers:
            self._draw_iso_cube(self.screen, self.COLOR_TOWER, tower["pos"], height=1.2)

        # Render enemies
        for enemy in sorted(self.enemies, key=lambda e: e["pos"][0] + e["pos"][1]):
            sx, sy = self._world_to_screen(enemy["pos"][0], enemy["pos"][1])
            pygame.draw.circle(self.screen, self.COLOR_ENEMY, (sx, sy), 8)
            # Health bar
            health_ratio = max(0, enemy["health"] / enemy["max_health"])
            bar_w = 16
            pygame.draw.rect(self.screen, (80, 0, 0), (sx - bar_w / 2, sy - 15, bar_w, 3))
            pygame.draw.rect(
                self.screen,
                self.COLOR_ENEMY_HEALTH,
                (sx - bar_w / 2, sy - 15, bar_w * health_ratio, 3),
            )

        # Render projectiles
        for p in self.projectiles:
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE, p["screen_pos"], 4)

        # Render particles
        for particle in self.particles:
            pos = (int(particle["screen_pos"][0]), int(particle["screen_pos"][1]))
            size = int(particle["size"])
            if size > 0:
                pygame.draw.circle(self.screen, particle["color"], pos, size)

        # Render cursor
        if not self.game_over:
            cursor_tuple = tuple(self.cursor_pos)
            is_buildable = cursor_tuple in self.buildable_tiles
            is_occupied = any(t["pos"] == cursor_tuple for t in self.towers)
            can_afford = self.gold >= self.TOWER_COST

            if is_buildable and not is_occupied and can_afford:
                color = self.COLOR_CURSOR_VALID
            else:
                color = self.COLOR_CURSOR_INVALID

            cursor_surface = self.screen.convert_alpha()
            cursor_surface.fill((0, 0, 0, 0))
            self._draw_iso_poly(cursor_surface, color, self.cursor_pos)
            self.screen.blit(cursor_surface, (0, 0))

    def _render_ui(self):
        # Gold and Wave display
        gold_text = self.font_small.render(f"GOLD: {self.gold}", True, self.COLOR_GOLD)
        wave_text = self.font_small.render(
            f"WAVE: {self.wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT
        )
        self.screen.blit(gold_text, (10, 10))
        self.screen.blit(wave_text, (10, 30))

        # Base Health Bar
        base_sx, base_sy = self._world_to_screen(self.base_pos[0], self.base_pos[1])
        health_ratio = max(0, self.base_health / self.INITIAL_BASE_HEALTH)
        bar_w, bar_h = 80, 10
        bar_x, bar_y = base_sx - bar_w / 2, base_sy + 40
        pygame.draw.rect(self.screen, (80, 0, 0), (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(
            self.screen, self.COLOR_BASE, (bar_x, bar_y, bar_w * health_ratio, bar_h)
        )
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (bar_x, bar_y, bar_w, bar_h), 1)

        # Intermission Timer
        if (
            self.wave_state == "intermission"
            and self.wave < self.MAX_WAVES
            and not self.game_over
        ):
            time_left = math.ceil(self.wave_timer / 30)
            timer_text = self.font_large.render(
                f"NEXT WAVE IN: {time_left}", True, self.COLOR_TEXT
            )
            text_rect = timer_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(timer_text, text_rect)

        # Game Over / Victory Text
        if self.game_over:
            if self.victory:
                msg = "VICTORY!"
                color = self.COLOR_GOLD
            else:
                msg = "GAME OVER"
                color = self.COLOR_ENEMY

            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == "__main__":
    # The os.environ line at the top of the file forces headless mode.
    # To play with a window, you would need to comment out that line.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # The main game loop will create a second display for human interaction.
    # This is not the most efficient way but works for demonstration.
    pygame.display.set_caption("Tower Defense")
    display_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))

    running = True
    while running:
        # --- Human Controls ---
        movement = 0  # no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Pygame Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        # The numpy array needs to be transposed back for pygame's `surfarray.make_surface`
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            pygame.time.wait(3000)  # Pause for 3 seconds before reset
            obs, info = env.reset()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    env.close()