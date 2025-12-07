
# Generated: 2025-08-28T00:44:31.314758
# Source Brief: brief_03886.md
# Brief Index: 3886

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. "
        "Press 'Shift' to cycle through tower types. "
        "Press 'Space' to build the selected tower at the cursor's location."
    )

    game_description = (
        "A classic tower defense game. Strategically place towers on the grid to defend your base "
        "from waves of incoming enemies. Survive all 10 waves to win."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 30 * 120 # 2 minutes max

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 70)
    COLOR_PATH = (60, 75, 100)
    COLOR_BASE = (60, 180, 75)
    COLOR_BASE_DMG = (255, 100, 100)
    COLOR_ENEMY = (230, 25, 75)
    COLOR_TEXT = (230, 230, 230)
    COLOR_UI_ACCENT = (70, 150, 255)
    
    TOWER_COLORS = [(0, 130, 200), (245, 130, 48)] # Blue, Orange
    PROJECTILE_COLORS = [(0, 190, 255), (255, 190, 110)]

    # Grid
    GRID_COLS, GRID_ROWS = 16, 10
    GRID_CELL_SIZE = 40
    GRID_OFFSET_X = 0
    GRID_OFFSET_Y = 0

    # Base
    BASE_MAX_HEALTH = 20

    # Towers
    TOWER_TYPES = [
        {
            "name": "Cannon", "cost": 50, "range": 80, "damage": 1, "fire_rate": 45, # 1.5s
            "shape": "circle", "color": TOWER_COLORS[0], "proj_color": PROJECTILE_COLORS[0]
        },
        {
            "name": "Missile", "cost": 75, "range": 120, "damage": 3, "fire_rate": 90, # 3.0s
            "shape": "square", "color": TOWER_COLORS[1], "proj_color": PROJECTILE_COLORS[1]
        },
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("monospace", 14)
        self.font_m = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_l = pygame.font.SysFont("monospace", 48, bold=True)

        self._define_enemy_path()
        
        self.reset()

        self.validate_implementation()

    def _define_enemy_path(self):
        self.path = []
        path_points = [
            (-1, 2), (2, 2), (2, 7), (13, 7), (13, 4), (16, 4)
        ]
        for i in range(len(path_points) - 1):
            x1, y1 = path_points[i]
            x2, y2 = path_points[i+1]
            self.path.append((x1 * self.GRID_CELL_SIZE, y1 * self.GRID_CELL_SIZE))
        self.path_pixels = []
        for i in range(len(path_points) - 1):
            p1 = (path_points[i][0] * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE//2, path_points[i][1] * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE//2)
            p2 = (path_points[i+1][0] * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE//2, path_points[i+1][1] * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE//2)
            
            length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            if length == 0: continue
            
            num_steps = int(length)
            for j in range(num_steps):
                t = j / num_steps
                x = p1[0] * (1 - t) + p2[0] * t
                y = p1[1] * (1 - t) + p2[1] * t
                self.path_pixels.append((x, y))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.terminal_reward_given = False

        self.base_health = self.BASE_MAX_HEALTH
        self.resources = 125
        
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_tower_index = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = deque()

        self.wave_num = 0
        self.wave_cooldown = self.FPS * 5 # 5 second countdown to first wave
        self.is_wave_active = False

        self.placement_feedback = 0 # Countdown for visual feedback
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self._handle_actions(action)
        
        reward += self._update_game_logic()
        
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated and not self.terminal_reward_given:
            if self.game_won:
                reward += 100
            else: # Lost by health or time
                reward -= 100 if self.base_health <= 0 else 0
            self.terminal_reward_given = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

        # --- Tower Selection ---
        shift_pressed = shift_held and not self.prev_shift_held
        if shift_pressed:
            self.selected_tower_index = (self.selected_tower_index + 1) % len(self.TOWER_TYPES)
        
        # --- Tower Placement ---
        space_pressed = space_held and not self.prev_space_held
        if space_pressed:
            self._place_tower()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
    def _place_tower(self):
        tower_type = self.TOWER_TYPES[self.selected_tower_index]
        cost = tower_type["cost"]
        
        is_on_path = False
        cursor_center_x = (self.cursor_pos[0] + 0.5) * self.GRID_CELL_SIZE
        cursor_center_y = (self.cursor_pos[1] + 0.5) * self.GRID_CELL_SIZE
        for px, py in self.path_pixels:
            if math.hypot(px - cursor_center_x, py - cursor_center_y) < self.GRID_CELL_SIZE * 0.7:
                is_on_path = True
                break

        is_occupied = any(t['grid_pos'] == self.cursor_pos for t in self.towers)

        if self.resources >= cost and not is_on_path and not is_occupied:
            # sfx: place_tower_success
            self.resources -= cost
            new_tower = {
                "type": tower_type,
                "pos": (cursor_center_x, cursor_center_y),
                "grid_pos": list(self.cursor_pos),
                "cooldown": 0,
            }
            self.towers.append(new_tower)
            self.placement_feedback = self.FPS // 3 # green flash
        else:
            # sfx: place_tower_fail
            self.placement_feedback = -self.FPS // 3 # red flash

    def _update_game_logic(self):
        reward = 0
        self._update_wave_manager()
        
        new_projectiles, hit_reward = self._update_towers()
        self.projectiles.extend(new_projectiles)
        reward += hit_reward
        
        kill_reward, leak_penalty = self._update_enemies()
        reward += kill_reward + leak_penalty
        
        self._update_projectiles()
        self._update_particles()

        if self.base_health < self.BASE_MAX_HEALTH:
            reward -= 0.01

        return reward

    def _update_wave_manager(self):
        if not self.is_wave_active:
            self.wave_cooldown -= 1
            if self.wave_cooldown <= 0 and self.wave_num < 10:
                self.wave_num += 1
                self._spawn_wave()
                self.is_wave_active = True
        elif not self.enemies: # Wave is active and no enemies left
            self.is_wave_active = False
            self.wave_cooldown = self.FPS * 7 # 7 seconds between waves

    def _spawn_wave(self):
        # sfx: wave_start
        num_enemies = 5 + self.wave_num * 2
        speed = 0.7 + self.wave_num * 0.05
        health = 1 + self.wave_num // 2
        for i in range(num_enemies):
            self.enemies.append({
                "path_index": -i * 15, # Staggered start
                "health": health,
                "max_health": health,
                "speed": speed,
                "pos": self.path_pixels[0],
            })

    def _update_towers(self):
        new_projectiles = []
        hit_reward = 0
        for tower in self.towers:
            if tower["cooldown"] > 0:
                tower["cooldown"] -= 1
                continue
            
            target = None
            farthest_path_index = -1
            for enemy in self.enemies:
                dist = math.hypot(enemy["pos"][0] - tower["pos"][0], enemy["pos"][1] - tower["pos"][1])
                if dist <= tower["type"]["range"] and enemy["path_index"] > farthest_path_index:
                    farthest_path_index = enemy["path_index"]
                    target = enemy
            
            if target:
                # sfx: tower_fire
                tower["cooldown"] = tower["type"]["fire_rate"]
                new_projectiles.append({
                    "pos": list(tower["pos"]),
                    "type": tower["type"],
                    "target": target,
                    "speed": 8,
                })
        return new_projectiles, hit_reward

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            if proj["target"] not in self.enemies:
                self.projectiles.remove(proj)
                continue
            
            target_pos = proj["target"]["pos"]
            dx = target_pos[0] - proj["pos"][0]
            dy = target_pos[1] - proj["pos"][1]
            dist = math.hypot(dx, dy)
            
            if dist < proj["speed"]:
                # sfx: projectile_hit
                proj["target"]["health"] -= proj["type"]["damage"]
                self.projectiles.remove(proj)
                self._create_particles(target_pos, proj["type"]["proj_color"], 5, 2)
            else:
                proj["pos"][0] += (dx / dist) * proj["speed"]
                proj["pos"][1] += (dy / dist) * proj["speed"]

    def _update_enemies(self):
        kill_reward = 0
        leak_penalty = 0
        for enemy in self.enemies[:]:
            if enemy["health"] <= 0:
                # sfx: enemy_die
                kill_reward += 1
                self.resources += 5 + self.wave_num // 2
                self._create_particles(enemy["pos"], self.COLOR_ENEMY, 15, 4)
                self.enemies.remove(enemy)
                continue

            enemy["path_index"] += enemy["speed"]
            current_pixel_index = int(enemy["path_index"])

            if current_pixel_index >= len(self.path_pixels):
                # sfx: base_damage
                self.base_health -= 1
                leak_penalty -= 10
                self.enemies.remove(enemy)
                self._create_particles((self.WIDTH - 15, self.HEIGHT/2 + 30), self.COLOR_BASE_DMG, 30, 8)
            elif current_pixel_index >= 0:
                enemy["pos"] = self.path_pixels[current_pixel_index]
        return kill_reward, leak_penalty

    def _create_particles(self, pos, color, count, max_life):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(max_life//2, max_life) * (self.FPS // 15),
                "color": color
            })

    def _update_particles(self):
        for p in list(self.particles):
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.95
            p["vel"][1] *= 0.95
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.popleft()

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
            return True
        if self.wave_num >= 10 and not self.enemies and not self.is_wave_active:
            self.game_over = True
            self.game_won = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_grid_and_path()
        self._render_base()
        self._render_towers()
        self._render_enemies()
        self._render_projectiles()
        self._render_particles()
        if not self.game_over:
            self._render_cursor()

    def _render_grid_and_path(self):
        for x in range(0, self.WIDTH, self.GRID_CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
        
        if len(self.path_pixels) > 1:
            pygame.draw.aalines(self.screen, self.COLOR_PATH, False, self.path_pixels)

    def _render_base(self):
        base_rect = pygame.Rect(self.WIDTH - self.GRID_CELL_SIZE, 4 * self.GRID_CELL_SIZE, self.GRID_CELL_SIZE, self.GRID_CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)
        
        health_ratio = self.base_health / self.BASE_MAX_HEALTH
        health_bar_width = self.GRID_CELL_SIZE * 2
        health_bar_rect = pygame.Rect(base_rect.centerx - health_bar_width/2, base_rect.top - 15, health_bar_width, 8)
        
        pygame.draw.rect(self.screen, self.COLOR_GRID, health_bar_rect, border_radius=2)
        if health_ratio > 0:
            fill_rect = health_bar_rect.copy()
            fill_rect.width = int(health_bar_width * health_ratio)
            pygame.draw.rect(self.screen, self.COLOR_BASE, fill_rect, border_radius=2)

    def _render_towers(self):
        for tower in self.towers:
            ttype = tower["type"]
            pos = (int(tower["pos"][0]), int(tower["pos"][1]))
            if ttype["shape"] == "circle":
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.GRID_CELL_SIZE // 3, ttype["color"])
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.GRID_CELL_SIZE // 3, self.COLOR_BG)
            elif ttype["shape"] == "square":
                size = self.GRID_CELL_SIZE // 2
                rect = pygame.Rect(pos[0] - size//2, pos[1] - size//2, size, size)
                pygame.draw.rect(self.screen, ttype["color"], rect, border_radius=3)
                pygame.draw.rect(self.screen, self.COLOR_BG, rect, width=2, border_radius=3)

    def _render_enemies(self):
        for enemy in self.enemies:
            pos = (int(enemy["pos"][0]), int(enemy["pos"][1]))
            radius = 8
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_BG)
            
            # Health bar
            if enemy["health"] < enemy["max_health"]:
                health_ratio = enemy["health"] / enemy["max_health"]
                bar_w = 16
                bar_h = 3
                bar_x = pos[0] - bar_w / 2
                bar_y = pos[1] - radius - bar_h - 2
                pygame.draw.rect(self.screen, self.COLOR_BG, (bar_x, bar_y, bar_w, bar_h))
                pygame.draw.rect(self.screen, self.COLOR_BASE, (bar_x, bar_y, bar_w * health_ratio, bar_h))

    def _render_projectiles(self):
        for proj in self.projectiles:
            pos = (int(proj["pos"][0]), int(proj["pos"][1]))
            pygame.draw.rect(self.screen, proj["type"]["proj_color"], (pos[0]-2, pos[1]-2, 4, 4))
    
    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["life"])) if p["life"] > 0 else 0
            size = int(p["life"] * 0.2 + 1)
            if size > 0:
                pygame.draw.rect(self.screen, p["color"], (int(p["pos"][0]-size/2), int(p["pos"][1]-size/2), size, size))

    def _render_cursor(self):
        x, y = self.cursor_pos
        rect = pygame.Rect(x * self.GRID_CELL_SIZE, y * self.GRID_CELL_SIZE, self.GRID_CELL_SIZE, self.GRID_CELL_SIZE)
        
        selected_tower = self.TOWER_TYPES[self.selected_tower_index]
        
        # Determine feedback color
        feedback_color = None
        if self.placement_feedback > 0:
            feedback_color = (*self.COLOR_BASE, 100)
            self.placement_feedback -= 1
        elif self.placement_feedback < 0:
            feedback_color = (*self.COLOR_ENEMY, 100)
            self.placement_feedback += 1
        
        if feedback_color:
            s = pygame.Surface((self.GRID_CELL_SIZE, self.GRID_CELL_SIZE), pygame.SRCALPHA)
            s.fill(feedback_color)
            self.screen.blit(s, rect.topleft)

        # Draw range indicator
        range_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(range_surface, rect.centerx, rect.centery, selected_tower["range"], (*self.COLOR_UI_ACCENT, 30))
        pygame.gfxdraw.aacircle(range_surface, rect.centerx, rect.centery, selected_tower["range"], (*self.COLOR_UI_ACCENT, 100))
        self.screen.blit(range_surface, (0, 0))
        
        # Draw cursor box
        pygame.draw.rect(self.screen, self.COLOR_UI_ACCENT, rect, 2)
        
    def _render_ui(self):
        # Top Bar
        bar_rect = pygame.Rect(0, 0, self.WIDTH, 30)
        pygame.draw.rect(self.screen, (0,0,0,150), bar_rect)
        
        # Wave Info
        if self.wave_num == 0:
            wave_text = f"Wave starts in: {self.wave_cooldown / self.FPS:.1f}s"
        elif self.is_wave_active:
            wave_text = f"Wave: {self.wave_num} / 10"
        elif self.wave_num < 10:
            wave_text = f"Next wave in: {self.wave_cooldown / self.FPS:.1f}s"
        else:
            wave_text = "All waves cleared!"
        self._render_text(wave_text, (10, 7), self.font_m, self.COLOR_TEXT)

        # Resources
        res_text = f"$: {self.resources}"
        self._render_text(res_text, (self.WIDTH - 240, 7), self.font_m, (255, 215, 0))

        # Selected Tower
        tower = self.TOWER_TYPES[self.selected_tower_index]
        tower_text = f"[{tower['name']}] Cost: {tower['cost']}"
        self._render_text(tower_text, (self.WIDTH - 150, 7), self.font_m, self.COLOR_TEXT)

        # Game Over Screen
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            
            end_text = "VICTORY" if self.game_won else "GAME OVER"
            color = self.COLOR_BASE if self.game_won else self.COLOR_ENEMY
            self._render_text(end_text, (self.WIDTH/2, self.HEIGHT/2 - 30), self.font_l, color, align="center")
            
            score_text = f"Final Score: {self.score:.0f}"
            self._render_text(score_text, (self.WIDTH/2, self.HEIGHT/2 + 20), self.font_m, self.COLOR_TEXT, align="center")

    def _render_text(self, text, pos, font, color, align="left"):
        surface = font.render(text, True, color)
        rect = surface.get_rect()
        if align == "center":
            rect.center = pos
        elif align == "right":
            rect.topright = pos
        else:
            rect.topleft = pos
        self.screen.blit(surface, rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_num,
            "base_health": self.base_health,
            "resources": self.resources,
            "enemies_left": len(self.enemies)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It's a demonstration of the environment's functionality
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Action mapping from keyboard to MultiDiscrete action space ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Wave: {info['wave']}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)
        
    env.close()