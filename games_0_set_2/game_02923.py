
# Generated: 2025-08-28T06:24:28.277732
# Source Brief: brief_02923.md
# Brief Index: 2923

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to place a tower. Shift to cycle tower type."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of enemies by placing towers on the grid."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
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
        self.screen_width, self.screen_height = 640, 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # Game constants
        self.max_steps = 3000 # Increased for longer gameplay
        self.max_waves = 20
        self.grid_size = (16, 10)
        self.tile_width, self.tile_height = 40, 20
        self.grid_origin = (self.screen_width // 2, 60)
        
        # Define enemy path in grid coordinates
        self.path = [
            (-1, 4), (0, 4), (1, 4), (2, 4), (2, 3), (2, 2), (3, 2), (4, 2), 
            (5, 2), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (7, 6), (8, 6), 
            (9, 6), (10, 6), (11, 6), (11, 5), (11, 4), (12, 4), (13, 4), (14, 4), (15, 4), (16, 4)
        ]
        self.path_pixels = [self._iso_to_screen(p[0], p[1]) for p in self.path]

        # Colors
        self.color_bg = (25, 28, 36)
        self.color_grid = (45, 50, 62)
        self.color_path = (80, 75, 70)
        self.color_base = (0, 200, 255)
        self.color_cursor = (255, 255, 0)
        self.color_cursor_invalid = (255, 0, 0)
        self.color_text = (220, 220, 220)
        self.color_enemy = (255, 50, 50)

        # Fonts
        self.font_main = pygame.font.Font(None, 24)
        self.font_title = pygame.font.Font(None, 48)

        # Tower definitions
        self.tower_types = [
            {"name": "Cannon", "cost": 100, "range": 80, "damage": 25, "fire_rate": 60, "color": (100, 150, 255)},
            {"name": "Rapid", "cost": 150, "range": 60, "damage": 10, "fire_rate": 20, "color": (255, 200, 80)},
            {"name": "Sniper", "cost": 250, "range": 150, "damage": 100, "fire_rate": 120, "color": (100, 255, 100)},
        ]
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.base_health = 0
        self.resources = 0
        self.current_wave = 0
        self.wave_timer = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [0,0]
        self.selected_tower_type = 0
        self.last_shift_press = False
        self.last_space_press = False
        self.wave_spawning_info = {}
        self.total_reward = 0
        
        self.reset()
        self.validate_implementation()

    def _iso_to_screen(self, x, y):
        screen_x = self.grid_origin[0] + (x - y) * self.tile_width / 2
        screen_y = self.grid_origin[1] + (x + y) * self.tile_height / 2
        return int(screen_x), int(screen_y)

    def _is_valid_placement(self, grid_pos):
        x, y = grid_pos
        if not (0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]):
            return False
        if tuple(grid_pos) in self.path:
            return False
        for tower in self.towers:
            if tower['grid_pos'] == grid_pos:
                return False
        return True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.total_reward = 0.0
        self.game_over = False
        self.win = False
        
        self.base_health = 100
        self.resources = 200
        self.current_wave = 0
        self.wave_timer = 150 # Time before first wave
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = [self.grid_size[0] // 2, self.grid_size[1] // 2]
        self.selected_tower_type = 0
        
        self.last_shift_press = True # Prevent action on first frame
        self.last_space_press = True
        
        self.wave_spawning_info = {}
        
        return self._get_observation(), self._get_info()

    def _start_next_wave(self):
        self.current_wave += 1
        self.wave_timer = 0
        if self.current_wave > self.max_waves:
            return

        num_enemies = 2 + self.current_wave * 2
        enemy_health = 50 * (1.05 ** (self.current_wave - 1))
        enemy_speed = 0.75 * (1.05 ** (self.current_wave - 1))
        
        self.wave_spawning_info = {
            "count": num_enemies,
            "health": enemy_health,
            "speed": enemy_speed,
            "spawn_delay": 30, # frames between spawns
            "timer": 0
        }

    def step(self, action):
        reward = -0.01 # Small penalty for surviving a step
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Handle Input ---
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.grid_size[0] - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.grid_size[1] - 1)

        if shift_held and not self.last_shift_press:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.tower_types)
        self.last_shift_press = shift_held

        if space_held and not self.last_space_press:
            tower_def = self.tower_types[self.selected_tower_type]
            if self.resources >= tower_def["cost"] and self._is_valid_placement(self.cursor_pos):
                self.resources -= tower_def["cost"]
                self.towers.append({
                    "type_idx": self.selected_tower_type,
                    "grid_pos": list(self.cursor_pos),
                    "pos": self._iso_to_screen(self.cursor_pos[0], self.cursor_pos[1]),
                    "cooldown": 0,
                    **tower_def
                })
                # sfx: place_tower.wav
        self.last_space_press = space_held

        # --- Game Logic Update ---
        # Wave management
        if not self.enemies and not self.wave_spawning_info.get("count", 0):
            if self.current_wave == 0:
                self.wave_timer -= 1
                if self.wave_timer <= 0: self._start_next_wave()
            elif self.current_wave < self.max_waves:
                if self.wave_timer == 0: reward += 1.0 # Wave clear bonus
                self.wave_timer += 1
                if self.wave_timer > 150: self._start_next_wave()
            elif self.current_wave >= self.max_waves:
                self.win = True
                self.game_over = True
        
        # Enemy spawning
        if self.wave_spawning_info.get("count", 0) > 0:
            self.wave_spawning_info["timer"] -= 1
            if self.wave_spawning_info["timer"] <= 0:
                self.wave_spawning_info["timer"] = self.wave_spawning_info["spawn_delay"]
                self.wave_spawning_info["count"] -= 1
                self.enemies.append({
                    "path_idx": 0,
                    "pos": list(self.path_pixels[0]),
                    "health": self.wave_spawning_info["health"],
                    "max_health": self.wave_spawning_info["health"],
                    "speed": self.wave_spawning_info["speed"],
                    "progress": 0.0
                })
                # sfx: enemy_spawn.wav

        # Update enemies
        for enemy in self.enemies[:]:
            path_idx = enemy["path_idx"]
            if path_idx >= len(self.path_pixels) - 1:
                self.enemies.remove(enemy)
                self.base_health -= 10
                self._create_particles(enemy["pos"], (255, 0, 0), 20)
                # sfx: base_damage.wav
                continue
            
            start_pos = self.path_pixels[path_idx]
            end_pos = self.path_pixels[path_idx + 1]
            dist = math.hypot(end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
            if dist == 0: dist = 1
            
            enemy["progress"] += enemy["speed"]
            if enemy["progress"] >= dist:
                enemy["path_idx"] += 1
                enemy["progress"] = 0
            
            t = enemy["progress"] / dist
            enemy["pos"][0] = start_pos[0] + (end_pos[0] - start_pos[0]) * t
            enemy["pos"][1] = start_pos[1] + (end_pos[1] - start_pos[1]) * t

        # Update towers
        for tower in self.towers:
            tower["cooldown"] = max(0, tower["cooldown"] - 1)
            if tower["cooldown"] > 0: continue

            target = None
            for enemy in self.enemies:
                dist = math.hypot(enemy["pos"][0] - tower["pos"][0], enemy["pos"][1] - tower["pos"][1])
                if dist <= tower["range"]:
                    target = enemy
                    break
            
            if target:
                tower["cooldown"] = tower["fire_rate"]
                self.projectiles.append({
                    "start_pos": list(tower["pos"]),
                    "pos": list(tower["pos"]),
                    "target": target,
                    "damage": tower["damage"],
                    "color": tower["color"],
                    "speed": 8
                })
                # sfx: tower_fire.wav

        # Update projectiles
        for proj in self.projectiles[:]:
            target_pos = proj["target"]["pos"]
            dx, dy = target_pos[0] - proj["pos"][0], target_pos[1] - proj["pos"][1]
            dist = math.hypot(dx, dy)
            if dist < proj["speed"]:
                proj["target"]["health"] -= proj["damage"]
                self._create_particles(proj["pos"], proj["color"], 5)
                self.projectiles.remove(proj)
                # sfx: projectile_hit.wav
                if proj["target"]["health"] <= 0:
                    if proj["target"] in self.enemies:
                        self.enemies.remove(proj["target"])
                        self.resources += 15
                        reward += 0.1
                        self._create_particles(proj["target"]["pos"], self.color_enemy, 15)
                        # sfx: enemy_death.wav
            else:
                proj["pos"][0] += (dx / dist) * proj["speed"]
                proj["pos"][1] += (dy / dist) * proj["speed"]

        # Update particles
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # Gravity
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

        # --- Termination Check ---
        self.steps += 1
        terminated = False
        if self.base_health <= 0:
            self.game_over = True
            reward = -100.0
        if self.win:
            reward = 100.0
        if self.steps >= self.max_steps and not self.win:
            self.game_over = True
        
        if self.game_over:
            terminated = True
        
        self.total_reward += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed - 2],
                "life": random.randint(15, 30),
                "color": color
            })

    def _get_observation(self):
        self.screen.fill(self.color_bg)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid and path
        for y in range(self.grid_size[1]):
            for x in range(self.grid_size[0]):
                p1 = self._iso_to_screen(x, y)
                p2 = self._iso_to_screen(x + 1, y)
                p3 = self._iso_to_screen(x + 1, y + 1)
                p4 = self._iso_to_screen(x, y + 1)
                color = self.color_path if (x,y) in self.path else self.color_grid
                pygame.draw.polygon(self.screen, color, [p1, p2, p3, p4])
                pygame.draw.aalines(self.screen, self.color_bg, True, [p1, p2, p3, p4])

        # Draw base
        base_pos = self._iso_to_screen(self.path[-1][0], self.path[-1][1])
        pygame.draw.circle(self.screen, self.color_base, base_pos, 15)
        pygame.draw.circle(self.screen, (255,255,255), base_pos, 15, 2)

        # Draw cursor
        cursor_screen_pos = self._iso_to_screen(self.cursor_pos[0], self.cursor_pos[1])
        is_valid = self._is_valid_placement(self.cursor_pos)
        cursor_color = self.color_cursor if is_valid else self.color_cursor_invalid
        p1 = self._iso_to_screen(self.cursor_pos[0], self.cursor_pos[1])
        p2 = self._iso_to_screen(self.cursor_pos[0] + 1, self.cursor_pos[1])
        p3 = self._iso_to_screen(self.cursor_pos[0] + 1, self.cursor_pos[1] + 1)
        p4 = self._iso_to_screen(self.cursor_pos[0], self.cursor_pos[1] + 1)
        pygame.draw.polygon(self.screen, (cursor_color[0], cursor_color[1], cursor_color[2], 50), [p1, p2, p3, p4])
        pygame.draw.aalines(self.screen, cursor_color, True, [p1, p2, p3, p4], 2)

        # Draw shadows (before units)
        for tower in self.towers:
            pygame.gfxdraw.filled_ellipse(self.screen, tower['pos'][0], tower['pos'][1] + 8, 12, 6, (0,0,0,100))
        for enemy in self.enemies:
            pygame.gfxdraw.filled_ellipse(self.screen, int(enemy['pos'][0]), int(enemy['pos'][1]) + 4, 8, 4, (0,0,0,100))

        # Draw towers
        for tower in self.towers:
            pos = tower['pos']
            pygame.draw.rect(self.screen, tower['color'], (pos[0]-6, pos[1]-12, 12, 14))
            pygame.draw.rect(self.screen, (50,50,50), (pos[0]-2, pos[1]-18, 4, 6))

        # Draw enemies
        for enemy in self.enemies:
            pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            pygame.draw.circle(self.screen, self.color_enemy, pos, 6)
            # Health bar
            health_pct = enemy['health'] / enemy['max_health']
            pygame.draw.rect(self.screen, (50,50,50), (pos[0]-8, pos[1]-12, 16, 3))
            pygame.draw.rect(self.screen, (0,255,0), (pos[0]-8, pos[1]-12, 16 * health_pct, 3))

        # Draw projectiles
        for proj in self.projectiles:
            pos = (int(proj['pos'][0]), int(proj['pos'][1]))
            pygame.draw.circle(self.screen, (255,255,255), pos, 4)
            pygame.draw.circle(self.screen, proj['color'], pos, 2)

        # Draw particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            size = max(1, int(p['life'] / 6))
            pygame.draw.circle(self.screen, p['color'], pos, size)

    def _render_ui(self):
        # Top bar
        ui_bar = pygame.Surface((self.screen_width, 30), pygame.SRCALPHA)
        ui_bar.fill((20, 20, 25, 180))
        self.screen.blit(ui_bar, (0, 0))

        # Base Health
        health_text = self.font_main.render(f"Base Health: {max(0, self.base_health)}/100", True, self.color_text)
        self.screen.blit(health_text, (10, 7))

        # Resources
        resource_text = self.font_main.render(f"Resources: {self.resources}", True, self.color_text)
        self.screen.blit(resource_text, (200, 7))

        # Wave Info
        wave_str = f"Wave: {self.current_wave}/{self.max_waves}"
        if self.wave_timer > 0 and self.current_wave < self.max_waves and not self.enemies:
            wave_str += f" (Next in {self.wave_timer // 30 + 1}s)"
        wave_text = self.font_main.render(wave_str, True, self.color_text)
        self.screen.blit(wave_text, (370, 7))
        
        # Reward
        reward_text = self.font_main.render(f"Score: {self.total_reward:.2f}", True, self.color_text)
        self.screen.blit(reward_text, (530, 7))

        # Bottom tower selection bar
        bar_height = 60
        bottom_bar = pygame.Surface((self.screen_width, bar_height), pygame.SRCALPHA)
        bottom_bar.fill((20, 20, 25, 180))
        self.screen.blit(bottom_bar, (0, self.screen_height - bar_height))

        for i, t_def in enumerate(self.tower_types):
            x_offset = 150 + i * 120
            box_rect = pygame.Rect(x_offset, self.screen_height - bar_height + 5, 110, 50)
            
            if i == self.selected_tower_type:
                pygame.draw.rect(self.screen, self.color_cursor, box_rect, 2, 3)

            name_text = self.font_main.render(t_def['name'], True, self.color_text)
            cost_text = self.font_main.render(f"Cost: {t_def['cost']}", True, self.color_text)
            self.screen.blit(name_text, (x_offset + 5, self.screen_height - bar_height + 10))
            self.screen.blit(cost_text, (x_offset + 5, self.screen_height - bar_height + 30))
        
        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            text = self.font_title.render(message, True, color)
            text_rect = text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.total_reward,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": self.current_wave,
            "enemies_left": len(self.enemies) + self.wave_spawning_info.get("count", 0)
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # To display the game, you'd need a different render_mode and setup
    # This example just runs steps and prints info
    
    # For interactive play:
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    
    terminated = False
    
    # Map pygame keys to actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    while not terminated:
        movement = 0
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        for key, move_action in key_to_action.items():
            if keys[key]:
                movement = move_action
                break # only one movement at a time
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        if keys[pygame.K_SHIFT]:
            shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Blit the observation onto the display screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS

    env.close()