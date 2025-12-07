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
        "Controls: Arrow keys to move cursor. Space to place tower or start wave. Shift to cycle tower type."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from zombie waves by placing towers. Survive 10 waves to win."
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
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Game Constants
        self.GRID_SIZE = 40
        self.GRID_W = self.WIDTH // self.GRID_SIZE
        self.GRID_H = self.HEIGHT // self.GRID_SIZE
        self.MAX_STEPS = 12000 # Approx 6.5 mins at 30fps, enough for 10 waves
        self.WAVES_TO_WIN = 10

        # Colors
        self.COLOR_BG = (25, 20, 35)
        self.COLOR_GRID = (40, 35, 55)
        self.COLOR_PATH = (50, 45, 65)
        self.COLOR_BASE = (60, 180, 75)
        self.COLOR_BASE_DMG = (255, 100, 100)
        self.COLOR_ZOMBIE = (230, 25, 75)
        self.COLOR_TEXT = (245, 245, 245)
        self.COLOR_CURSOR = (255, 255, 25, 150)
        
        self.TOWER_SPECS = {
            0: {"name": "Gatling", "cost": 100, "range": 80, "cooldown": 5, "damage": 5, "color": (0, 130, 200)},
            1: {"name": "Cannon", "cost": 250, "range": 120, "cooldown": 30, "damage": 35, "color": (245, 130, 48)},
        }

        # Fonts
        try:
            self.font_s = pygame.font.SysFont("Consolas", 16)
            self.font_m = pygame.font.SysFont("Consolas", 24)
            self.font_l = pygame.font.SysFont("Consolas", 48)
        except pygame.error:
            self.font_s = pygame.font.Font(None, 20)
            self.font_m = pygame.font.Font(None, 30)
            self.font_l = pygame.font.Font(None, 54)

        # Initialize state variables
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        
        self.game_phase = "PLACEMENT" # "PLACEMENT" or "WAVE_ACTIVE"
        self.base_max_health = 100
        self.base_health = self.base_max_health
        self.money = 300 # Starting money
        self.wave_number = 0
        
        self.grid = np.zeros((self.GRID_W, self.GRID_H), dtype=int)
        self._define_path_and_base()
        
        self.towers = []
        self.zombies = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.selected_tower_type = 0
        
        self.last_space_held = False
        self.last_shift_held = False

        self.zombies_to_spawn = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        self.steps += 1

        reward += self._handle_input(action)

        if self.game_phase == "WAVE_ACTIVE":
            self._spawn_zombies_step()
            self._update_towers()
            reward += self._update_zombies()
            reward += self._update_projectiles()

            # Check for wave completion
            if not self.zombies and not self.zombies_to_spawn:
                self.game_phase = "PLACEMENT"
                reward += 1.0 # Wave complete reward
                self.money += 150 + self.wave_number * 10
                # Sound: wave_complete.wav
                if self.wave_number >= self.WAVES_TO_WIN:
                    self.game_over = True
                    self.victory = True
                    reward += 50.0 # Victory reward

        self._update_particles()
        
        if self.base_health <= 0:
            self.game_over = True
            self.victory = False
            reward -= 50.0 # Defeat penalty

        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        self.last_space_held, self.last_shift_held = space_held, shift_held

        if self.game_phase == "PLACEMENT":
            # Cursor movement
            if movement == 1: self.cursor_pos[1] -= 1
            elif movement == 2: self.cursor_pos[1] += 1
            elif movement == 3: self.cursor_pos[0] -= 1
            elif movement == 4: self.cursor_pos[0] += 1
            self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_W - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_H - 1)

            # Cycle tower type
            if shift_pressed:
                self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
                # Sound: ui_cycle.wav

            # Place tower or start wave
            if space_pressed:
                cx, cy = self.cursor_pos
                # Check if cursor is on "Start Wave" button
                if self.start_wave_button_rect.collidepoint(cx * self.GRID_SIZE + self.GRID_SIZE//2, cy * self.GRID_SIZE + self.GRID_SIZE//2):
                    self._start_next_wave()
                    # Sound: wave_start.wav
                else: # Try to place tower
                    self._place_tower(cx, cy)
        return 0.0

    def _define_path_and_base(self):
        self.path_waypoints = [
            (-1, 2), (2, 2), (2, 7), (12, 7), (12, 4), (16, 4)
        ]
        self.path_pixels = [(p[0] * self.GRID_SIZE, p[1] * self.GRID_SIZE) for p in self.path_waypoints]
        
        self.base_grid_pos = (14, 2)
        self.base_rect = pygame.Rect(self.base_grid_pos[0] * self.GRID_SIZE, self.base_grid_pos[1] * self.GRID_SIZE, self.GRID_SIZE * 2, self.GRID_SIZE * 2)

        # Mark path and base areas as unbuildable
        for i in range(len(self.path_waypoints) - 1):
            p1 = self.path_waypoints[i]
            p2 = self.path_waypoints[i+1]
            x1, y1 = p1
            x2, y2 = p2

            if x1 == x2:  # Vertical line
                for y in range(min(y1, y2), max(y1, y2) + 1):
                    if 0 <= x1 < self.GRID_W and 0 <= y < self.GRID_H:
                        self.grid[x1, y] = -1
            elif y1 == y2:  # Horizontal line
                for x in range(min(x1, x2), max(x1, x2) + 1):
                    if 0 <= x < self.GRID_W and 0 <= y1 < self.GRID_H:
                        self.grid[x, y1] = -1
        
        for x in range(self.base_grid_pos[0], self.base_grid_pos[0] + 2):
            for y in range(self.base_grid_pos[1], self.base_grid_pos[1] + 2):
                 if 0 <= x < self.GRID_W and 0 <= y < self.GRID_H: self.grid[x, y] = -1

    def _start_next_wave(self):
        self.wave_number += 1
        self.game_phase = "WAVE_ACTIVE"
        num_zombies = 5 + (self.wave_number - 1) * 2
        base_health = 10 * (1.05 ** (self.wave_number - 1))
        base_speed = 1.0 * (1.02 ** (self.wave_number - 1))

        self.zombies_to_spawn = []
        for i in range(num_zombies):
            zombie = {
                "pos": [self.path_pixels[0][0] - i * 20, self.path_pixels[0][1] + self.GRID_SIZE//2],
                "health": base_health,
                "max_health": base_health,
                "speed": base_speed * random.uniform(0.9, 1.1),
                "path_index": 1,
                "id": self.steps + i
            }
            self.zombies_to_spawn.append(zombie)

    def _spawn_zombies_step(self):
        if self.zombies_to_spawn and self.steps % 15 == 0: # Spawn one zombie every 0.5s
            self.zombies.append(self.zombies_to_spawn.pop(0))

    def _place_tower(self, gx, gy):
        spec = self.TOWER_SPECS[self.selected_tower_type]
        if self.grid[gx, gy] == 0 and self.money >= spec["cost"]:
            self.money -= spec["cost"]
            self.grid[gx, gy] = 1
            self.towers.append({
                "pos": (gx * self.GRID_SIZE + self.GRID_SIZE//2, gy * self.GRID_SIZE + self.GRID_SIZE//2),
                "type": self.selected_tower_type,
                "cooldown": 0,
                "angle": 0,
            })
            # Sound: place_tower.wav
            self._create_particles(gx * self.GRID_SIZE + self.GRID_SIZE//2, gy * self.GRID_SIZE + self.GRID_SIZE//2, spec["color"], 20, 2)

    def _update_towers(self):
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            if tower["cooldown"] > 0:
                tower["cooldown"] -= 1
                continue
            
            target = None
            min_dist = spec["range"] ** 2
            for zombie in self.zombies:
                dist_sq = (tower["pos"][0] - zombie["pos"][0])**2 + (tower["pos"][1] - zombie["pos"][1])**2
                if dist_sq < min_dist:
                    min_dist = dist_sq
                    target = zombie
            
            if target:
                tower["cooldown"] = spec["cooldown"]
                dx = target["pos"][0] - tower["pos"][0]
                dy = target["pos"][1] - tower["pos"][1]
                tower["angle"] = math.atan2(dy, dx)
                
                self.projectiles.append({
                    "pos": list(tower["pos"]),
                    "vel": [math.cos(tower["angle"]) * 8, math.sin(tower["angle"]) * 8],
                    "damage": spec["damage"],
                    "type": tower["type"],
                    "id": self.steps + random.random()
                })
                # Sound: shoot_gatling.wav or shoot_cannon.wav
                self._create_particles(tower["pos"][0], tower["pos"][1], (255,255,255), 3, 1)


    def _update_zombies(self):
        reward = 0.0
        for zombie in reversed(self.zombies):
            if zombie["path_index"] >= len(self.path_pixels):
                self.base_health -= 10
                self._create_particles(self.base_rect.centerx, self.base_rect.centery, self.COLOR_ZOMBIE, 30, 4)
                # Sound: base_damage.wav
                self.zombies.remove(zombie)
                continue

            target_pos = self.path_pixels[zombie["path_index"]]
            dx, dy = target_pos[0] - zombie["pos"][0], target_pos[1] - zombie["pos"][1]
            dist = math.hypot(dx, dy)

            if dist < zombie["speed"]:
                zombie["path_index"] += 1
            else:
                zombie["pos"][0] += (dx / dist) * zombie["speed"]
                zombie["pos"][1] += (dy / dist) * zombie["speed"]
        return reward

    def _update_projectiles(self):
        reward = 0.0
        for p in reversed(self.projectiles):
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]

            if not (0 < p["pos"][0] < self.WIDTH and 0 < p["pos"][1] < self.HEIGHT):
                self.projectiles.remove(p)
                continue

            for z in reversed(self.zombies):
                if math.hypot(p["pos"][0] - z["pos"][0], p["pos"][1] - z["pos"][1]) < 10:
                    z["health"] -= p["damage"]
                    self._create_particles(p["pos"][0], p["pos"][1], (255,255,255), 5, 2)
                    # Sound: hit_zombie.wav
                    if z["health"] <= 0:
                        reward += 0.1
                        self.score += 10
                        self.money += 5
                        self._create_particles(z["pos"][0], z["pos"][1], self.COLOR_ZOMBIE, 25, 3)
                        self.zombies.remove(z)
                        # Sound: kill_zombie.wav
                    if p in self.projectiles:
                        self.projectiles.remove(p)
                    break
        return reward
    
    def _create_particles(self, x, y, color, count, max_life):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                "pos": [x, y],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": random.randint(max_life * 5, max_life * 10),
                "color": color
            })

    def _update_particles(self):
        for p in reversed(self.particles):
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.95
            p["vel"][1] *= 0.95
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid and path
        for i in range(1, self.GRID_W):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i * self.GRID_SIZE, 0), (i * self.GRID_SIZE, self.HEIGHT))
        for i in range(1, self.GRID_H):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i * self.GRID_SIZE), (self.WIDTH, i * self.GRID_SIZE))
        
        for i in range(len(self.path_pixels) - 1):
            p1_pixel = (self.path_pixels[i][0] + self.GRID_SIZE // 2, self.path_pixels[i][1] + self.GRID_SIZE // 2)
            p2_pixel = (self.path_pixels[i+1][0] + self.GRID_SIZE // 2, self.path_pixels[i+1][1] + self.GRID_SIZE // 2)
            pygame.draw.line(self.screen, self.COLOR_PATH, p1_pixel, p2_pixel, self.GRID_SIZE)

        # Draw base
        pygame.draw.rect(self.screen, self.COLOR_BASE, self.base_rect)
        pygame.gfxdraw.rectangle(self.screen, self.base_rect, (255,255,255))
        
        # Draw towers
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            pos = (int(tower["pos"][0]), int(tower["pos"][1]))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 12, spec["color"])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 12, spec["color"])
            # Draw barrel
            end_x = pos[0] + 15 * math.cos(tower["angle"])
            end_y = pos[1] + 15 * math.sin(tower["angle"])
            pygame.draw.line(self.screen, self.COLOR_TEXT, pos, (int(end_x), int(end_y)), 3)

        # Draw projectiles
        for p in self.projectiles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            color = self.TOWER_SPECS[p["type"]]["color"]
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, color)

        # Draw zombies
        for z in self.zombies:
            pos = (int(z["pos"][0]), int(z["pos"][1]))
            size = 8
            rect = pygame.Rect(pos[0] - size, pos[1] - size, size*2, size*2)
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, rect)
            # Health bar
            health_pct = z["health"] / z["max_health"]
            pygame.draw.rect(self.screen, (50,0,0), (rect.left, rect.top - 6, rect.width, 4))
            pygame.draw.rect(self.screen, self.COLOR_BASE, (rect.left, rect.top - 6, rect.width * health_pct, 4))
            
        # Draw particles
        for p in self.particles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            life_pct = max(0, p["life"] / 30.0)
            size = int(2 * life_pct)
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, p["color"])

    def _render_ui(self):
        # Top bar
        bar_surf = pygame.Surface((self.WIDTH, 30))
        bar_surf.set_alpha(180)
        bar_surf.fill((10,10,10))
        self.screen.blit(bar_surf, (0,0))
        
        wave_text = self.font_m.render(f"WAVE: {self.wave_number}/{self.WAVES_TO_WIN}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 5))
        
        money_text = self.font_m.render(f"$ {self.money}", True, (255, 223, 0))
        self.screen.blit(money_text, (self.WIDTH // 2 - money_text.get_width() // 2, 5))
        
        health_text = self.font_m.render(f"BASE: {int(self.base_health)}%", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (self.WIDTH - health_text.get_width() - 10, 5))
        
        # Placement UI
        if self.game_phase == "PLACEMENT":
            # Draw cursor and tower range
            cx, cy = self.cursor_pos
            spec = self.TOWER_SPECS[self.selected_tower_type]
            cursor_rect = pygame.Rect(cx * self.GRID_SIZE, cy * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            
            can_build = self.grid[cx, cy] == 0 and self.money >= spec["cost"]
            cursor_fill_color = (0, 255, 0, 50) if can_build else (255, 0, 0, 50)
            
            surf = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
            surf.fill(cursor_fill_color)
            self.screen.blit(surf, cursor_rect.topleft)
            pygame.draw.rect(self.screen, (255,255,255), cursor_rect, 1)

            # Draw range indicator
            pygame.gfxdraw.aacircle(self.screen, cursor_rect.centerx, cursor_rect.centery, spec["range"], (255,255,255,100))
            
            # Draw tower info panel
            panel_rect = pygame.Rect(10, self.HEIGHT - 70, 220, 60)
            pygame.draw.rect(self.screen, (10,10,10,180), panel_rect)
            pygame.draw.rect(self.screen, self.COLOR_TEXT, panel_rect, 1)
            
            name_txt = self.font_m.render(f"{spec['name']}", True, spec["color"])
            cost_txt = self.font_s.render(f"Cost: {spec['cost']}", True, self.COLOR_TEXT)
            info_txt = self.font_s.render(f"Dmg: {spec['damage']} | Range: {spec['range']}", True, self.COLOR_TEXT)
            self.screen.blit(name_txt, (panel_rect.x + 10, panel_rect.y + 5))
            self.screen.blit(cost_txt, (panel_rect.x + 120, panel_rect.y + 10))
            self.screen.blit(info_txt, (panel_rect.x + 10, panel_rect.y + 35))

            # Draw "Start Wave" button
            self.start_wave_button_rect = pygame.Rect(self.WIDTH - 160, self.HEIGHT - 60, 150, 50)
            pygame.draw.rect(self.screen, self.COLOR_BASE, self.start_wave_button_rect)
            pygame.draw.rect(self.screen, self.COLOR_TEXT, self.start_wave_button_rect, 2)
            start_txt = self.font_m.render("START WAVE", True, self.COLOR_TEXT)
            self.screen.blit(start_txt, (self.start_wave_button_rect.centerx - start_txt.get_width()//2, self.start_wave_button_rect.centery - start_txt.get_height()//2))

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "VICTORY!" if self.victory else "GAME OVER"
            end_text_color = self.COLOR_BASE if self.victory else self.COLOR_ZOMBIE
            end_text = self.font_l.render(end_text_str, True, end_text_color)
            self.screen.blit(end_text, (self.WIDTH//2 - end_text.get_width()//2, self.HEIGHT//2 - 50))
            
            score_text = self.font_m.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            self.screen.blit(score_text, (self.WIDTH//2 - score_text.get_width()//2, self.HEIGHT//2 + 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "money": self.money,
            "base_health": self.base_health,
            "game_phase": self.game_phase,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Use a separate display for human play
    pygame.display.set_caption("Tower Defense")
    human_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    action = env.action_space.sample() # Start with a no-op
    action.fill(0)
    
    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        mov_action = 0 # None
        if keys[pygame.K_UP]: mov_action = 1
        elif keys[pygame.K_DOWN]: mov_action = 2
        elif keys[pygame.K_LEFT]: mov_action = 3
        elif keys[pygame.K_RIGHT]: mov_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([mov_action, space_action, shift_action])
        # --- End Human Controls ---

        obs, reward, terminated, truncated, info = env.step(action)

        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Phase: {info['game_phase']}")

        # Render for human
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS

    env.close()
    print("Game Over!")
    print(f"Final Info: {info}")