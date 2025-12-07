
# Generated: 2025-08-27T17:17:04.993426
# Source Brief: brief_01484.md
# Brief Index: 1484

        
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
        "Controls: Use arrow keys to move the placement cursor. "
        "Press Space to build a short-range Cannon Tower. "
        "Hold Shift to build a long-range Laser Tower."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of enemies by strategically placing two types of towers "
        "in this top-down tower defense game. Survive all 10 waves to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Game settings
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 30 * 180 # 3 minutes max
    NUM_WAVES = 10

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_PATH = (40, 50, 70)
    COLOR_BASE = (100, 200, 100)
    COLOR_HEALTH_GREEN = (50, 220, 50)
    COLOR_HEALTH_RED = (220, 50, 50)
    
    COLOR_ENEMY = (255, 70, 70)
    COLOR_ENEMY_GLOW = (255, 120, 120)

    COLOR_TOWER1 = (80, 150, 255) # Cannon
    COLOR_TOWER2 = (255, 220, 80) # Laser
    
    COLOR_CURSOR = (255, 255, 255)
    COLOR_PLACEMENT_VALID = (0, 255, 0, 50)
    
    COLOR_TEXT = (220, 220, 240)
    
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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 64)
        
        # Path and placement setup
        self._define_path_and_placements()

        # Initialize state variables
        self.reset()
        
        # Run validation
        # self.validate_implementation() # Uncomment to run validation on init

    def _define_path_and_placements(self):
        self.path_points = [
            pygame.math.Vector2(-20, 100),
            pygame.math.Vector2(150, 100),
            pygame.math.Vector2(150, 300),
            pygame.math.Vector2(450, 300),
            pygame.math.Vector2(450, 100),
            pygame.math.Vector2(self.SCREEN_WIDTH + 20, 100)
        ]
        
        self.placement_spots = []
        self.placement_grid_size = (12, 8)
        self.placement_cell_size = 50
        for y in range(self.placement_grid_size[1]):
            for x in range(self.placement_grid_size[0]):
                px, py = 25 + x * self.placement_cell_size, 25 + y * self.placement_cell_size
                is_on_path = False
                for i in range(len(self.path_points) - 1):
                    p1 = self.path_points[i]
                    p2 = self.path_points[i+1]
                    rect = pygame.Rect(min(p1.x, p2.x) - 25, min(p1.y, p2.y) - 25, abs(p1.x - p2.x) + 50, abs(p1.y - p2.y) + 50)
                    if rect.collidepoint(px, py):
                        is_on_path = True
                        break
                if not is_on_path:
                    self.placement_spots.append(pygame.math.Vector2(px, py))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.base_health = 100
        self.max_base_health = 100
        
        self.current_wave = 0
        self.enemies_to_spawn = []
        self.wave_spawn_timer = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_index = len(self.placement_spots) // 2
        self.occupied_placement_indices = set()
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self._start_next_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        # --- Handle Input and Player Actions ---
        if not self.game_over:
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            self._handle_cursor_movement(movement)
            
            if space_held and not self.prev_space_held:
                if self._place_tower(1):
                    pass # Optional: reward for placing a tower
            
            if shift_held and not self.prev_shift_held:
                if self._place_tower(2):
                    pass # Optional: reward for placing a tower

            self.prev_space_held = space_held
            self.prev_shift_held = shift_held

        # --- Update Game State ---
        if not self.game_over:
            self._update_wave_spawning()
            
            hit_reward, kill_reward = self._update_projectiles()
            reward += hit_reward + kill_reward
            
            base_damage_reward = self._update_enemies()
            reward += base_damage_reward
            
            self._update_towers()
            self._update_particles()

            # --- Wave Management ---
            if not self.enemies and not self.enemies_to_spawn:
                if self.current_wave <= self.NUM_WAVES:
                    if self.current_wave > 0: # Don't reward for wave 0
                        reward += 50
                        self.score += 250
                    if self.current_wave == self.NUM_WAVES:
                        self.win = True
                        self.game_over = True
                        reward += 100
                    else:
                        self._start_next_wave()

        self.steps += 1
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    # --- Private Helper Methods: Game Logic ---

    def _handle_cursor_movement(self, movement):
        # This is a bit complex to map grid movement to a 1D list index
        # A simple but effective way is to find the nearest neighbor in the direction
        current_pos = self.placement_spots[self.cursor_index]
        best_candidate = -1
        min_dist_sq = float('inf')

        if movement == 0: return # no-op

        for i, pos in enumerate(self.placement_spots):
            if i == self.cursor_index: continue
            
            vec = pos - current_pos
            if movement == 1 and vec.y < 0 and abs(vec.y) > abs(vec.x): # Up
                if vec.length_squared() < min_dist_sq:
                    min_dist_sq = vec.length_squared()
                    best_candidate = i
            elif movement == 2 and vec.y > 0 and abs(vec.y) > abs(vec.x): # Down
                 if vec.length_squared() < min_dist_sq:
                    min_dist_sq = vec.length_squared()
                    best_candidate = i
            elif movement == 3 and vec.x < 0 and abs(vec.x) > abs(vec.y): # Left
                 if vec.length_squared() < min_dist_sq:
                    min_dist_sq = vec.length_squared()
                    best_candidate = i
            elif movement == 4 and vec.x > 0 and abs(vec.x) > abs(vec.y): # Right
                 if vec.length_squared() < min_dist_sq:
                    min_dist_sq = vec.length_squared()
                    best_candidate = i
        
        if best_candidate != -1:
            self.cursor_index = best_candidate

    def _place_tower(self, tower_type):
        if self.cursor_index not in self.occupied_placement_indices:
            pos = self.placement_spots[self.cursor_index]
            if tower_type == 1: # Cannon
                tower = {
                    "pos": pos, "type": 1, "range": 80, "damage": 25, 
                    "fire_rate": 0.8, "cooldown": 0, "size": 12
                }
            else: # Laser
                tower = {
                    "pos": pos, "type": 2, "range": 180, "damage": 8, 
                    "fire_rate": 0.3, "cooldown": 0, "size": 12
                }
            self.towers.append(tower)
            self.occupied_placement_indices.add(self.cursor_index)
            # sfx: place_tower.wav
            return True
        return False

    def _start_next_wave(self):
        self.current_wave += 1
        if self.current_wave > self.NUM_WAVES: return

        num_enemies = 2 + self.current_wave
        base_health = 40 + self.current_wave * 10
        speed = 0.8 + self.current_wave * 0.05
        
        self.enemies_to_spawn = []
        for _ in range(num_enemies):
            enemy = {
                "pos": pygame.math.Vector2(self.path_points[0]),
                "health": base_health,
                "max_health": base_health,
                "speed": speed + self.np_random.uniform(-0.1, 0.1),
                "path_index": 0,
                "value": 10 + self.current_wave * 2
            }
            self.enemies_to_spawn.append(enemy)
        self.wave_spawn_timer = 0

    def _update_wave_spawning(self):
        if self.enemies_to_spawn:
            self.wave_spawn_timer -= 1 / self.FPS
            if self.wave_spawn_timer <= 0:
                self.enemies.append(self.enemies_to_spawn.pop(0))
                self.wave_spawn_timer = 1.0 # Spawn one enemy per second
                # sfx: enemy_spawn.wav

    def _update_towers(self):
        for tower in self.towers:
            tower["cooldown"] = max(0, tower["cooldown"] - 1 / self.FPS)
            if tower["cooldown"] <= 0:
                target = None
                min_dist = tower["range"] ** 2
                for enemy in self.enemies:
                    dist_sq = tower["pos"].distance_squared_to(enemy["pos"])
                    if dist_sq < min_dist:
                        min_dist = dist_sq
                        target = enemy
                
                if target:
                    tower["cooldown"] = tower["fire_rate"]
                    projectile = {
                        "start_pos": pygame.math.Vector2(tower["pos"]),
                        "end_pos": pygame.math.Vector2(target["pos"]),
                        "pos": pygame.math.Vector2(tower["pos"]),
                        "target": target,
                        "damage": tower["damage"],
                        "speed": 15 if tower["type"] == 1 else 1000, # Lasers are instant
                        "type": tower["type"]
                    }
                    self.projectiles.append(projectile)
                    # sfx: cannon_fire.wav or laser_fire.wav

    def _update_projectiles(self):
        hit_reward = 0
        kill_reward = 0
        
        for p in self.projectiles[:]:
            if p["target"] not in self.enemies:
                self.projectiles.remove(p)
                continue

            p["end_pos"].update(p["target"]["pos"])
            direction = (p["end_pos"] - p["pos"])
            
            if direction.length_squared() < (p["speed"] * 5/self.FPS)**2 or p["type"] == 2:
                # Hit
                p["target"]["health"] -= p["damage"]
                hit_reward += 0.1
                # sfx: enemy_hit.wav
                for _ in range(5):
                    self._create_particle(p["pos"], self.COLOR_HEALTH_RED)

                if p["target"]["health"] <= 0:
                    kill_reward += 1
                    self.score += p["target"]["value"]
                    # sfx: enemy_explode.wav
                    for _ in range(20):
                        self._create_particle(p["target"]["pos"], self.COLOR_ENEMY_GLOW)
                    self.enemies.remove(p["target"])

                self.projectiles.remove(p)
            else:
                p["pos"] += direction.normalize() * p["speed"] * 50/self.FPS
                
        return hit_reward, kill_reward

    def _update_enemies(self):
        base_damage_reward = 0
        for enemy in self.enemies[:]:
            if enemy["path_index"] >= len(self.path_points) - 1:
                self.base_health -= 10
                base_damage_reward -= 10
                self.enemies.remove(enemy)
                # sfx: base_damage.wav
                for _ in range(30):
                    self._create_particle(self.path_points[-1] + (-30, -50), self.COLOR_HEALTH_RED)
                continue

            target_pos = self.path_points[enemy["path_index"] + 1]
            direction = (target_pos - enemy["pos"])
            
            if direction.length_squared() < (enemy["speed"])**2:
                enemy["path_index"] += 1
            else:
                enemy["pos"] += direction.normalize() * enemy["speed"]
        return base_damage_reward

    def _create_particle(self, pos, color):
        self.particles.append({
            "pos": pygame.math.Vector2(pos),
            "vel": pygame.math.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)),
            "lifespan": self.np_random.uniform(0.3, 0.8),
            "color": color
        })
        
    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1 / self.FPS
            p["vel"] *= 0.95 # Damping
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
            self.base_health = 0
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
        return self.game_over

    # --- Private Helper Methods: Rendering ---

    def _render_path(self):
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, [tuple(p) for p in self.path_points], 30)

    def _render_base(self):
        base_rect = pygame.Rect(self.SCREEN_WIDTH - 40, 50, 40, 100)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)
        pygame.draw.rect(self.screen, tuple(c*0.8 for c in self.COLOR_BASE), base_rect, 3)

    def _render_placement_spots_and_cursor(self):
        cursor_pos = self.placement_spots[self.cursor_index]
        for i, pos in enumerate(self.placement_spots):
            size = 30
            rect = pygame.Rect(pos.x - size/2, pos.y - size/2, size, size)
            if i not in self.occupied_placement_indices:
                s = pygame.Surface((size, size), pygame.SRCALPHA)
                s.fill(self.COLOR_PLACEMENT_VALID)
                self.screen.blit(s, rect.topleft)
        
        # Draw cursor
        cursor_size = 40
        cursor_rect = pygame.Rect(cursor_pos.x - cursor_size/2, cursor_pos.y - cursor_size/2, cursor_size, cursor_size)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=5)

    def _render_towers(self):
        for tower in self.towers:
            color = self.COLOR_TOWER1 if tower["type"] == 1 else self.COLOR_TOWER2
            pos = (int(tower["pos"].x), int(tower["pos"].y))
            pygame.draw.rect(self.screen, color, (pos[0]-tower["size"], pos[1]-tower["size"], tower["size"]*2, tower["size"]*2), border_radius=3)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in color), (pos[0]-tower["size"], pos[1]-tower["size"], tower["size"]*2, tower["size"]*2), 2, border_radius=3)

    def _render_enemies(self):
        for enemy in self.enemies:
            pos = (int(enemy["pos"].x), int(enemy["pos"].y))
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 12, self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 12, self.COLOR_ENEMY_GLOW)
            # Main body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 10, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 10, self.COLOR_ENEMY)
            
            # Health bar
            hb_width = 24
            hb_height = 4
            hb_pos_x = pos[0] - hb_width / 2
            hb_pos_y = pos[1] - 20
            health_ratio = enemy["health"] / enemy["max_health"]
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (hb_pos_x, hb_pos_y, hb_width, hb_height))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (hb_pos_x, hb_pos_y, int(hb_width * health_ratio), hb_height))

    def _render_projectiles(self):
        for p in self.projectiles:
            start = (int(p["start_pos"].x), int(p["start_pos"].y))
            end = (int(p["pos"].x), int(p["pos"].y))
            if p["type"] == 1: # Cannon Ball
                pygame.draw.circle(self.screen, (255,255,255), end, 4)
            else: # Laser Beam
                pygame.draw.aaline(self.screen, self.COLOR_TOWER2, start, end, 2)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(p["lifespan"] * 255)))
            color = p["color"] + (alpha,)
            size = int(p["lifespan"] * 5)
            if size > 0:
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (size, size), size)
                self.screen.blit(s, (int(p["pos"].x - size), int(p["pos"].y - size)))

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Wave
        wave_text = self.font_small.render(f"WAVE: {self.current_wave}/{self.NUM_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))
        
        # Base Health
        health_text = self.font_small.render("BASE HEALTH", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, self.SCREEN_HEIGHT - 30))
        hb_width = 200
        hb_height = 15
        hb_pos_x = 120
        hb_pos_y = self.SCREEN_HEIGHT - 28
        health_ratio = self.base_health / self.max_base_health
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (hb_pos_x, hb_pos_y, hb_width, hb_height), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (hb_pos_x, hb_pos_y, int(hb_width * health_ratio), hb_height), border_radius=3)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            
            status_text_str = "YOU WIN!" if self.win else "GAME OVER"
            status_text = self.font_large.render(status_text_str, True, self.COLOR_TEXT)
            text_rect = status_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(status_text, text_rect)

    # --- Gymnasium Required Methods ---
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_path()
        self._render_base()
        self._render_placement_spots_and_cursor()
        self._render_towers()
        self._render_enemies()
        self._render_projectiles()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "current_wave": self.current_wave,
            "towers_placed": len(self.towers),
            "enemies_on_screen": len(self.enemies)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        ''' Call this at the end of __init__ to verify implementation. '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        # Test game-specific assertions
        self.reset()
        assert self.base_health == self.max_base_health
        self._start_next_wave()
        assert self.current_wave == 2 # reset calls it once, then we call it again
        assert len(self.enemies_to_spawn) == 2 + self.current_wave
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # To display the game, we need a Pygame screen
    pygame.display.set_caption("Tower Defense")
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    action = [0, 0, 0] # no-op, no space, no shift
    
    print("--- Game Start ---")
    print(env.game_description)
    print(env.user_guide)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- Player Controls ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space, shift]
        
        # --- Environment Step ---
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Base HP: {info['base_health']}")
            if terminated:
                print("--- Game Over ---")
                print(f"Final Score: {info['score']}")

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)
        
    env.close()