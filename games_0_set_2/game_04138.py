
# Generated: 2025-08-28T01:32:04.440753
# Source Brief: brief_04138.md
# Brief Index: 4138

        
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

    # User-facing control string
    user_guide = (
        "Controls: Arrow keys to move the placement cursor. "
        "Shift to cycle tower types. Space to place a tower."
    )

    # User-facing description of the game
    game_description = (
        "A top-down tower defense game. Strategically place towers to defend your base "
        "from waves of incoming enemies. Survive all 10 waves to win."
    )

    # Frames advance only when an action is received
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)

        # Colors
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_PATH = (40, 50, 60)
        self.COLOR_PATH_BORDER = (60, 70, 80)
        self.COLOR_BASE = (0, 150, 50)
        self.COLOR_BASE_BORDER = (0, 200, 80)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BG = (100, 0, 0)
        self.COLOR_HEALTH_FG = (0, 255, 0)
        
        # Game constants
        self.MAX_STEPS = 3000
        self.TOTAL_WAVES = 10
        self.CURSOR_SPEED = 10
        
        # Game assets (defined in code)
        self.path = [
            (-20, 200), (80, 200), (80, 80), (240, 80), (240, 320),
            (400, 320), (400, 150), (560, 150), (560, 250), (self.WIDTH + 20, 250)
        ]
        self.base_pos = (self.WIDTH - 40, 250)
        self.base_size = 30
        
        self.tower_types = [
            {
                "name": "Gatling", "cost": 75, "range": 80, "damage": 2, "fire_rate": 8,
                "color": (0, 150, 255), "proj_color": (100, 200, 255), "proj_speed": 8
            },
            {
                "name": "Cannon", "cost": 150, "range": 120, "damage": 10, "fire_rate": 40,
                "color": (200, 50, 200), "proj_color": (255, 100, 255), "proj_speed": 6
            }
        ]

        # Initialize state variables
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Game state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        
        # Player state
        self.base_health = 100
        self.coins = 150
        self.cursor_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        self.selected_tower_type_idx = 0
        
        # Entity lists
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        # Wave management
        self.wave_number = 0
        self.time_to_next_wave = 150 # Start first wave after 5 seconds (150 steps at 30fps)
        self.wave_in_progress = False

        # Input handling
        self.shift_pressed_last_frame = False
        self.space_pressed_last_frame = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self.steps += 1
        reward = -0.001  # Small penalty for time passing

        # --- Handle player input ---
        self._handle_input(movement, space_held, shift_held)
        
        # --- Update game logic ---
        self._update_towers()
        self._update_projectiles(reward)
        reward += self._update_enemies()
        self._update_particles()
        reward += self._update_waves()

        # --- Check for termination ---
        terminated = self._check_termination()
        if terminated and not self.win:
            reward -= 10.0 # Large penalty for losing
        elif self.win:
            reward += 20.0 # Large reward for winning

        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT)

        # Cycle tower type (on press)
        if shift_held and not self.shift_pressed_last_frame:
            self.selected_tower_type_idx = (self.selected_tower_type_idx + 1) % len(self.tower_types)
        self.shift_pressed_last_frame = shift_held
        
        # Place tower (on press)
        if space_held and not self.space_pressed_last_frame:
            self._try_place_tower()
        self.space_pressed_last_frame = space_held

    def _try_place_tower(self):
        tower_spec = self.tower_types[self.selected_tower_type_idx]
        if self.coins >= tower_spec["cost"] and self._is_valid_placement(self.cursor_pos):
            self.coins -= tower_spec["cost"]
            self.towers.append({
                "pos": self.cursor_pos.copy(),
                "spec": tower_spec,
                "cooldown": 0,
            })
            # // Sound: tower_place.wav
            self._create_particles(self.cursor_pos, 15, tower_spec["color"], 20, 2)

    def _is_valid_placement(self, pos):
        # Check proximity to path
        for i in range(len(self.path) - 1):
            p1 = np.array(self.path[i])
            p2 = np.array(self.path[i+1])
            d = np.linalg.norm(np.cross(p2-p1, p1-pos))/np.linalg.norm(p2-p1) if np.linalg.norm(p2-p1) > 0 else np.linalg.norm(p1-pos)
            if d < 30: return False
        
        # Check proximity to other towers
        for tower in self.towers:
            if np.linalg.norm(pos - tower["pos"]) < 20: return False
        
        # Check proximity to base
        if np.linalg.norm(pos - self.base_pos) < self.base_size + 15: return False
            
        return True

    def _update_towers(self):
        for tower in self.towers:
            if tower["cooldown"] > 0:
                tower["cooldown"] -= 1
                continue
            
            target = self._find_target(tower)
            if target:
                self.projectiles.append({
                    "pos": tower["pos"].copy(),
                    "target": target,
                    "spec": tower["spec"]
                })
                tower["cooldown"] = tower["spec"]["fire_rate"]
                # // Sound: tower_fire.wav
                self._create_particles(tower["pos"], 3, tower["spec"]["proj_color"], 5, 1)

    def _find_target(self, tower):
        in_range_enemies = []
        for enemy in self.enemies:
            dist = np.linalg.norm(tower["pos"] - enemy["pos"])
            if dist <= tower["spec"]["range"]:
                in_range_enemies.append(enemy)
        
        # Target enemy furthest along the path
        if not in_range_enemies:
            return None
        return max(in_range_enemies, key=lambda e: e["path_progress"])

    def _update_projectiles(self, reward):
        for proj in self.projectiles[:]:
            target_pos = proj["target"]["pos"]
            direction = target_pos - proj["pos"]
            dist = np.linalg.norm(direction)
            
            if dist < proj["spec"]["proj_speed"]:
                # Hit target
                proj["target"]["health"] -= proj["spec"]["damage"]
                self.projectiles.remove(proj)
                # // Sound: enemy_hit.wav
                self._create_particles(target_pos, 8, (255, 255, 100), 10, 1.5)
            else:
                # Move projectile
                proj["pos"] += (direction / dist) * proj["spec"]["proj_speed"]

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            # Check for death
            if enemy["health"] <= 0:
                reward += 0.5  # Reward for killing an enemy
                self.coins += enemy["bounty"]
                self.enemies.remove(enemy)
                # // Sound: enemy_death.wav
                self._create_particles(enemy["pos"], 20, (255, 80, 80), 25, 3)
                continue

            # Move enemy along path
            path_idx = enemy["path_index"]
            if path_idx >= len(self.path) - 1:
                # Reached the base
                self.base_health -= enemy["damage"]
                self.base_health = max(0, self.base_health)
                self.enemies.remove(enemy)
                # // Sound: base_damage.wav
                self._create_particles(self.base_pos, 30, (255, 0, 0), 40, 4)
                continue

            target_waypoint = np.array(self.path[path_idx + 1])
            direction = target_waypoint - enemy["pos"]
            dist = np.linalg.norm(direction)

            if dist < enemy["speed"]:
                enemy["path_index"] += 1
                enemy["path_progress"] += dist
            else:
                enemy["pos"] += (direction / dist) * enemy["speed"]
                enemy["path_progress"] += enemy["speed"]
        return reward
    
    def _update_waves(self):
        reward = 0
        if not self.wave_in_progress and not self.game_over:
            self.time_to_next_wave -= 1
            if self.time_to_next_wave <= 0:
                self.wave_number += 1
                if self.wave_number > self.TOTAL_WAVES:
                    self.win = True
                    self.game_over = True
                else:
                    self._spawn_wave()
                    self.wave_in_progress = True
        
        if self.wave_in_progress and not self.enemies:
            reward += 1.0 # Reward for clearing a wave
            self.wave_in_progress = False
            self.time_to_next_wave = 240 # 8 seconds between waves
        return reward

    def _spawn_wave(self):
        num_enemies = 5 + self.wave_number * 2
        health = 10 + self.wave_number * 5
        speed = 1.0 + self.wave_number * 0.1
        bounty = 5 + self.wave_number
        damage = 5 + self.wave_number
        
        for i in range(num_enemies):
            self.enemies.append({
                "pos": np.array(self.path[0]) - np.array([i * 20, 0]), # Stagger spawn
                "health": health,
                "max_health": health,
                "speed": speed,
                "bounty": bounty,
                "damage": damage,
                "path_index": 0,
                "path_progress": -i * 20,
                "color": (200 - self.wave_number * 10, 20, 20)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            p["radius"] -= p["decay"]
            if p["life"] <= 0 or p["radius"] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, count, color, life, speed_scale):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 1.5) * speed_scale
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": random.randint(int(life/2), life),
                "radius": random.uniform(2, 5),
                "color": color,
                "decay": random.uniform(0.05, 0.1)
            })

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
        return self.game_over
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "coins": self.coins, "wave": self.wave_number}

    def _render_game(self):
        # Draw path
        pygame.draw.lines(self.screen, self.COLOR_PATH_BORDER, False, self.path, 28)
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path, 24)

        # Draw base
        base_rect = pygame.Rect(0, 0, self.base_size*2, self.base_size*2)
        base_rect.center = self.base_pos
        pygame.draw.rect(self.screen, self.COLOR_BASE_BORDER, base_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect.inflate(-6, -6), border_radius=3)

        # Draw towers and their ranges (if cursor is near)
        for tower in self.towers:
            pos_int = tower["pos"].astype(int)
            color = tower["spec"]["color"]
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 8, color)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], 8, tuple(c*0.8 for c in color))
            if np.linalg.norm(self.cursor_pos - tower["pos"]) < 30:
                 pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], tower["spec"]["range"], (*color, 60))

        # Draw projectiles
        for proj in self.projectiles:
            pos_int = proj["pos"].astype(int)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 3, proj["spec"]["proj_color"])
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], 3, (255,255,255))
            
        # Draw enemies and health bars
        for enemy in self.enemies:
            pos_int = enemy["pos"].astype(int)
            pygame.gfxdraw.filled_trigon(self.screen, pos_int[0], pos_int[1]-7, pos_int[0]-6, pos_int[1]+5, pos_int[0]+6, pos_int[1]+5, enemy["color"])
            pygame.gfxdraw.aatrigon(self.screen, pos_int[0], pos_int[1]-7, pos_int[0]-6, pos_int[1]+5, pos_int[0]+6, pos_int[1]+5, (255, 255, 255))
            
            # Health bar
            health_ratio = enemy["health"] / enemy["max_health"]
            bar_w = 16
            bar_h = 3
            bar_x = pos_int[0] - bar_w / 2
            bar_y = pos_int[1] - 15
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (bar_x, bar_y, bar_w * health_ratio, bar_h))

        # Draw particles
        for p in self.particles:
            pos_int = p["pos"].astype(int)
            alpha = int(255 * (p["life"] / p["life_max"] if "life_max" in p else 1))
            color = (*p["color"], alpha)
            if p["radius"] > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(p["radius"]), color)

        # Draw cursor and range indicator
        cursor_pos_int = self.cursor_pos.astype(int)
        tower_spec = self.tower_types[self.selected_tower_type_idx]
        is_valid = self.coins >= tower_spec["cost"] and self._is_valid_placement(self.cursor_pos)
        cursor_color = (0, 255, 0) if is_valid else (255, 0, 0)
        
        pygame.gfxdraw.aacircle(self.screen, cursor_pos_int[0], cursor_pos_int[1], tower_spec["range"], (*cursor_color, 80))
        pygame.gfxdraw.filled_circle(self.screen, cursor_pos_int[0], cursor_pos_int[1], 3, cursor_color)
        
    def _render_ui(self):
        # Base Health Bar
        health_bar_width = 150
        health_ratio = self.base_health / 100
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (10, 10, health_bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (10, 10, health_bar_width * health_ratio, 20))
        health_text = self.font_ui.render(f"BASE: {self.base_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Coins
        coins_text = self.font_ui.render(f"COINS: {self.coins}", True, self.COLOR_TEXT)
        self.screen.blit(coins_text, (self.WIDTH - 120, 12))

        # Wave
        wave_str = f"WAVE: {self.wave_number}/{self.TOTAL_WAVES}" if self.wave_in_progress else f"WAVE {self.wave_number+1} in {self.time_to_next_wave//30}s"
        if self.win: wave_str = "VICTORY!"
        if self.game_over and not self.win: wave_str = "DEFEAT"
        wave_text = self.font_ui.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH / 2 - wave_text.get_width() / 2, 12))

        # Selected Tower UI
        tower_spec = self.tower_types[self.selected_tower_type_idx]
        tower_info_y = self.HEIGHT - 40
        tower_text = self.font_ui.render(f"Selected: {tower_spec['name']} (Cost: {tower_spec['cost']})", True, self.COLOR_TEXT)
        self.screen.blit(tower_text, (10, tower_info_y))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        msg = "VICTORY!" if self.win else "GAME OVER"
        color = (100, 255, 100) if self.win else (255, 100, 100)
        
        text = self.font_game_over.render(msg, True, color)
        text_rect = text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        self.screen.blit(text, text_rect)

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv()
    
    # --- Manual Play ---
    # To play manually, you need a way to get keyboard inputs.
    # This is a simple example.
    
    obs, info = env.reset()
    done = False
    
    # Setup a window to see the game
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    while running:
        # Action defaults to no-op
        action = [0, 0, 0] # move, space, shift
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        # Map keys to actions
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Waves Survived: {info['wave']-1}")
            pygame.time.wait(3000) # Wait 3 seconds
            obs, info = env.reset()

        # Since auto_advance is False, we control the tick rate here for manual play
        env.clock.tick(30)
        
    env.close()