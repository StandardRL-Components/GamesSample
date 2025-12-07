
# Generated: 2025-08-28T00:04:41.885790
# Source Brief: brief_03678.md
# Brief Index: 3678

        
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
        "Controls: Arrows to move cursor, Space to place selected tower, Shift to cycle tower types."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of geometric enemies by strategically placing defensive towers."
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
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # --- Game Constants ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_PATH = (40, 50, 80)
        self.COLOR_PATH_BORDER = (50, 65, 100)
        self.COLOR_BASE = (0, 180, 120)
        self.COLOR_BASE_BORDER = (100, 255, 220)
        self.COLOR_ENEMY = (220, 50, 50)
        self.COLOR_ENEMY_BORDER = (255, 150, 150)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_UI_BG = (30, 40, 60, 180)
        
        self.CURSOR_SPEED = 8
        self.MAX_STEPS = 5000 # Increased for longer games
        self.PLACEMENT_PHASE_DURATION = 30 * 10 # 10 seconds at 30fps
        self.MAX_WAVES = 10

        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 48)

        # Game path definition
        self.path_points = [
            (-20, 200), (100, 200), (100, 100), (400, 100),
            (400, 300), (200, 300), (200, 200), (540, 200), (660, 200)
        ]
        self.path_width = 40
        
        # Tower definitions
        self.TOWER_TYPES = {
            0: {
                "name": "Cannon", "cost": 100, "range": 80, "damage": 25, 
                "fire_rate": 40, "color": (255, 200, 0), "proj_speed": 8,
                "unlock_wave": 1
            },
            1: {
                "name": "Missile", "cost": 250, "range": 120, "damage": 75, 
                "fire_rate": 90, "color": (0, 150, 255), "proj_speed": 6,
                "unlock_wave": 3
            }
        }
        
        # Initialize state variables
        self.reset()
        
        # Run validation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.base_health = 100
        self.max_base_health = 100
        self.resources = 250
        
        self.wave_number = 0
        self.enemies_spawned_this_wave = 0
        self.enemies_on_screen = 0
        self.game_phase = "placement" # 'placement', 'wave_active', 'game_over', 'victory'
        self.placement_timer = self.PLACEMENT_PHASE_DURATION
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = np.array([self.screen_width / 2, self.screen_height / 2], dtype=float)
        self.selected_tower_type = 0
        self.available_tower_types = [0]
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        self.steps += 1

        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        
        reward += self._handle_input(movement, space_pressed, shift_pressed)
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Game Logic ---
        if self.game_phase == "placement":
            self.placement_timer -= 1
            if self.placement_timer <= 0:
                self.game_phase = "wave_active"
                self.wave_number += 1
                self._spawn_wave()
                if self.wave_number > 1:
                    reward += 50 # Wave survived reward
                    self.resources += 100 + self.wave_number * 10
                # Unlock new towers
                self.available_tower_types = [
                    t_id for t_id, t_info in self.TOWER_TYPES.items() 
                    if self.wave_number >= t_info["unlock_wave"]
                ]
                if self.selected_tower_type not in self.available_tower_types:
                    self.selected_tower_type = self.available_tower_types[0]


        elif self.game_phase == "wave_active":
            wave_update_rewards = self._update_wave_phase()
            reward += wave_update_rewards
            
            if self.enemies_spawned_this_wave == self._get_wave_enemy_count() and self.enemies_on_screen == 0:
                self.game_phase = "placement"
                self.placement_timer = self.PLACEMENT_PHASE_DURATION
                if self.wave_number >= self.MAX_WAVES:
                    self.game_phase = "victory"

        # --- Update Particles ---
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

        # --- Continuous Penalties ---
        if self.base_health < self.max_base_health:
            reward -= 0.01

        # --- Termination Check ---
        terminated = False
        if self.base_health <= 0:
            self.game_phase = "game_over"
            reward = -100
            terminated = True
        elif self.game_phase == "victory":
            reward += 100 # Final victory reward
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed, shift_pressed):
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.screen_width)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.screen_height)

        # Cycle tower type
        if shift_pressed:
            # Sfx: UI_Cycle.wav
            current_idx = self.available_tower_types.index(self.selected_tower_type)
            next_idx = (current_idx + 1) % len(self.available_tower_types)
            self.selected_tower_type = self.available_tower_types[next_idx]

        # Place tower
        if space_pressed:
            tower_info = self.TOWER_TYPES[self.selected_tower_type]
            if self.resources >= tower_info["cost"] and self._is_valid_placement(self.cursor_pos):
                # Sfx: Tower_Place.wav
                self.resources -= tower_info["cost"]
                self.towers.append({
                    "pos": self.cursor_pos.copy(),
                    "type": self.selected_tower_type,
                    "cooldown": 0,
                    "target": None
                })
                return 0.5 # Small reward for placing a tower
            else:
                # Sfx: Error.wav
                self._create_particles(self.cursor_pos, 10, self.COLOR_ENEMY, 1, 3, 15) # Error effect
                return -0.2 # Small penalty for invalid placement
        return 0

    def _update_wave_phase(self):
        reward = 0
        
        # Update Towers
        for tower in self.towers:
            tower_info = self.TOWER_TYPES[tower["type"]]
            tower["cooldown"] = max(0, tower["cooldown"] - 1)
            
            # Find new target if needed
            if tower["target"] is None or tower["target"] not in self.enemies:
                tower["target"] = None
                closest_enemy = None
                min_dist = float('inf')
                for enemy in self.enemies:
                    dist = np.linalg.norm(tower["pos"] - enemy["pos"])
                    if dist < tower_info["range"]:
                        if dist < min_dist:
                            min_dist = dist
                            closest_enemy = enemy
                tower["target"] = closest_enemy

            # Fire if ready and has target
            if tower["target"] and tower["cooldown"] == 0:
                # Sfx: Cannon_Fire.wav or Missile_Launch.wav
                tower["cooldown"] = tower_info["fire_rate"]
                self.projectiles.append({
                    "pos": tower["pos"].copy(),
                    "target": tower["target"],
                    "speed": tower_info["proj_speed"],
                    "damage": tower_info["damage"],
                    "type": tower["type"]
                })

        # Update Projectiles
        projectiles_to_remove = []
        for proj in self.projectiles:
            if proj["target"] not in self.enemies:
                projectiles_to_remove.append(proj)
                continue
            
            direction = proj["target"]["pos"] - proj["pos"]
            dist = np.linalg.norm(direction)
            if dist < proj["speed"]:
                # Hit target
                # Sfx: Impact.wav
                proj["target"]["health"] -= proj["damage"]
                reward += 0.1 # Damage reward
                self._create_particles(proj["pos"], 20, self.TOWER_TYPES[proj["type"]]["color"], 1, 4, 20)
                projectiles_to_remove.append(proj)
            else:
                proj["pos"] += (direction / dist) * proj["speed"]
        self.projectiles = [p for p in self.projectiles if p not in projectiles_to_remove]

        # Update Enemies
        enemies_to_remove = []
        for enemy in self.enemies:
            # Move enemy
            if enemy["path_index"] < len(self.path_points) - 1:
                target_point = np.array(self.path_points[enemy["path_index"] + 1])
                direction = target_point - enemy["pos"]
                dist = np.linalg.norm(direction)
                if dist < enemy["speed"]:
                    enemy["pos"] = target_point
                    enemy["path_index"] += 1
                else:
                    enemy["pos"] += (direction / dist) * enemy["speed"]
            else:
                # Reached base
                # Sfx: Base_Damage.wav
                self.base_health -= 10
                self.base_health = max(0, self.base_health)
                enemies_to_remove.append(enemy)
                self._create_particles(enemy["pos"], 30, self.COLOR_ENEMY, 2, 5, 25)
                reward -= 5

            # Check for death
            if enemy["health"] <= 0:
                # Sfx: Enemy_Explode.wav
                enemies_to_remove.append(enemy)
                reward += 1 # Defeat reward
                self.resources += 15
                self._create_particles(enemy["pos"], 25, self.COLOR_ENEMY_BORDER, 1, 3, 20)
                
        if enemies_to_remove:
            self.enemies = [e for e in self.enemies if e not in enemies_to_remove]
        
        self.enemies_on_screen = len(self.enemies)
        return reward
    
    def _spawn_wave(self):
        self.enemies_spawned_this_wave = self._get_wave_enemy_count()
        self.enemies_on_screen = self.enemies_spawned_this_wave
        
        enemy_health = 50 * (1.05 ** (self.wave_number - 1))
        enemy_speed = 1.5 * (1.05 ** (self.wave_number - 1))
        
        for i in range(self.enemies_spawned_this_wave):
            self.enemies.append({
                "pos": np.array(self.path_points[0]) + np.array([-i * 30, 0]),
                "health": enemy_health,
                "max_health": enemy_health,
                "speed": enemy_speed,
                "path_index": 0
            })
    
    def _get_wave_enemy_count(self):
        return 3 + self.wave_number * 2

    def _is_valid_placement(self, pos):
        # Check if on path
        for i in range(len(self.path_points) - 1):
            p1 = np.array(self.path_points[i])
            p2 = np.array(self.path_points[i+1])
            # Check distance to line segment
            l2 = np.sum((p1 - p2)**2)
            if l2 == 0.0:
                if np.linalg.norm(pos - p1) < self.path_width: return False
            t = max(0, min(1, np.dot(pos - p1, p2 - p1) / l2))
            projection = p1 + t * (p2 - p1)
            if np.linalg.norm(pos - projection) < self.path_width:
                return False
        
        # Check if too close to another tower
        for tower in self.towers:
            if np.linalg.norm(pos - tower["pos"]) < 40:
                return False
        
        return True

    def _create_particles(self, pos, count, color, min_speed, max_speed, life):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(min_speed, max_speed)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel / 10, # Slower visual speed
                'life': random.randint(life // 2, life),
                'color': color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw path
        for i in range(len(self.path_points) - 1):
            p1 = tuple(map(int, self.path_points[i]))
            p2 = tuple(map(int, self.path_points[i+1]))
            pygame.draw.line(self.screen, self.COLOR_PATH, p1, p2, self.path_width * 2)
        for i in range(len(self.path_points) - 1):
            p1 = tuple(map(int, self.path_points[i]))
            p2 = tuple(map(int, self.path_points[i+1]))
            pygame.draw.line(self.screen, self.COLOR_PATH_BORDER, p1, p2, self.path_width * 2 + 2)
            pygame.draw.line(self.screen, self.COLOR_PATH, p1, p2, self.path_width * 2)
        for p in self.path_points:
            pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), self.path_width, self.COLOR_PATH)
            pygame.gfxdraw.aacircle(self.screen, int(p[0]), int(p[1]), self.path_width, self.COLOR_PATH_BORDER)

        # Draw base
        base_rect = pygame.Rect(self.path_points[-1][0]-10, self.path_points[-1][1]-20, 20, 40)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BASE_BORDER, base_rect, 2, border_radius=5)

        # Draw towers
        for tower in self.towers:
            info = self.TOWER_TYPES[tower["type"]]
            pos = tuple(map(int, tower["pos"]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 15, self.COLOR_PATH_BORDER)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 12, info["color"])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 5, self.COLOR_BG)
            if tower["cooldown"] > info["fire_rate"] - 5: # Firing flash
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 18, (*info["color"], 100))

        # Draw enemies
        for enemy in self.enemies:
            pos = tuple(map(int, enemy["pos"]))
            size = 12
            rect = pygame.Rect(pos[0] - size, pos[1] - size, size*2, size*2)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_BORDER, rect, 2, border_radius=3)
            # Health bar
            health_pct = max(0, enemy["health"] / enemy["max_health"])
            bar_w = int(size * 2 * health_pct)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_BORDER, (rect.left, rect.top - 8, size*2, 5))
            pygame.draw.rect(self.screen, self.COLOR_BASE, (rect.left, rect.top - 8, bar_w, 5))

        # Draw projectiles
        for proj in self.projectiles:
            pos = tuple(map(int, proj["pos"]))
            color = self.TOWER_TYPES[proj["type"]]["color"]
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 4, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 4, (255,255,255))

        # Draw particles
        for p in self.particles:
            pos = tuple(map(int, p["pos"]))
            alpha = int(255 * (p['life'] / 20))
            color = (*p['color'], max(0, min(255, alpha)))
            temp_surf = pygame.Surface((4,4), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, 2, 2, 2, color)
            self.screen.blit(temp_surf, (pos[0]-2, pos[1]-2))

        # Draw cursor
        cursor_pos_int = tuple(map(int, self.cursor_pos))
        tower_info = self.TOWER_TYPES[self.selected_tower_type]
        is_valid = self._is_valid_placement(self.cursor_pos) and self.resources >= tower_info["cost"]
        cursor_color = self.COLOR_BASE if is_valid else self.COLOR_ENEMY
        
        s = pygame.Surface((tower_info["range"]*2, tower_info["range"]*2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(s, tower_info["range"], tower_info["range"], tower_info["range"], (*cursor_color, 30))
        pygame.gfxdraw.aacircle(s, tower_info["range"], tower_info["range"], tower_info["range"], (*cursor_color, 100))
        self.screen.blit(s, (cursor_pos_int[0] - tower_info["range"], cursor_pos_int[1] - tower_info["range"]))
        
        pygame.gfxdraw.filled_circle(self.screen, cursor_pos_int[0], cursor_pos_int[1], 10, cursor_color)
        pygame.gfxdraw.aacircle(self.screen, cursor_pos_int[0], cursor_pos_int[1], 10, (255,255,255))

    def _render_ui(self):
        # UI Background
        ui_panel = pygame.Surface((self.screen_width, 40), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))
        
        # Base Health
        health_text = self.font_medium.render("BASE HP", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 10))
        health_pct = max(0, self.base_health / self.max_base_health)
        pygame.draw.rect(self.screen, self.COLOR_ENEMY, (90, 12, 150, 16))
        pygame.draw.rect(self.screen, self.COLOR_BASE, (90, 12, 150 * health_pct, 16))

        # Resources
        res_text = self.font_medium.render(f"$ {self.resources}", True, (255, 200, 0))
        self.screen.blit(res_text, (260, 10))

        # Wave Info
        wave_str = f"WAVE {self.wave_number}/{self.MAX_WAVES}"
        if self.game_phase == 'placement':
            wave_str += f" (Starts in {int(self.placement_timer / 30) + 1}s)"
        wave_text = self.font_medium.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.screen_width - wave_text.get_width() - 10, 10))
        
        # Selected Tower
        tower_info = self.TOWER_TYPES[self.selected_tower_type]
        tower_name = self.font_small.render(f"Selected: {tower_info['name']}", True, self.COLOR_TEXT)
        tower_cost = self.font_small.render(f"Cost: ${tower_info['cost']}", True, self.COLOR_TEXT)
        self.screen.blit(tower_name, (10, self.screen_height - 45))
        self.screen.blit(tower_cost, (10, self.screen_height - 25))
        
        # Game Over / Victory Text
        if self.game_phase == "game_over":
            text = self.font_large.render("GAME OVER", True, self.COLOR_ENEMY)
            text_rect = text.get_rect(center=(self.screen_width/2, self.screen_height/2))
            self.screen.blit(text, text_rect)
        elif self.game_phase == "victory":
            text = self.font_large.render("VICTORY!", True, self.COLOR_BASE_BORDER)
            text_rect = text.get_rect(center=(self.screen_width/2, self.screen_height/2))
            self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "base_health": self.base_health,
            "resources": self.resources,
            "enemies_left": self.enemies_on_screen
        }
        
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

if __name__ == "__main__":
    # This block allows you to play the game manually
    # Requires pygame to be installed with video support
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    game_screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Tower Defense")
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while running:
        # Human input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        
        # Reset actions
        movement, space, shift = 0, 0, 0
        
        # Map keys to actions
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = np.array([movement, space, shift])

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render to screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        game_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Wave: {info['wave']}")
            # Wait for a moment before auto-resetting
            pygame.time.wait(3000)
            obs, info = env.reset()

        env.clock.tick(30) # 30 FPS

    pygame.quit()