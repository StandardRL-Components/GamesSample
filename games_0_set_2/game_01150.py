
# Generated: 2025-08-27T16:11:38.972827
# Source Brief: brief_01150.md
# Brief Index: 1150

        
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
        "Controls: Arrow keys to move the placement cursor. "
        "Space to place a tower. Shift to cycle tower types."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from enemy waves by strategically placing towers. "
        "Survive 10 waves to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 30000  # Approx 16 minutes at 30fps
    CURSOR_SPEED = 8
    
    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_PATH = (40, 50, 60)
    COLOR_PATH_BORDER = (60, 70, 80)
    COLOR_BASE = (0, 200, 100)
    COLOR_BASE_GLOW = (0, 200, 100, 50)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (255, 50, 50, 60)
    COLOR_PROJECTILE = (100, 200, 255)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (20, 20, 20)
    COLOR_HEALTH_BAR = (50, 255, 50)
    COLOR_HEALTH_BAR_BG = (255, 50, 50)
    
    # Tower Definitions
    TOWER_SPECS = [
        {
            "name": "Gatling",
            "cost": 50,
            "range": 80,
            "damage": 2,
            "fire_rate": 5, # steps between shots
            "color": (0, 255, 255),
        },
        {
            "name": "Cannon",
            "cost": 125,
            "range": 120,
            "damage": 15,
            "fire_rate": 45,
            "color": (255, 255, 0),
        },
    ]

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
        
        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.np_random = None
        self.base_pos = None
        self.base_health = 0
        self.max_base_health = 0
        self.resources = 0
        self.path_waypoints = []
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = None
        self.selected_tower_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.current_wave_idx = -1
        self.wave_timer = 0
        self.enemies_in_wave = 0
        self.enemies_spawned_this_wave = 0
        
        # Initialize state
        self.reset()
        
        # Validate implementation
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False

        self.max_base_health = 100
        self.base_health = self.max_base_health
        self.resources = 150
        
        self.base_pos = (self.SCREEN_WIDTH - 40, self.SCREEN_HEIGHT / 2)
        self._generate_path()

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=float)
        self.selected_tower_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self.current_wave_idx = -1
        self.wave_timer = 150 # Time until the first wave
        self.enemies_in_wave = 0
        self.enemies_spawned_this_wave = 0

        return self._get_observation(), self._get_info()

    def _generate_path(self):
        self.path_waypoints = []
        start_y = self.np_random.uniform(50, self.SCREEN_HEIGHT - 50)
        self.path_waypoints.append(np.array([-20, start_y]))
        
        x_coords = sorted(self.np_random.uniform(100, self.SCREEN_WIDTH - 150, size=3))
        
        for x in x_coords:
            y = self.np_random.uniform(50, self.SCREEN_HEIGHT - 50)
            self.path_waypoints.append(np.array([x, y]))
        
        self.path_waypoints.append(np.array([self.base_pos[0], self.base_pos[1]]))

    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing and just return the final state.
            reward = 0
            terminated = True
            return self._get_observation(), reward, terminated, False, self._get_info()

        reward = -0.001 # Small penalty for time passing
        
        self._handle_input(action)
        self._update_waves()
        
        step_reward = self._update_game_logic()
        reward += step_reward

        self.steps += 1
        self.score += reward
        
        terminated = self._check_termination()

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_action, shift_action = action[0], action[1] == 1, action[2] == 1

        # --- Move Cursor ---
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT)

        # --- Place Tower (on key press) ---
        if space_action and not self.prev_space_held:
            self._place_tower()
        self.prev_space_held = space_action

        # --- Cycle Tower Type (on key press) ---
        if shift_action and not self.prev_shift_held:
            self.selected_tower_idx = (self.selected_tower_idx + 1) % len(self.TOWER_SPECS)
        self.prev_shift_held = shift_action

    def _place_tower(self):
        spec = self.TOWER_SPECS[self.selected_tower_idx]
        if self.resources >= spec["cost"]:
            # Check for valid placement (not on path or other towers)
            is_valid_placement = True
            if self._is_on_path(self.cursor_pos, 20):
                is_valid_placement = False
            for t in self.towers:
                if np.linalg.norm(self.cursor_pos - t["pos"]) < 20:
                    is_valid_placement = False
                    break
            
            if is_valid_placement:
                self.resources -= spec["cost"]
                self.towers.append({
                    "pos": self.cursor_pos.copy(),
                    "spec": spec,
                    "cooldown": 0,
                    "target": None
                })
                # sfx: tower_place.wav

    def _is_on_path(self, pos, tolerance):
        for i in range(len(self.path_waypoints) - 1):
            p1 = self.path_waypoints[i]
            p2 = self.path_waypoints[i+1]
            d = np.linalg.norm(np.cross(p2-p1, p1-pos))/np.linalg.norm(p2-p1) if np.linalg.norm(p2-p1) != 0 else np.linalg.norm(p1-pos)
            if d < tolerance:
                # Check if point is between segment endpoints
                dot_product = np.dot(pos-p1, p2-p1)
                if 0 <= dot_product <= np.dot(p2-p1, p2-p1):
                    return True
        return False

    def _update_waves(self):
        is_wave_active = len(self.enemies) > 0 or self.enemies_spawned_this_wave < self.enemies_in_wave
        
        if self.current_wave_idx > 9 and not is_wave_active:
            self.game_won = True
            return

        if not is_wave_active and self.current_wave_idx <= 9:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                # Start next wave
                if self.current_wave_idx >= 0: # Don't give reward for clearing nothing
                    # sfx: wave_complete.wav
                    self.score += 10 # Wave survived reward
                self.current_wave_idx += 1
                self.enemies_in_wave = 10 + self.current_wave_idx * 3
                self.enemies_spawned_this_wave = 0
                self.wave_timer = 15 # Spawn interval
        elif self.enemies_spawned_this_wave < self.enemies_in_wave:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self._spawn_enemy()
                self.wave_timer = max(5, 15 - self.current_wave_idx) # Spawn faster in later waves

    def _spawn_enemy(self):
        # sfx: enemy_spawn.wav
        self.enemies_spawned_this_wave += 1
        difficulty_mod = 1 + (self.current_wave_idx * 0.05)
        self.enemies.append({
            "pos": self.path_waypoints[0].copy(),
            "max_health": 10 * difficulty_mod,
            "health": 10 * difficulty_mod,
            "speed": self.np_random.uniform(1.0, 1.5) * difficulty_mod,
            "waypoint_idx": 1,
            "value": 2 + self.current_wave_idx
        })

    def _update_game_logic(self):
        reward = 0
        
        # --- Update Towers ---
        for tower in self.towers:
            tower["cooldown"] = max(0, tower["cooldown"] - 1)
            if tower["cooldown"] == 0:
                target = None
                min_dist = tower["spec"]["range"]
                for enemy in self.enemies:
                    dist = np.linalg.norm(tower["pos"] - enemy["pos"])
                    if dist < min_dist:
                        min_dist = dist
                        target = enemy
                
                if target:
                    # sfx: tower_shoot.wav
                    self.projectiles.append({
                        "pos": tower["pos"].copy(),
                        "target": target,
                        "spec": tower["spec"]
                    })
                    tower["cooldown"] = tower["spec"]["fire_rate"]

        # --- Update Projectiles ---
        for proj in self.projectiles[:]:
            if proj["target"] not in self.enemies: # Target is already dead
                self.projectiles.remove(proj)
                continue
            
            direction = proj["target"]["pos"] - proj["pos"]
            dist = np.linalg.norm(direction)
            if dist < 8: # Hit
                # sfx: enemy_hit.wav
                proj["target"]["health"] -= proj["spec"]["damage"]
                reward += 0.01 # Small reward for hitting
                self._create_particles(proj["pos"], self.COLOR_PROJECTILE, 5)
                self.projectiles.remove(proj)
            else:
                proj["pos"] += (direction / dist) * 12 # Projectile speed

        # --- Update Enemies ---
        for enemy in self.enemies[:]:
            if enemy["health"] <= 0:
                # sfx: enemy_die.wav
                reward += 1 # Reward for kill
                self.resources += enemy["value"]
                self._create_particles(enemy["pos"], self.COLOR_ENEMY, 10)
                self.enemies.remove(enemy)
                continue

            target_waypoint = self.path_waypoints[enemy["waypoint_idx"]]
            direction = target_waypoint - enemy["pos"]
            dist = np.linalg.norm(direction)

            if dist < enemy["speed"]:
                enemy["pos"] = target_waypoint.copy()
                if enemy["waypoint_idx"] < len(self.path_waypoints) - 1:
                    enemy["waypoint_idx"] += 1
                else: # Reached base
                    # sfx: base_damage.wav
                    self.base_health -= 10
                    self._create_particles(enemy["pos"], self.COLOR_BASE, 15, 2)
                    self.enemies.remove(enemy)
            else:
                enemy["pos"] += (direction / dist) * enemy["speed"]
        
        # --- Update Particles ---
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
                
        return reward

    def _create_particles(self, pos, color, count, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            self.particles.append({
                "pos": pos.copy(),
                "vel": np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                "life": self.np_random.integers(10, 20),
                "max_life": 20,
                "color": color
            })

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
            self.score -= 100 # Penalty for losing
            return True
        if self.game_won:
            self.game_over = True
            self.score += 100 # Bonus for winning
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # --- Render Path ---
        for i in range(len(self.path_waypoints) - 1):
            p1 = tuple(self.path_waypoints[i].astype(int))
            p2 = tuple(self.path_waypoints[i+1].astype(int))
            pygame.draw.line(self.screen, self.COLOR_PATH_BORDER, p1, p2, 30)
            pygame.draw.line(self.screen, self.COLOR_PATH, p1, p2, 26)

        # --- Render Base ---
        base_rect = pygame.Rect(self.base_pos[0]-15, self.base_pos[1]-15, 30, 30)
        pygame.gfxdraw.filled_circle(self.screen, int(self.base_pos[0]), int(self.base_pos[1]), 25, self.COLOR_BASE_GLOW)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect, border_radius=3)
        
        # --- Render Towers ---
        for tower in self.towers:
            pos = tower["pos"].astype(int)
            spec = tower["spec"]
            # Draw range indicator
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], spec["range"], (*spec["color"], 50))
            # Draw tower body
            points = [
                (pos[0], pos[1] - 8),
                (pos[0] - 7, pos[1] + 5),
                (pos[0] + 7, pos[1] + 5)
            ]
            pygame.draw.polygon(self.screen, spec["color"], points)
            pygame.draw.aalines(self.screen, (*spec["color"], 150), True, points)

        # --- Render Enemies ---
        for enemy in self.enemies:
            pos = enemy["pos"].astype(int)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 6, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 6, (0,0,0))
            # Health bar
            health_ratio = enemy["health"] / enemy["max_health"]
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (pos[0]-6, pos[1]-12, 12, 3))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (pos[0]-6, pos[1]-12, 12 * health_ratio, 3))

        # --- Render Projectiles ---
        for proj in self.projectiles:
            pos = proj["pos"].astype(int)
            pygame.draw.rect(self.screen, self.COLOR_PROJECTILE, (pos[0]-2, pos[1]-2, 4, 4))
        
        # --- Render Particles ---
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            color = (*p["color"], alpha)
            size = int(3 * (p["life"] / p["max_life"]))
            if size > 0:
                rect = pygame.Rect(int(p["pos"][0] - size/2), int(p["pos"][1] - size/2), size, size)
                pygame.draw.rect(self.screen, color, rect)

        # --- Render Cursor ---
        if not self.game_over:
            pos = self.cursor_pos.astype(int)
            spec = self.TOWER_SPECS[self.selected_tower_idx]
            # Range indicator
            range_color = (*spec["color"], 100) if self.resources >= spec["cost"] else (255, 0, 0, 100)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], spec["range"], range_color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], spec["range"], (*range_color[:3], 20))
            # Cursor crosshair
            pygame.draw.line(self.screen, self.COLOR_CURSOR, (pos[0]-10, pos[1]), (pos[0]+10, pos[1]), 1)
            pygame.draw.line(self.screen, self.COLOR_CURSOR, (pos[0], pos[1]-10), (pos[0], pos[1]+10), 1)

    def _render_text(self, text, pos, font, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        text_rect = text_surf.get_rect(topleft=pos)
        shadow_rect = shadow_surf.get_rect(topleft=(pos[0]+1, pos[1]+1))
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # --- Base Health Bar ---
        bar_width = 150
        bar_height = 15
        health_ratio = max(0, self.base_health / self.max_base_health)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, bar_width * health_ratio, bar_height))
        self._render_text(f"BASE HP", (15, 11), self.font_small, (0,0,0))
        
        # --- Resources ---
        self._render_text(f"RESOURCES: ${self.resources}", (10, 35), self.font_small)

        # --- Wave Info ---
        wave_text = f"WAVE: {self.current_wave_idx + 1} / 10" if self.current_wave_idx < 10 else "ALL WAVES CLEARED"
        self._render_text(wave_text, (self.SCREEN_WIDTH - 170, 10), self.font_small)

        # --- Score ---
        self._render_text(f"SCORE: {int(self.score)}", (self.SCREEN_WIDTH - 170, 35), self.font_small)
        
        # --- Selected Tower ---
        if not self.game_over:
            spec = self.TOWER_SPECS[self.selected_tower_idx]
            cost_color = self.COLOR_TEXT if self.resources >= spec["cost"] else self.COLOR_ENEMY
            self._render_text(f"Selected: {spec['name']}", (10, self.SCREEN_HEIGHT - 45), self.font_small)
            self._render_text(f"Cost: ${spec['cost']}", (10, self.SCREEN_HEIGHT - 25), self.font_small, color=cost_color)

        # --- Game Over / Win Screen ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WON!" if self.game_won else "GAME OVER"
            text_surf = self.font_large.render(message, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "current_wave": self.current_wave_idx + 1,
            "game_won": self.game_won,
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    # Game loop
    running = True
    total_reward = 0
    
    action = env.action_space.sample() # Start with a random action
    action.fill(0) # But actually do nothing at first

    while running:
        # Event handling (for closing the window and keyboard input)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Keyboard input mapping
        keys = pygame.key.get_pressed()
        
        # Movement
        action[0] = 0 # No movement
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        # Buttons
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
        
        # Control the frame rate
        env.clock.tick(30)
        
    env.close()