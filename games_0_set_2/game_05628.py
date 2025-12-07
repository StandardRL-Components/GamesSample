
# Generated: 2025-08-28T05:35:22.547992
# Source Brief: brief_05628.md
# Brief Index: 5628

        
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
        "Controls: Use ← and → to move, ↑ to jump. Hold Space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Leap and blast through a procedurally generated forest, taking down swarming enemies before they deplete your health in this side-scrolling action game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.WORLD_WIDTH = self.SCREEN_WIDTH * 3
        self.FPS = 30
        self.GROUND_Y = 350
        self.MAX_STEPS = 1000
        self.TOTAL_ENEMIES_TO_DEFEAT = 15

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GROUND = (60, 45, 35)
        self.COLOR_PLAYER = (50, 220, 50)
        self.COLOR_PLAYER_DMG = (255, 100, 100)
        self.COLOR_ENEMY = (210, 40, 40)
        self.COLOR_PROJECTILE = (255, 255, 100)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_HEART = (255, 50, 50)
        self.BG_COLORS = [(30, 40, 50), (40, 50, 60)]

        # Physics & Gameplay
        self.GRAVITY = 0.8
        self.PLAYER_SPEED = 6
        self.PLAYER_JUMP_STRENGTH = 15
        self.PROJECTILE_SPEED = 15
        self.SHOOT_COOLDOWN_FRAMES = 8
        self.PLAYER_INVINCIBILITY_FRAMES = 60
        
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
        self.font_large = pygame.font.Font(None, 48)
        
        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player = {}
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.background_elements = []
        self.camera_x = 0
        self.enemies_defeated = 0
        self.base_enemy_spawn_interval = 90 # frames
        self.current_enemy_spawn_interval = 0
        self.enemy_spawn_timer = 0
        self.base_enemy_speed = 1.5
        self.current_enemy_speed = 0
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        player_size = (20, 40)
        self.player = {
            "rect": pygame.Rect(self.WORLD_WIDTH // 2, self.GROUND_Y - player_size[1], player_size[0], player_size[1]),
            "vel": [0, 0],
            "on_ground": True,
            "direction": 1, # 1 for right, -1 for left
            "health": 3,
            "shoot_cooldown": 0,
            "invincibility_timer": 0
        }
        
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.enemies_defeated = 0
        
        self.current_enemy_spawn_interval = self.base_enemy_spawn_interval
        self.enemy_spawn_timer = self.current_enemy_spawn_interval
        self.current_enemy_speed = self.base_enemy_speed

        self.camera_x = self.player["rect"].centerx - self.SCREEN_WIDTH / 2
        
        if not self.background_elements:
            for i in range(100):
                self.background_elements.append({
                    "x": self.np_random.uniform(0, self.WORLD_WIDTH),
                    "y": self.np_random.uniform(self.GROUND_Y - 150, self.GROUND_Y),
                    "radius": self.np_random.uniform(5, 30),
                    "depth": self.np_random.uniform(0.2, 0.7), # for parallax
                    "color": self.BG_COLORS[self.np_random.integers(0, len(self.BG_COLORS))]
                })
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        
        self._handle_input(action)
        reward += self._update_state()
        
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated:
            if self.player["health"] <= 0:
                reward -= 100
            elif self.enemies_defeated >= self.TOTAL_ENEMIES_TO_DEFEAT:
                reward += 100
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        # Horizontal Movement
        if movement == 3: # Left
            self.player["vel"][0] = -self.PLAYER_SPEED
            self.player["direction"] = -1
        elif movement == 4: # Right
            self.player["vel"][0] = self.PLAYER_SPEED
            self.player["direction"] = 1
        else:
            self.player["vel"][0] = 0
            
        # Jumping
        if movement == 1 and self.player["on_ground"]:
            self.player["vel"][1] = -self.PLAYER_JUMP_STRENGTH
            self.player["on_ground"] = False
            # sfx: player_jump
        
        # Shooting
        if space_held and self.player["shoot_cooldown"] == 0:
            proj_y = self.player["rect"].centery
            proj_x = self.player["rect"].right if self.player["direction"] == 1 else self.player["rect"].left
            self.projectiles.append({
                "rect": pygame.Rect(proj_x, proj_y - 2, 8, 4),
                "vel_x": self.PROJECTILE_SPEED * self.player["direction"]
            })
            self.player["shoot_cooldown"] = self.SHOOT_COOLDOWN_FRAMES
            self._create_particles(self.player["rect"].center, 3, self.COLOR_PROJECTILE, 1, 3) # Muzzle flash
            # sfx: player_shoot

    def _update_state(self):
        step_reward = 0

        # Update timers
        if self.player["shoot_cooldown"] > 0: self.player["shoot_cooldown"] -= 1
        if self.player["invincibility_timer"] > 0: self.player["invincibility_timer"] -= 1

        # Update player
        self.player["vel"][1] += self.GRAVITY
        self.player["rect"].x += self.player["vel"][0]
        self.player["rect"].y += self.player["vel"][1]
        
        # Player world boundaries
        self.player["rect"].left = max(0, self.player["rect"].left)
        self.player["rect"].right = min(self.WORLD_WIDTH, self.player["rect"].right)

        # Player ground collision
        if self.player["rect"].bottom >= self.GROUND_Y:
            self.player["rect"].bottom = self.GROUND_Y
            self.player["vel"][1] = 0
            self.player["on_ground"] = True

        # Update projectiles and check collisions
        projectiles_to_remove = []
        enemies_to_remove = []
        
        for p in self.projectiles:
            p["rect"].x += p["vel_x"]
            if not (0 < p["rect"].centerx < self.WORLD_WIDTH):
                projectiles_to_remove.append(p)
                step_reward -= 0.1 # Miss penalty
                continue
            
            for e in self.enemies:
                if p["rect"].colliderect(e["rect"]):
                    projectiles_to_remove.append(p)
                    enemies_to_remove.append(e)
                    self.score += 1
                    self.enemies_defeated += 1
                    step_reward += 1.1 # +1 for defeat, +0.1 for hit
                    self._create_particles(e["rect"].center, 15, self.COLOR_ENEMY, 2, 5)
                    # sfx: enemy_explode
                    break
        
        # Update enemies and check player collision
        for e in self.enemies:
            e["rect"].x += e["vel_x"]
            if e["rect"].left < e["patrol_min"] or e["rect"].right > e["patrol_max"]:
                e["vel_x"] *= -1

            if self.player["rect"].colliderect(e["rect"]) and self.player["invincibility_timer"] == 0:
                self.player["health"] -= 1
                self.player["invincibility_timer"] = self.PLAYER_INVINCIBILITY_FRAMES
                self._create_particles(self.player["rect"].center, 10, self.COLOR_PLAYER_DMG, 2, 4)
                # sfx: player_hurt
        
        # Clean up destroyed entities
        self.projectiles = [p for p in self.projectiles if p not in projectiles_to_remove]
        self.enemies = [e for e in self.enemies if e not in enemies_to_remove]

        # Update particles
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1

        # Spawn enemies
        self._spawn_enemies()

        # Update difficulty
        if self.steps > 0 and self.steps % 50 == 0:
             self.current_enemy_spawn_interval = max(20, self.current_enemy_spawn_interval * 0.99)
        if self.steps > 0 and self.steps % 200 == 0:
            self.current_enemy_speed += 0.02

        # Update camera
        target_camera_x = self.player["rect"].centerx - self.SCREEN_WIDTH / 2
        self.camera_x += (target_camera_x - self.camera_x) * 0.1 # Smooth camera
        self.camera_x = max(0, min(self.WORLD_WIDTH - self.SCREEN_WIDTH, self.camera_x))

        return step_reward

    def _spawn_enemies(self):
        self.enemy_spawn_timer -= 1
        num_enemies_alive = len(self.enemies)
        num_enemies_spawned = num_enemies_alive + self.enemies_defeated
        
        if self.enemy_spawn_timer <= 0 and num_enemies_spawned < self.TOTAL_ENEMIES_TO_DEFEAT:
            self.enemy_spawn_timer = self.current_enemy_spawn_interval
            
            side = self.np_random.choice([-1, 1])
            enemy_size = (25, 25)
            patrol_width = 150
            
            if side == -1: # Spawn left
                spawn_x = self.camera_x - enemy_size[0]
                patrol_min = spawn_x - patrol_width
                patrol_max = spawn_x + patrol_width
            else: # Spawn right
                spawn_x = self.camera_x + self.SCREEN_WIDTH + enemy_size[0]
                patrol_min = spawn_x - patrol_width
                patrol_max = spawn_x + patrol_width
            
            self.enemies.append({
                "rect": pygame.Rect(spawn_x, self.GROUND_Y - enemy_size[1], enemy_size[0], enemy_size[1]),
                "vel_x": self.current_enemy_speed * side,
                "patrol_min": patrol_min,
                "patrol_max": patrol_max
            })
            
    def _create_particles(self, pos, count, color, min_speed, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(min_speed, max_speed)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(10, 20),
                "color": color
            })

    def _check_termination(self):
        return (self.player["health"] <= 0 or 
                self.enemies_defeated >= self.TOTAL_ENEMIES_TO_DEFEAT or
                self.steps >= self.MAX_STEPS)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # Render parallax background
        for bg_el in self.background_elements:
            render_x = (bg_el["x"] - self.camera_x * bg_el["depth"]) % self.WORLD_WIDTH
            pygame.gfxdraw.filled_circle(
                self.screen, int(render_x), int(bg_el["y"]), int(bg_el["radius"]), bg_el["color"]
            )
        
        # Render ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y))
        
        # Render game elements
        self._render_enemies()
        self._render_projectiles()
        self._render_player()
        self._render_particles()
        
        # Render UI overlay
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_player(self):
        render_rect = self.player["rect"].copy()
        render_rect.x -= self.camera_x
        
        color = self.COLOR_PLAYER
        if self.player["invincibility_timer"] > 0 and (self.steps // 3) % 2 == 0:
            color = self.COLOR_PLAYER_DMG

        pygame.draw.rect(self.screen, color, render_rect, border_radius=3)
        
        # Gun nozzle
        gun_x = render_rect.right if self.player["direction"] == 1 else render_rect.left - 5
        gun_y = render_rect.centery - 2
        pygame.draw.rect(self.screen, (100, 100, 100), (gun_x, gun_y, 5, 4))

    def _render_enemies(self):
        for e in self.enemies:
            render_rect = e["rect"].copy()
            render_rect.x -= self.camera_x
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, render_rect, border_radius=5)

    def _render_projectiles(self):
        for p in self.projectiles:
            render_rect = p["rect"].copy()
            render_rect.x -= self.camera_x
            pygame.draw.rect(self.screen, self.COLOR_PROJECTILE, render_rect, border_radius=2)
    
    def _render_particles(self):
        for p in self.particles:
            render_pos = [p["pos"][0] - self.camera_x, p["pos"][1]]
            alpha = max(0, 255 * (p["life"] / 20.0))
            color = (*p["color"], alpha)
            size = max(1, int(p["life"] / 4))
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size, size), size)
            self.screen.blit(temp_surf, (int(render_pos[0]-size), int(render_pos[1]-size)))

    def _render_ui(self):
        # Health
        for i in range(self.player["health"]):
            pygame.gfxdraw.filled_circle(self.screen, 25 + i * 30, 25, 10, self.COLOR_HEART)
            pygame.gfxdraw.aacircle(self.screen, 25 + i * 30, 25, 10, self.COLOR_HEART)
        
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 15, 15))

        # Enemies defeated
        enemies_text = self.font_small.render(f"DEFEATED: {self.enemies_defeated}/{self.TOTAL_ENEMIES_TO_DEFEAT}", True, self.COLOR_TEXT)
        self.screen.blit(enemies_text, (self.SCREEN_WIDTH - enemies_text.get_width() - 15, 35))

        if self.game_over:
            result_text_str = "YOU WIN!" if self.enemies_defeated >= self.TOTAL_ENEMIES_TO_DEFEAT else "GAME OVER"
            result_text = self.font_large.render(result_text_str, True, self.COLOR_TEXT)
            text_rect = result_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(result_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player["health"],
            "enemies_defeated": self.enemies_defeated
        }

    def validate_implementation(self):
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
        assert self.player["health"] <= 3
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        assert len(self.enemies) <= self.TOTAL_ENEMIES_TO_DEFEAT
        assert self.score >= 0

        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()