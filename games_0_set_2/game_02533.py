
# Generated: 2025-08-28T05:09:26.128896
# Source Brief: brief_02533.md
# Brief Index: 2533

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: ←→ to move, ↑ to jump. Press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a jumping, shooting robot through a side-scrolling obstacle course filled with enemies to reach the exit."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    WORLD_WIDTH = 3200
    
    # Colors
    COLOR_BG = (25, 25, 40) # Dark blue
    COLOR_BG_LAYER_1 = (35, 35, 55)
    COLOR_BG_LAYER_2 = (45, 45, 70)
    COLOR_GROUND = (60, 60, 80)
    COLOR_PLAYER = (46, 204, 113) # Green
    COLOR_PLAYER_DMG = (231, 76, 60) # Red
    COLOR_ENEMY_1 = (231, 76, 60) # Red
    COLOR_ENEMY_2 = (230, 126, 34) # Orange
    COLOR_ENEMY_3 = (155, 89, 182) # Purple
    COLOR_PLAYER_PROJ = (241, 196, 15) # Yellow
    COLOR_ENEMY_PROJ = (236, 112, 255) # Magenta
    COLOR_EXIT = (52, 152, 219) # Blue
    COLOR_UI_TEXT = (236, 240, 241)
    COLOR_UI_FRAME = (44, 62, 80)
    
    # Physics & Gameplay
    FPS = 30
    GRAVITY = 0.8
    GROUND_Y = SCREEN_HEIGHT - 50
    MAX_STEPS = 2000

    # Player
    PLAYER_SPEED = 6
    PLAYER_JUMP_POWER = -14
    PLAYER_MAX_HEALTH = 5
    PLAYER_SHOOT_COOLDOWN = 6 # frames
    PLAYER_INVINCIBILITY_FRAMES = 45

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        # State variables are initialized in reset()
        self.player = {}
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.bg_elements = []
        self.camera_x = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.screen_shake = 0
        self.prev_space_held = False
        self.np_random = None

        self.reset()

        # Run validation check
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # Seed the internal random number generator
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.camera_x = 0
        self.screen_shake = 0
        self.prev_space_held = False

        self.player = {
            "rect": pygame.Rect(100, self.GROUND_Y - 40, 30, 40),
            "vel_x": 0,
            "vel_y": 0,
            "on_ground": False,
            "health": self.PLAYER_MAX_HEALTH,
            "shoot_cooldown": 0,
            "damage_timer": 0,
        }
        
        self.enemies = []
        self.projectiles = []
        self.particles = []

        # Generate parallax background elements
        self.bg_elements = []
        for _ in range(50):
            self.bg_elements.append({
                "x": self.np_random.integers(0, self.WORLD_WIDTH),
                "y": self.np_random.integers(0, self.GROUND_Y),
                "w": self.np_random.integers(20, 100),
                "h": self.np_random.integers(5, 15),
                "layer": self.np_random.choice([1, 2])
            })

        self.exit_rect = pygame.Rect(self.WORLD_WIDTH - 100, self.GROUND_Y - 100, 20, 100)

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False
        
        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if not self.game_over:
            # --- Update game logic ---
            # Handle player input and calculate movement reward
            reward += self._handle_input(movement, space_held)
            
            # Update game objects
            self._update_player()
            self._update_enemies()
            self._update_projectiles()
            self._update_particles()
            
            # Handle collisions and calculate event-based rewards
            reward += self._handle_collisions()
            
            # Spawn new enemies
            self._spawn_enemies()

            # Update camera
            self._update_camera()

        # --- Check termination conditions ---
        self.steps += 1
        if self.player["health"] <= 0 and not self.game_over:
            self.game_over = True
            terminated = True
            reward = -100.0
            self._create_explosion(self.player["rect"].center, 50, self.COLOR_PLAYER)
        
        if self.player["rect"].colliderect(self.exit_rect) and not self.game_over:
            self.game_over = True
            self.game_won = True
            terminated = True
            reward = 100.0

        if self.steps >= self.MAX_STEPS:
            terminated = True
            if not self.game_over: # Time out without winning or losing
                reward = -10.0 # Penalty for not finishing

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        reward = 0
        # Horizontal Movement
        if movement == 3:  # Left
            self.player["vel_x"] = -self.PLAYER_SPEED
            reward -= 0.01 # Small penalty for moving away from goal
        elif movement == 4:  # Right
            self.player["vel_x"] = self.PLAYER_SPEED
            reward += 0.01 # Small reward for moving towards goal
        else:
            self.player["vel_x"] = 0

        # Jumping
        if movement == 1 and self.player["on_ground"]:
            self.player["vel_y"] = self.PLAYER_JUMP_POWER
            self.player["on_ground"] = False
            # Sound: Jump
            # Create jump particles
            for _ in range(10):
                self.particles.append(self._create_particle(
                    self.player["rect"].midbottom,
                    color=self.COLOR_PLAYER,
                    velocity=[self.np_random.uniform(-1, 1), self.np_random.uniform(0, 2)],
                    lifespan=10
                ))

        # Shooting
        if space_held and not self.prev_space_held and self.player["shoot_cooldown"] == 0:
            self.player["shoot_cooldown"] = self.PLAYER_SHOOT_COOLDOWN
            proj_y = self.player["rect"].centery - 5 # Fire from 'chest'
            self.projectiles.append({
                "rect": pygame.Rect(self.player["rect"].right, proj_y, 10, 4),
                "vel_x": 15,
                "owner": "player",
            })
            # Sound: Player shoot
            # Muzzle flash
            self.particles.append(self._create_particle(
                (self.player["rect"].right, proj_y),
                color=self.COLOR_PLAYER_PROJ,
                velocity=[0,0],
                lifespan=2,
                size_range=(10,12)
            ))
        
        self.prev_space_held = space_held
        return reward

    def _update_player(self):
        # Apply gravity
        self.player["vel_y"] += self.GRAVITY
        
        # Move player
        self.player["rect"].x += int(self.player["vel_x"])
        self.player["rect"].y += int(self.player["vel_y"])
        
        # World boundaries
        self.player["rect"].left = max(0, self.player["rect"].left)
        self.player["rect"].right = min(self.WORLD_WIDTH, self.player["rect"].right)
        
        # Ground collision
        if self.player["rect"].bottom >= self.GROUND_Y:
            self.player["rect"].bottom = self.GROUND_Y
            self.player["vel_y"] = 0
            self.player["on_ground"] = True
        else:
            self.player["on_ground"] = False

        # Update timers
        if self.player["shoot_cooldown"] > 0:
            self.player["shoot_cooldown"] -= 1
        if self.player["damage_timer"] > 0:
            self.player["damage_timer"] -= 1
        if self.screen_shake > 0:
            self.screen_shake -= 1

    def _update_enemies(self):
        # Difficulty scaling
        base_speed = 1.5 + (self.steps / 200) * 0.05

        for enemy in self.enemies:
            # Type 1: Patrol
            if enemy["type"] == 1:
                enemy["rect"].x += enemy["dir"] * base_speed
                if enemy["rect"].x < enemy["patrol_min"] or enemy["rect"].x > enemy["patrol_max"]:
                    enemy["dir"] *= -1
            # Type 2: Sine wave flyer
            elif enemy["type"] == 2:
                enemy["rect"].x += base_speed
                enemy["rect"].y = enemy["start_y"] + math.sin(enemy["rect"].x * 0.02) * 40
            # Type 3: Stationary shooter
            elif enemy["type"] == 3:
                enemy["shoot_cooldown"] -= 1
                dist_to_player = abs(self.player["rect"].centerx - enemy["rect"].centerx)
                if dist_to_player < 400 and enemy["shoot_cooldown"] <= 0:
                    enemy["shoot_cooldown"] = self.np_random.integers(90, 120)
                    proj_y = enemy["rect"].centery
                    proj_vel = -10 if self.player["rect"].centerx < enemy["rect"].centerx else 10
                    self.projectiles.append({
                        "rect": pygame.Rect(enemy["rect"].centerx, proj_y, 8, 8),
                        "vel_x": proj_vel,
                        "owner": "enemy"
                    })
                    # Sound: Enemy shoot

    def _update_projectiles(self):
        for proj in self.projectiles:
            proj["rect"].x += proj["vel_x"]
        
        # Remove off-screen projectiles
        self.projectiles = [p for p in self.projectiles if 0 < p["rect"].centerx < self.WORLD_WIDTH]

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs Enemies
        for proj in self.projectiles[:]:
            if proj["owner"] == "player":
                for enemy in self.enemies[:]:
                    if proj["rect"].colliderect(enemy["rect"]):
                        enemy["health"] -= 1
                        if proj in self.projectiles: self.projectiles.remove(proj)
                        if enemy["health"] <= 0:
                            # Sound: Explosion
                            self._create_explosion(enemy["rect"].center, 20, enemy['color'])
                            self.enemies.remove(enemy)
                            self.score += 10
                            reward += 1.0
                        break
        
        # Enemy projectiles vs Player
        if self.player["damage_timer"] == 0:
            for proj in self.projectiles[:]:
                if proj["owner"] == "enemy" and self.player["rect"].colliderect(proj["rect"]):
                    self.player["health"] -= 1
                    self.player["damage_timer"] = self.PLAYER_INVINCIBILITY_FRAMES
                    self.screen_shake = 10
                    reward -= 1.0
                    if proj in self.projectiles: self.projectiles.remove(proj)
                    # Sound: Player hit
                    break
        
        # Player vs Enemies
        if self.player["damage_timer"] == 0:
            for enemy in self.enemies:
                if self.player["rect"].colliderect(enemy["rect"]):
                    self.player["health"] -= 1
                    self.player["damage_timer"] = self.PLAYER_INVINCIBILITY_FRAMES
                    self.screen_shake = 10
                    reward -= 1.0
                    # Sound: Player hit
                    # Apply knockback
                    self.player["vel_y"] = -5
                    self.player["vel_x"] = -self.player["vel_x"] if self.player["vel_x"] != 0 else -self.PLAYER_SPEED * np.sign(enemy["rect"].centerx - self.player["rect"].centerx)
                    break
        
        return reward

    def _spawn_enemies(self):
        # Spawn rate increases over time
        spawn_chance = 0.01 * (1.01 ** (self.steps // 50))
        if self.np_random.random() < spawn_chance:
            spawn_x = self.camera_x + self.SCREEN_WIDTH + 50
            if spawn_x > self.WORLD_WIDTH - 200: # Don't spawn too close to the exit
                return
            
            enemy_type = self.np_random.integers(1, 4)
            
            # Type 1: Patrol
            if enemy_type == 1:
                self.enemies.append({
                    "rect": pygame.Rect(spawn_x, self.GROUND_Y - 30, 30, 30),
                    "type": 1, "health": 1, "dir": -1,
                    "patrol_min": spawn_x - 100, "patrol_max": spawn_x + 100,
                    "color": self.COLOR_ENEMY_1
                })
            # Type 2: Flyer
            elif enemy_type == 2:
                start_y = self.GROUND_Y - self.np_random.integers(80, 150)
                self.enemies.append({
                    "rect": pygame.Rect(spawn_x, start_y, 35, 20),
                    "type": 2, "health": 2, "start_y": start_y,
                    "color": self.COLOR_ENEMY_2
                })
            # Type 3: Shooter
            elif enemy_type == 3:
                self.enemies.append({
                    "rect": pygame.Rect(spawn_x, self.GROUND_Y - 50, 40, 50),
                    "type": 3, "health": 3, "shoot_cooldown": self.np_random.integers(60, 90),
                    "color": self.COLOR_ENEMY_3
                })

    def _update_camera(self):
        target_camera_x = self.player["rect"].centerx - self.SCREEN_WIDTH / 2
        # Smooth camera follow
        self.camera_x += (target_camera_x - self.camera_x) * 0.1
        # Clamp camera to world bounds
        self.camera_x = max(0, min(self.camera_x, self.WORLD_WIDTH - self.SCREEN_WIDTH))

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Apply screen shake
        render_offset_x, render_offset_y = 0, 0
        if self.screen_shake > 0:
            render_offset_x = self.np_random.integers(-5, 6)
            render_offset_y = self.np_random.integers(-5, 6)

        # Render all game elements
        self._render_background(render_offset_x, render_offset_y)
        self._render_game(render_offset_x, render_offset_y)
        
        # Render UI overlay (not affected by shake)
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self, ox, oy):
        # Parallax layer 1 (moves slower)
        for bg in self.bg_elements:
            if bg["layer"] == 1:
                x = (bg["x"] - self.camera_x * 0.5) % self.WORLD_WIDTH
                pygame.draw.rect(self.screen, self.COLOR_BG_LAYER_1, (x + ox, bg["y"] + oy, bg["w"], bg["h"]))

        # Parallax layer 2 (moves faster)
        for bg in self.bg_elements:
            if bg["layer"] == 2:
                x = (bg["x"] - self.camera_x * 0.8) % self.WORLD_WIDTH
                pygame.draw.rect(self.screen, self.COLOR_BG_LAYER_2, (x + ox, bg["y"] + oy, bg["w"], bg["h"]))
        
        # Ground
        ground_rect = pygame.Rect(ox, self.GROUND_Y + oy, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y)
        pygame.draw.rect(self.screen, self.COLOR_GROUND, ground_rect)

    def _render_game(self, ox, oy):
        cam_x = self.camera_x - ox
        cam_y = -oy

        # Exit
        exit_screen_rect = self.exit_rect.move(-cam_x, -cam_y)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_screen_rect)
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, exit_screen_rect, 2)

        # Enemies
        for enemy in self.enemies:
            enemy_screen_rect = enemy["rect"].move(-cam_x, -cam_y)
            if self.screen.get_rect().colliderect(enemy_screen_rect):
                pygame.draw.rect(self.screen, enemy["color"], enemy_screen_rect)

        # Player
        if self.player["health"] > 0:
            player_screen_rect = self.player["rect"].move(-cam_x, -cam_y)
            # Flash when invincible
            if self.player["damage_timer"] > 0 and (self.steps // 3) % 2 == 0:
                pass # Don't draw to make it flash
            else:
                color = self.COLOR_PLAYER_DMG if self.player["damage_timer"] > 0 else self.COLOR_PLAYER
                pygame.draw.rect(self.screen, color, player_screen_rect)
                # Player "eye"
                eye_rect = pygame.Rect(0,0,8,8)
                eye_rect.center = player_screen_rect.center
                eye_rect.y -= 8
                pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, eye_rect)

        # Projectiles
        for proj in self.projectiles:
            proj_screen_rect = proj["rect"].move(-cam_x, -cam_y)
            color = self.COLOR_PLAYER_PROJ if proj["owner"] == "player" else self.COLOR_ENEMY_PROJ
            pygame.draw.rect(self.screen, color, proj_screen_rect)

        # Particles
        for p in self.particles:
            pos = (int(p["pos"][0] - cam_x), int(p["pos"][1] - cam_y))
            size = int(p["size"] * (p["lifespan"] / p["max_lifespan"]))
            if size > 0:
                pygame.draw.circle(self.screen, p["color"], pos, size)

    def _render_ui(self):
        # Health Bar
        health_pct = max(0, self.player["health"] / self.PLAYER_MAX_HEALTH)
        bar_width = 150
        pygame.draw.rect(self.screen, self.COLOR_UI_FRAME, (10, 10, bar_width, 20))
        if health_pct > 0:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, (10, 10, int(bar_width * health_pct), 20))
        health_text = self.font_small.render(f"HP", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 12))
        
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Game Over / Win Text
        if self.game_over:
            msg = "YOU REACHED THE EXIT!" if self.game_won else "GAME OVER"
            color = self.COLOR_PLAYER if self.game_won else self.COLOR_ENEMY_1
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            pygame.draw.rect(self.screen, self.COLOR_BG, text_rect.inflate(20, 20))
            self.screen.blit(end_text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player["health"],
            "player_pos_x": self.player["rect"].x,
        }

    def _create_explosion(self, pos, num_particles, color):
        for _ in range(num_particles):
            self.particles.append(self._create_particle(pos, color))
    
    def _create_particle(self, pos, color, velocity=None, lifespan=None, size_range=(2,5)):
        if velocity is None:
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        else:
            vel = velocity
        
        life = lifespan if lifespan is not None else self.np_random.integers(15, 30)
        
        return {
            "pos": list(pos),
            "vel": vel,
            "lifespan": life,
            "max_lifespan": life,
            "color": color,
            "size": self.np_random.uniform(size_range[0], size_range[1])
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset(seed=42)
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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    
    # Use a different screen for display
    pygame.display.set_caption("GameEnv Test")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    terminated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        # Get keyboard input
        keys = pygame.key.get_pressed()
        
        # Map keys to action space
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment...")
                obs, info = env.reset(seed=random.randint(0, 10000))
                total_reward = 0
                terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        # Convert observation back to a Pygame surface for display
        # The observation is (H, W, C), but pygame wants (W, H) surface
        # and surfarray.make_surface expects (W, H, C) array
        obs_swapped = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(obs_swapped)
        display_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # Control the frame rate
        env.clock.tick(GameEnv.FPS)

    env.close()