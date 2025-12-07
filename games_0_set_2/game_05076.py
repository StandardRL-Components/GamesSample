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
        "Controls: Use ←→ to move, ↑ to jump, and Space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Control a robot in a side-scrolling action environment, blasting enemies to achieve total robotic domination."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Level Dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.LEVEL_WIDTH = 1280
        self.GROUND_Y = 350

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GROUND = (40, 45, 60)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_GUN = (200, 200, 200)
        self.COLOR_ENEMY = (255, 80, 80)
        self.COLOR_ENEMY_GUN = (200, 200, 200)
        self.COLOR_PLAYER_PROJ = (255, 255, 100)
        self.COLOR_ENEMY_PROJ = (200, 100, 255)
        self.COLOR_UI_TEXT = (230, 230, 230)
        self.COLOR_HEALTH_GREEN = (100, 220, 100)
        self.COLOR_HEALTH_RED = (220, 100, 100)
        
        # Physics Constants
        self.GRAVITY = 0.4
        self.PLAYER_SPEED = 6
        self.JUMP_STRENGTH = -10
        self.PROJECTILE_SPEED = 12
        
        # Game Settings
        self.MAX_STEPS = 1000
        self.NUM_ENEMIES = 20
        self.PLAYER_MAX_HEALTH = 100
        self.ENEMY_MAX_HEALTH = 10
        self.PLAYER_SHOOT_COOLDOWN = 10
        self.ENEMY_SHOOT_INTERVAL = 40

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
        self.font_large = pygame.font.Font(None, 36)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player = {}
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.camera_offset_x = 0
        self.prev_space_held = False
        self.background_buildings = []
        self.rng = np.random.default_rng()

        # self.reset() is called by the wrapper, but we can call it for validation
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # Use a new generator seeded with the provided seed
            self.rng = np.random.default_rng(seed)
        # If seed is None, we continue using the existing RNG

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.prev_space_held = False
        
        self._initialize_player()
        self._initialize_enemies()
        self._initialize_background()
        
        self.player_projectiles.clear()
        self.enemy_projectiles.clear()
        self.particles.clear()
        
        return self._get_observation(), self._get_info()

    def _initialize_player(self):
        self.player = {
            "pos": pygame.Vector2(self.LEVEL_WIDTH / 2, self.GROUND_Y),
            "vel": pygame.Vector2(0, 0),
            "health": self.PLAYER_MAX_HEALTH,
            "on_ground": True,
            "shoot_cooldown": 0,
            "facing_dir": 1, # 1 for right, -1 for left
            "size": pygame.Vector2(20, 40)
        }

    def _initialize_enemies(self):
        self.enemies.clear()
        for i in range(self.NUM_ENEMIES):
            x_pos = self.rng.integers(100, self.LEVEL_WIDTH - 100)
            self.enemies.append({
                "pos": pygame.Vector2(x_pos, self.GROUND_Y),
                "health": self.ENEMY_MAX_HEALTH,
                "size": pygame.Vector2(20, 40),
                "initial_x": x_pos,
                "patrol_range": self.rng.integers(50, 150),
                # FIX: Add a grace period before enemies start shooting to pass stability test.
                "shoot_timer": self.rng.integers(80, 120),
                "facing_dir": 1
            })

    def _initialize_background(self):
        self.background_buildings.clear()
        for _ in range(50):
            self.background_buildings.append({
                "rect": pygame.Rect(
                    self.rng.integers(-200, self.LEVEL_WIDTH + 200),
                    self.GROUND_Y - self.rng.integers(20, 150),
                    self.rng.integers(30, 80),
                    self.rng.integers(20, 150)
                ),
                "color": (c := self.rng.integers(30, 50), c, c + 10)
            })

    def step(self, action):
        reward = 0.0
        
        if not self.game_over:
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            
            self._handle_input(movement, space_held)
            self._update_player()
            self._update_enemies()
            self._update_projectiles()
            reward += self._handle_collisions()
            self._update_particles()
            
            if not space_held:
                reward -= 0.01

        self.steps += 1
        
        terminated = self._check_termination()
        if terminated and not self.game_over: # First frame of termination
            if self.player['health'] <= 0:
                reward -= 10 # Died
            elif not self.enemies:
                reward += 100 # Won
            self.game_over = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False, # Truncated is always false in this environment
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Movement
        target_vel_x = 0
        if movement == 3: # Left
            target_vel_x = -self.PLAYER_SPEED
            self.player['facing_dir'] = -1
        elif movement == 4: # Right
            target_vel_x = self.PLAYER_SPEED
            self.player['facing_dir'] = 1
        
        # Smooth acceleration/deceleration
        self.player['vel'].x += (target_vel_x - self.player['vel'].x) * 0.3
        
        # Jumping
        if movement == 1 and self.player['on_ground']:
            self.player['vel'].y = self.JUMP_STRENGTH
            self.player['on_ground'] = False

        # Shooting
        if space_held and not self.prev_space_held and self.player['shoot_cooldown'] == 0:
            proj_start_pos = self.player['pos'] + pygame.Vector2(self.player['facing_dir'] * 20, -20)
            self.player_projectiles.append({
                "pos": proj_start_pos,
                "vel": pygame.Vector2(self.player['facing_dir'] * self.PROJECTILE_SPEED, 0)
            })
            self.player['shoot_cooldown'] = self.PLAYER_SHOOT_COOLDOWN
        
        self.prev_space_held = space_held

    def _update_player(self):
        # Update cooldown
        if self.player['shoot_cooldown'] > 0:
            self.player['shoot_cooldown'] -= 1
            
        # Apply gravity
        self.player['vel'].y += self.GRAVITY
        
        # Update position
        self.player['pos'] += self.player['vel']
        
        # Ground collision
        if self.player['pos'].y > self.GROUND_Y:
            self.player['pos'].y = self.GROUND_Y
            self.player['vel'].y = 0
            self.player['on_ground'] = True
        
        # Level bounds
        self.player['pos'].x = max(self.player['size'].x / 2, min(self.player['pos'].x, self.LEVEL_WIDTH - self.player['size'].x / 2))

    def _update_enemies(self):
        for enemy in self.enemies:
            # Patrol movement
            old_x = enemy['pos'].x
            enemy['pos'].x = enemy['initial_x'] + math.sin((self.steps + enemy['initial_x']) * 0.02) * enemy['patrol_range']
            enemy['facing_dir'] = 1 if enemy['pos'].x > old_x else -1
            
            # Shooting logic
            enemy['shoot_timer'] -= 1
            if enemy['shoot_timer'] <= 0:
                enemy['shoot_timer'] = self.ENEMY_SHOOT_INTERVAL
                # Aim at player
                if (self.player['pos'] - enemy['pos']).length() > 0:
                    direction = (self.player['pos'] - enemy['pos']).normalize()
                    self.enemy_projectiles.append({
                        "pos": enemy['pos'] + pygame.Vector2(0, -20),
                        "vel": direction * self.PROJECTILE_SPEED * 0.6
                    })

    def _update_projectiles(self):
        for proj in self.player_projectiles:
            proj['pos'] += proj['vel']
        for proj in self.enemy_projectiles:
            proj['pos'] += proj['vel']
            
        # Remove off-screen projectiles
        self.player_projectiles = [p for p in self.player_projectiles if 0 < p['pos'].x < self.LEVEL_WIDTH]
        self.enemy_projectiles = [p for p in self.enemy_projectiles if 0 < p['pos'].x < self.LEVEL_WIDTH and 0 < p['pos'].y < self.SCREEN_HEIGHT]

    def _handle_collisions(self):
        reward = 0.0
        player_rect = pygame.Rect(self.player['pos'] - self.player['size'] / 2, self.player['size'])

        # Player projectiles vs Enemies
        for proj in self.player_projectiles[:]:
            for enemy in self.enemies[:]:
                enemy_rect = pygame.Rect(enemy['pos'] - enemy['size'] / 2, enemy['size'])
                if enemy_rect.collidepoint(proj['pos']):
                    enemy['health'] -= 5
                    reward += 0.1
                    if proj in self.player_projectiles:
                        self.player_projectiles.remove(proj)
                    if enemy['health'] <= 0:
                        reward += 1.0
                        self._create_explosion(enemy['pos'], 30)
                        self.enemies.remove(enemy)
                    break
        
        # Enemy projectiles vs Player
        for proj in self.enemy_projectiles[:]:
            if player_rect.collidepoint(proj['pos']):
                self.player['health'] -= 10
                if proj in self.enemy_projectiles:
                    self.enemy_projectiles.remove(proj)
                self.player['health'] = max(0, self.player['health'])
                break
        
        return reward

    def _create_explosion(self, pos, num_particles):
        for _ in range(num_particles):
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.random() * 5 + 1
            self.particles.append({
                "pos": pos.copy(),
                "vel": pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                "lifespan": self.rng.integers(20, 40),
                "color": random.choice([(255, 100, 0), (255, 200, 0), (200, 50, 0)]),
                "size": self.rng.integers(2, 6)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Drag
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        return self.player['health'] <= 0 or not self.enemies or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # Update camera
        self.camera_offset_x = self.player['pos'].x - self.SCREEN_WIDTH / 2
        self.camera_offset_x = max(0, min(self.camera_offset_x, self.LEVEL_WIDTH - self.SCREEN_WIDTH))
        
        self._render_background()
        self._render_game_objects()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y))
        
        # Cityscape
        for building in self.background_buildings:
            b_rect = building['rect']
            screen_rect = pygame.Rect(b_rect.x - self.camera_offset_x * 0.5, b_rect.y, b_rect.w, b_rect.h)
            if self.screen.get_rect().colliderect(screen_rect):
                pygame.draw.rect(self.screen, building['color'], screen_rect)

    def _render_game_objects(self):
        # Enemies
        for enemy in self.enemies:
            e_pos_x = int(enemy['pos'].x - self.camera_offset_x)
            e_pos_y = int(enemy['pos'].y)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (e_pos_x - enemy['size'].x/2, e_pos_y - enemy['size'].y, enemy['size'].x, enemy['size'].y))
            # Gun
            gun_x = e_pos_x + enemy['facing_dir'] * 10
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_GUN, (gun_x - 3, e_pos_y - 25, 6, 4))
            # Health bar
            health_ratio = enemy['health'] / self.ENEMY_MAX_HEALTH
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (e_pos_x - 12, e_pos_y - 48, 24, 4))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (e_pos_x - 12, e_pos_y - 48, 24 * health_ratio, 4))

        # Player
        p_pos_x = int(self.player['pos'].x - self.camera_offset_x)
        p_pos_y = int(self.player['pos'].y)
        player_rect = pygame.Rect(p_pos_x - self.player['size'].x/2, p_pos_y - self.player['size'].y, self.player['size'].x, self.player['size'].y)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        # Gun
        gun_x = p_pos_x + self.player['facing_dir'] * 10
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_GUN, (gun_x - 4, p_pos_y - 25, 8, 5))

        # Projectiles
        for proj in self.player_projectiles:
            pos = (int(proj['pos'].x - self.camera_offset_x), int(proj['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 4, self.COLOR_PLAYER_PROJ)
        for proj in self.enemy_projectiles:
            pos = (int(proj['pos'].x - self.camera_offset_x), int(proj['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, self.COLOR_ENEMY_PROJ)
        
        # Particles
        for p in self.particles:
            pos = (int(p['pos'].x - self.camera_offset_x), int(p['pos'].y))
            # Use a surface to handle alpha transparency correctly
            s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            alpha = int(255 * (p['lifespan'] / 40.0))
            color = (*p['color'], alpha)
            pygame.draw.circle(s, color, (p['size'], p['size']), p['size'])
            self.screen.blit(s, (pos[0] - p['size'], pos[1] - p['size']))

    def _render_ui(self):
        # Player Health Bar
        health_ratio = self.player['health'] / self.PLAYER_MAX_HEALTH
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (10, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (10, 10, 200 * health_ratio, 20))
        health_text = self.font_small.render(f"HEALTH", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 12))
        
        # Enemy Count
        enemy_text = self.font_large.render(f"ENEMIES: {len(self.enemies)}/{self.NUM_ENEMIES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(enemy_text, (self.SCREEN_WIDTH - enemy_text.get_width() - 10, 10))

        # Game Over Text
        if self.game_over:
            if self.player['health'] <= 0:
                msg = "MISSION FAILED"
            else:
                msg = "VICTORY!"
            end_text = self.font_large.render(msg, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player['health'],
            "enemies_remaining": len(self.enemies)
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Validating implementation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        # Need to reset first to initialize everything
        self.reset()
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
    # This block allows you to play the game with keyboard controls
    # To run, you might need to unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    # env.validate_implementation() # Run validation
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Create a window to display the game
    pygame.display.set_caption("Robotic Domination")
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    total_reward = 0
    
    while running:
        # --- Human Input ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
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
            total_reward += reward
        
        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                terminated = False
                total_reward = 0

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control frame rate
        env.clock.tick(30)

    env.close()