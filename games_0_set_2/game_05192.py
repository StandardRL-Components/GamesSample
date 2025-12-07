
# Generated: 2025-08-28T04:16:14.221970
# Source Brief: brief_05192.md
# Brief Index: 5192

        
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
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold shift to drift and press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade racer. Drift through corners, grab boosts, and use fire at your opponents."
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
        
        # Constants
        self.FPS = 60
        self.MAX_STEPS = 60 * self.FPS

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_WALL = (100, 100, 120)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_ZOMBIE = (220, 50, 50)
        self.COLOR_PROJECTILE = (255, 255, 100)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_HEALTH_BAR = (50, 200, 50)
        self.COLOR_HEALTH_BAR_BG = (70, 70, 70)
        self.COLOR_PARTICLE_ZOMBIE = (255, 200, 0)
        self.COLOR_PARTICLE_PLAYER = (255, 0, 0)

        # Game Parameters
        self.PLAYER_SIZE = 20
        self.PLAYER_SPEED = 4
        self.MAX_HEALTH = 100
        self.SHOOT_COOLDOWN_FRAMES = 10 # 6 shots per second
        self.PROJECTILE_WIDTH = 4
        self.PROJECTILE_HEIGHT = 12
        self.PROJECTILE_SPEED = 10
        self.ZOMBIE_SIZE = 20
        self.ZOMBIE_SPEED = 1.2
        self.ZOMBIE_DAMAGE = 10
        
        # Fonts
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("impact", 60)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [0, 0]
        self.player_health = 0
        self.last_movement_direction = np.array([0, -1])
        self.zombies = []
        self.projectiles = []
        self.particles = []
        self.shoot_cooldown = 0
        self.current_zombie_rate = 0.0
        
        # Initialize state
        self.reset()
        
        # Validate implementation
        try:
            self.validate_implementation()
        except AssertionError as e:
            print(f"Implementation validation failed: {e}")


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = [self.WIDTH / 2, self.HEIGHT / 2]
        self.player_health = self.MAX_HEALTH
        self.last_movement_direction = np.array([0, -1]) # Default aim up

        self.zombies = []
        self.projectiles = []
        self.particles = []

        self.shoot_cooldown = 0
        self.current_zombie_rate = 0.5 # Initial rate: 0.5 zombies/sec
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            # If the game is over, just return the final state
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. ACTION HANDLING ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        move_vec = np.array([0, 0])
        if movement == 1: move_vec[1] = -1 # UP
        elif movement == 2: move_vec[1] = 1 # DOWN
        elif movement == 3: move_vec[0] = -1 # LEFT
        elif movement == 4: move_vec[0] = 1 # RIGHT
        
        if np.any(move_vec):
            self.last_movement_direction = move_vec.astype(float)
        
        self.player_pos[0] += move_vec[0] * self.PLAYER_SPEED
        self.player_pos[1] += move_vec[1] * self.PLAYER_SPEED
        
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE/2, self.WIDTH - self.PLAYER_SIZE/2)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE/2, self.HEIGHT - self.PLAYER_SIZE/2)

        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1

        if space_held and self.shoot_cooldown <= 0:
            self.shoot_cooldown = self.SHOOT_COOLDOWN_FRAMES
            proj_vel = self.last_movement_direction * self.PROJECTILE_SPEED
            angle = math.degrees(math.atan2(-proj_vel[1], proj_vel[0]))
            self.projectiles.append({'pos': list(self.player_pos), 'vel': proj_vel, 'angle': angle})
            # SFX: Laser Shot

        # --- 2. GAME STATE UPDATE ---
        reward = 0.1 # Survival reward per step

        # Update projectiles
        for p in self.projectiles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            if not (0 < p['pos'][0] < self.WIDTH and 0 < p['pos'][1] < self.HEIGHT):
                self.projectiles.remove(p)

        # Update zombies and check for player collision
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_SIZE/2, self.player_pos[1] - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        for z in self.zombies[:]:
            direction = np.array(self.player_pos) - np.array(z['pos'])
            norm = np.linalg.norm(direction)
            if norm > 1: # Avoid division by zero if zombie is on player
                direction /= norm
            
            z['pos'] += direction * self.ZOMBIE_SPEED
            
            zombie_rect = pygame.Rect(z['pos'][0] - self.ZOMBIE_SIZE/2, z['pos'][1] - self.ZOMBIE_SIZE/2, self.ZOMBIE_SIZE, self.ZOMBIE_SIZE)
            if player_rect.colliderect(zombie_rect):
                self.player_health -= self.ZOMBIE_DAMAGE
                self._create_particles(self.player_pos, self.COLOR_PARTICLE_PLAYER, 20)
                self.zombies.remove(z)
                # SFX: Player Hurt

        # Update particles
        for particle in self.particles[:]:
            particle['pos'][0] += particle['vel'][0]
            particle['pos'][1] += particle['vel'][1]
            particle['life'] -= 1
            if particle['life'] <= 0:
                self.particles.remove(particle)

        # --- 3. COLLISION DETECTION (PROJECTILE-ZOMBIE) ---
        zombies_to_remove = set()
        projectiles_to_remove = set()

        for i, p in enumerate(self.projectiles):
            # Create a rotated rect for collision
            rotated_surf = pygame.Surface((self.PROJECTILE_WIDTH, self.PROJECTILE_HEIGHT), pygame.SRCALPHA)
            rotated_surf = pygame.transform.rotate(rotated_surf, p['angle'])
            proj_rect = rotated_surf.get_rect(center=p['pos'])

            for j, z in enumerate(self.zombies):
                if j in zombies_to_remove: continue
                
                zombie_rect = pygame.Rect(z['pos'][0] - self.ZOMBIE_SIZE/2, z['pos'][1] - self.ZOMBIE_SIZE/2, self.ZOMBIE_SIZE, self.ZOMBIE_SIZE)
                if proj_rect.colliderect(zombie_rect):
                    zombies_to_remove.add(j)
                    projectiles_to_remove.add(i)
                    self.score += 1
                    reward += 1
                    self._create_particles(z['pos'], self.COLOR_PARTICLE_ZOMBIE, 30)
                    # SFX: Zombie Hit/Explode
                    break

        if zombies_to_remove:
            self.zombies = [z for i, z in enumerate(self.zombies) if i not in zombies_to_remove]
        if projectiles_to_remove:
            self.projectiles = [p for i, p in enumerate(self.projectiles) if i not in projectiles_to_remove]

        # --- 4. ZOMBIE SPAWNING ---
        seconds_elapsed = self.steps / self.FPS
        base_rate = 0.5 + math.floor(seconds_elapsed / 10)
        self.current_zombie_rate = min(5.0, base_rate)
        
        if self.np_random.random() < self.current_zombie_rate / self.FPS:
            self._spawn_zombie()

        # --- 5. TERMINATION CHECK ---
        self.steps += 1
        terminated = False
        
        if self.player_health <= 0:
            self.player_health = 0
            reward = -100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            reward = 100
            terminated = True
            self.game_over = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Arena Walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WIDTH, self.HEIGHT), 5)

        # Particles
        for p in self.particles:
            size = p['size'] * (p['life'] / p['max_life'])
            if size > 0:
                pygame.draw.rect(self.screen, p['color'], (p['pos'][0] - size/2, p['pos'][1] - size/2, size, size))

        # Zombies
        for z in self.zombies:
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, (int(z['pos'][0] - self.ZOMBIE_SIZE/2), int(z['pos'][1] - self.ZOMBIE_SIZE/2), self.ZOMBIE_SIZE, self.ZOMBIE_SIZE))

        # Projectiles
        for p in self.projectiles:
            rotated_surf = pygame.Surface((self.PROJECTILE_WIDTH, self.PROJECTILE_HEIGHT), pygame.SRCALPHA)
            rotated_surf.fill(self.COLOR_PROJECTILE)
            rotated_surf = pygame.transform.rotate(rotated_surf, p['angle'])
            rect = rotated_surf.get_rect(center=(int(p['pos'][0]), int(p['pos'][1])))
            self.screen.blit(rotated_surf, rect)

        # Player
        player_x, player_y = int(self.player_pos[0]), int(self.player_pos[1])
        glow_size = int(self.PLAYER_SIZE * 1.5)
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER, 50), (glow_size//2, glow_size//2), glow_size//2)
        self.screen.blit(glow_surf, (player_x - glow_size//2, player_y - glow_size//2))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (player_x - self.PLAYER_SIZE//2, player_y - self.PLAYER_SIZE//2, self.PLAYER_SIZE, self.PLAYER_SIZE))
        pygame.draw.rect(self.screen, (255,255,255), (player_x - self.PLAYER_SIZE//2, player_y - self.PLAYER_SIZE//2, self.PLAYER_SIZE, self.PLAYER_SIZE), 1)

    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.player_health / self.MAX_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(200 * health_ratio), 20))
        
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(center=(self.WIDTH / 2, 22))
        self.screen.blit(score_text, score_rect)

        # Timer
        time_elapsed = min(60.0, self.steps / self.FPS)
        timer_text = self.font_ui.render(f"TIME: {time_elapsed:.1f}", True, self.COLOR_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.WIDTH - 10, 12))
        self.screen.blit(timer_text, timer_rect)

        # Game Over Message
        if self.game_over:
            win = self.player_health > 0 and self.steps >= self.MAX_STEPS
            message = "YOU SURVIVED!" if win else "GAME OVER"
            color = (0, 255, 128) if win else (255, 50, 50)
            
            end_text = self.font_game_over.render(message, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            
            # Text shadow
            shadow_text = self.font_game_over.render(message, True, (0,0,0))
            shadow_rect = shadow_text.get_rect(center=(self.WIDTH / 2 + 3, self.HEIGHT / 2 + 3))
            self.screen.blit(shadow_text, shadow_rect)
            self.screen.blit(end_text, end_rect)

    def _spawn_zombie(self):
        edge = self.np_random.integers(4)
        pos = [0.0, 0.0]
        if edge == 0: # Top
            pos = [self.np_random.uniform(0, self.WIDTH), -self.ZOMBIE_SIZE]
        elif edge == 1: # Bottom
            pos = [self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.ZOMBIE_SIZE]
        elif edge == 2: # Left
            pos = [-self.ZOMBIE_SIZE, self.np_random.uniform(0, self.HEIGHT)]
        else: # Right
            pos = [self.WIDTH + self.ZOMBIE_SIZE, self.np_random.uniform(0, self.HEIGHT)]
        self.zombies.append({'pos': np.array(pos, dtype=float)})

    def _create_particles(self, position, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': list(position),
                'vel': vel,
                'life': life,
                'max_life': life,
                'color': color,
                'size': self.np_random.integers(3, 7)
            })
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health
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