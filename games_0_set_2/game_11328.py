import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:48:27.477712
# Source Brief: brief_01328.md
# Brief Index: 1328
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player pilots a ship upward through a barrage of
    alien projectiles. The goal is to reach the top of the screen without being hit.
    Visual quality and game feel are prioritized.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = "Pilot your ship through a dangerous barrage of alien projectiles. Evade enemy fire and race to the top of the screen to win."
    user_guide = "Controls: Use ↑↓←→ arrow keys to move your ship. Press space to fire."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Critical Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        
        # --- Visual & Game Feel Constants ---
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_PLAYER_GLOW = (0, 255, 150, 50)
        self.COLOR_PLAYER_PROJ = (100, 200, 255)
        self.COLOR_PLAYER_PROJ_GLOW = (100, 200, 255, 100)
        self.COLOR_ALIEN_PROJ = (255, 80, 80)
        self.COLOR_ALIEN_PROJ_GLOW = (255, 80, 80, 100)
        self.COLOR_STAR = (220, 220, 240)
        self.COLOR_THRUSTER = (255, 180, 50)
        self.COLOR_UI = (255, 255, 255)

        self.PLAYER_SPEED = 6
        self.PLAYER_PROJECTILE_SPEED = 10
        self.ALIEN_PROJECTILE_SPEED = 4
        self.PLAYER_FIRE_COOLDOWN_MAX = 5  # Can fire every 5 steps
        self.MAX_EPISODE_STEPS = 1000

        # --- Initialize state variables (will be properly set in reset) ---
        self.player_pos = None
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []
        self.stars = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_fire_cooldown = 0
        self.alien_fire_timer = 0
        self.alien_fire_interval = 60 # Initially, fire every 2 seconds (60 frames at 30fps)
        
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Reset Game State ---
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT - 40], dtype=np.float32)
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_fire_cooldown = 0
        self.alien_fire_timer = 0
        self.alien_fire_interval = 60 # Reset to initial fire rate
        
        # Generate a new starfield for visual variety
        self.stars = [
            [self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.uniform(0.5, 1.5), self.np_random.uniform(0.5, 1.5)]
            for _ in range(150)
        ]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing but return the final state
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        
        # --- Handle Actions ---
        self._handle_input(action)

        # --- Update Game Logic ---
        self._update_projectiles()
        self._update_difficulty_and_spawn_aliens()
        self._update_particles()
        
        # --- Check for Terminal Conditions & Calculate Reward ---
        won = self.player_pos[1] <= 0
        collision = self._check_collisions()
        
        terminated = won or collision or self.steps >= self.MAX_EPISODE_STEPS
        truncated = False
        
        if won:
            reward = 100.0
            self.score += 100
        elif collision:
            reward = -100.0
            self.score -= 100
            self._create_explosion(self.player_pos)
        else:
            # Reward for surviving
            reward = 0.1
            self.score += 0.1

        if terminated:
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # --- Movement ---
        move_vector = np.array([0, 0], dtype=np.float32)
        if movement == 1:  # Up
            move_vector[1] -= self.PLAYER_SPEED
        elif movement == 2:  # Down
            move_vector[1] += self.PLAYER_SPEED
        elif movement == 3:  # Left
            move_vector[0] -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            move_vector[0] += self.PLAYER_SPEED
        
        if np.any(move_vector):
            self.player_pos += move_vector
            self._create_thruster_particles(self.player_pos, move_vector)

        # Clamp player position to screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WIDTH)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.HEIGHT)
        
        # --- Firing ---
        if self.player_fire_cooldown > 0:
            self.player_fire_cooldown -= 1

        if space_held and self.player_fire_cooldown == 0:
            # Fire a projectile from the ship's nose
            proj_pos = [self.player_pos[0], self.player_pos[1] - 15]
            self.player_projectiles.append(proj_pos)
            self.player_fire_cooldown = self.PLAYER_FIRE_COOLDOWN_MAX
            # sfx: player_shoot.wav

    def _update_projectiles(self):
        # Move player projectiles up
        for proj in self.player_projectiles:
            proj[1] -= self.PLAYER_PROJECTILE_SPEED
        self.player_projectiles = [p for p in self.player_projectiles if p[1] > 0]

        # Move alien projectiles up
        # This was a bug: alien projectiles should move down the screen (increasing y)
        for proj in self.alien_projectiles:
            proj[1] += self.ALIEN_PROJECTILE_SPEED
        self.alien_projectiles = [p for p in self.alien_projectiles if p[1] < self.HEIGHT]

    def _update_difficulty_and_spawn_aliens(self):
        # Difficulty scales by reducing the firing interval over time
        self.alien_fire_interval = max(10, 60 - 0.05 * self.steps)
        self.alien_fire_timer += 1
        
        if self.alien_fire_timer >= self.alien_fire_interval:
            self.alien_fire_timer = 0
            # Spawn a new alien projectile at the top of the screen
            spawn_x = self.np_random.integers(20, self.WIDTH - 20)
            self.alien_projectiles.append([spawn_x, 0])
            # sfx: alien_spawn.wav

    def _check_collisions(self):
        player_rect = pygame.Rect(self.player_pos[0] - 7, self.player_pos[1] - 10, 14, 20)
        for proj in self.alien_projectiles:
            proj_rect = pygame.Rect(proj[0] - 4, proj[1] - 4, 8, 8)
            if player_rect.colliderect(proj_rect):
                # sfx: player_explosion.wav
                return True
        return False

    def _create_thruster_particles(self, pos, move_vector):
        # Create particles opposite to the direction of movement
        for _ in range(3):
            particle_pos = pos + np.array([0, 10]) # Emitter at the back of the ship
            particle_vel = -move_vector * self.np_random.uniform(0.2, 0.5) + self.np_random.standard_normal(2) * 0.5
            lifespan = self.np_random.integers(10, 21)
            self.particles.append({'pos': particle_pos, 'vel': particle_vel, 'life': lifespan, 'max_life': lifespan, 'color': self.COLOR_THRUSTER})

    def _create_explosion(self, pos):
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            lifespan = self.np_random.integers(20, 41)
            color = random.choice([self.COLOR_ALIEN_PROJ, self.COLOR_THRUSTER, (255, 255, 255)])
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': lifespan, 'max_life': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_stars()
        self._render_projectiles()
        if not self.game_over:
            self._render_player()
        self._render_particles()

    def _render_stars(self):
        for star in self.stars:
            x, y, size, brightness_mod = star
            brightness = 150 + 100 * math.sin(pygame.time.get_ticks() * 0.001 * brightness_mod)
            color = (min(255, self.COLOR_STAR[0] * brightness / 255), 
                     min(255, self.COLOR_STAR[1] * brightness / 255), 
                     min(255, self.COLOR_STAR[2] * brightness / 255))
            pygame.draw.circle(self.screen, color, (int(x), int(y)), int(size))

    def _render_player(self):
        x, y = int(self.player_pos[0]), int(self.player_pos[1])
        points = [(x, y - 12), (x - 8, y + 8), (x + 8, y + 8)]
        
        # Glow effect
        glow_points = [(x, y - 16), (x - 12, y + 12), (x + 12, y + 12)]
        pygame.gfxdraw.aapolygon(self.screen, glow_points, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_polygon(self.screen, glow_points, self.COLOR_PLAYER_GLOW)

        # Main ship
        pygame.draw.polygon(self.screen, self.COLOR_PLAYER, points)
        
    def _render_projectiles(self):
        # Player projectiles
        for p_pos in self.player_projectiles:
            x, y = int(p_pos[0]), int(p_pos[1])
            # Glow
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJ_GLOW, (x - 3, y - 10, 6, 20))
            # Core
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJ, (x - 1, y - 8, 2, 16))

        # Alien projectiles
        for a_pos in self.alien_projectiles:
            x, y = int(a_pos[0]), int(a_pos[1])
            # Glow
            pygame.gfxdraw.aacircle(self.screen, x, y, 8, self.COLOR_ALIEN_PROJ_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, x, y, 8, self.COLOR_ALIEN_PROJ_GLOW)
            # Core
            pygame.draw.circle(self.screen, self.COLOR_ALIEN_PROJ, (x, y), 5)

    def _render_particles(self):
        for p in self.particles:
            x, y = int(p['pos'][0]), int(p['pos'][1])
            life_ratio = p['life'] / p['max_life']
            size = int(5 * life_ratio)
            if size > 0:
                # Fade color with life
                color = tuple(int(c * life_ratio) for c in p['color'])
                pygame.draw.circle(self.screen, color, (x, y), size)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))
        
        progress = (self.HEIGHT - self.player_pos[1]) / self.HEIGHT
        progress_width = int(progress * self.WIDTH)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (0, self.HEIGHT - 5, progress_width, 5))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_GLOW, (0, self.HEIGHT - 5, progress_width, 5))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos.tolist(),
            "alien_projectiles": len(self.alien_projectiles)
        }

    def close(self):
        pygame.quit()

    def render(self):
        return self._get_observation()

if __name__ == "__main__":
    # Example of how to run the environment
    env = GameEnv()
    
    # --- Manual Play ---
    # Use arrow keys for movement, space to shoot.
    # The window will be controlled by Pygame directly.
    
    obs, info = env.reset()
    
    # Create a display window
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.display.init()
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Alien Barrage")
    
    running = True
    while running:
        # Map keyboard inputs to action space
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.metadata["render_fps"])

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000) # Pause for 2 seconds before reset
            obs, info = env.reset()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    env.close()