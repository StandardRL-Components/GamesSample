
# Generated: 2025-08-28T04:53:44.152230
# Source Brief: brief_05402.md
# Brief Index: 5402

        
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
    """
    A minimalist, side-scrolling survival game where the player dodges waves of zombies.

    The player controls a white rectangle and must move up and down to avoid red
    zombie rectangles that fly across the screen from right to left. The game's
    difficulty increases over time as zombies become faster and more numerous.
    The goal is to survive for 60 seconds.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ↑ and ↓ to move your character. Avoid the red zombies."
    )

    game_description = (
        "Survive for 60 seconds against an onslaught of procedurally generated zombies in this minimalist arcade game."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 60 * FPS  # 60 seconds

    # Colors
    COLOR_BG = (10, 10, 15)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (200, 200, 255, 60)
    COLOR_ZOMBIE = (255, 60, 60)
    COLOR_ZOMBIE_GLOW = (200, 20, 20, 80)
    COLOR_TEXT = (240, 240, 240)
    COLOR_HEART = (220, 20, 60)
    COLOR_HIT_PARTICLE = (255, 150, 150)

    # Player settings
    PLAYER_WIDTH, PLAYER_HEIGHT = 15, 30
    PLAYER_START_X = 60
    PLAYER_SPEED = 8

    # Game settings
    INITIAL_LIVES = 3
    DIFFICULTY_INTERVAL = 10 * FPS  # every 10 seconds

    # Difficulty progression
    INITIAL_ZOMBIE_SPEED = 3.0
    ZOMBIE_SPEED_INCREMENT = 0.75
    MAX_ZOMBIE_SPEED = 12.0
    INITIAL_ZOMBIE_SPAWN_RATE = 0.04
    ZOMBIE_SPAWN_RATE_INCREMENT = 0.015
    MAX_ZOMBIE_SPAWN_RATE = 0.25

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Initialize state variables
        self.player_rect = pygame.Rect(0, 0, 0, 0)
        self.zombies = []
        self.particles = []
        self.lives = 0
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.zombie_speed = 0.0
        self.zombie_spawn_rate = 0.0
        self.last_difficulty_update = 0
        self.screen_flash_timer = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_rect = pygame.Rect(
            self.PLAYER_START_X,
            self.HEIGHT // 2 - self.PLAYER_HEIGHT // 2,
            self.PLAYER_WIDTH,
            self.PLAYER_HEIGHT
        )
        
        self.zombies = []
        self.particles = []
        
        self.lives = self.INITIAL_LIVES
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self.zombie_speed = self.INITIAL_ZOMBIE_SPEED
        self.zombie_spawn_rate = self.INITIAL_ZOMBIE_SPAWN_RATE
        self.last_difficulty_update = 0
        self.screen_flash_timer = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing but return the final state
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        
        movement, _, _ = action
        self._handle_player_movement(movement)
        
        self._update_zombies()
        self._update_particles()
        
        # Check for collisions
        collisions = self._check_collisions()
        if collisions > 0:
            self.lives -= collisions
            reward -= 5.0 * collisions  # -5 for each hit
            self.screen_flash_timer = 5 # Flash screen for 5 frames
            # sfx: player_hit.wav

        self._update_difficulty()

        self.steps += 1
        reward += 0.1  # +0.1 for each step survived

        terminated = self.lives <= 0 or self.steps >= self.MAX_STEPS

        if terminated:
            self.game_over = True
            if self.lives > 0:
                reward += 50.0  # +50 for surviving
                # sfx: victory.wav
            else:
                # sfx: game_over.wav
                pass

        self.score += reward
        if self.auto_advance:
            self.clock.tick(self.FPS)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_movement(self, movement):
        if movement == 1:  # Up
            self.player_rect.y -= self.PLAYER_SPEED
        elif movement == 2:  # Down
            self.player_rect.y += self.PLAYER_SPEED
        
        # Clamp player position to screen bounds
        self.player_rect.y = max(0, min(self.player_rect.y, self.HEIGHT - self.PLAYER_HEIGHT))

    def _update_zombies(self):
        # Move existing zombies
        for zombie in self.zombies[:]:
            zombie['rect'].x -= zombie['speed']
            if zombie['rect'].right < 0:
                self.zombies.remove(zombie)
        
        # Spawn new zombies
        if self.np_random.random() < self.zombie_spawn_rate:
            zombie_height = self.np_random.integers(20, 60)
            zombie_y = self.np_random.integers(0, self.HEIGHT - zombie_height)
            zombie_rect = pygame.Rect(self.WIDTH, zombie_y, self.np_random.integers(15, 30), zombie_height)
            self.zombies.append({'rect': zombie_rect, 'speed': self.zombie_speed * self.np_random.uniform(0.8, 1.2)})
            # sfx: zombie_spawn.wav

    def _check_collisions(self):
        collisions = 0
        collided_zombies = []
        for zombie in self.zombies:
            if self.player_rect.colliderect(zombie['rect']):
                collisions += 1
                collided_zombies.append(zombie)
                self._create_hit_particles(self.player_rect.center)
        
        self.zombies = [z for z in self.zombies if z not in collided_zombies]
        return collisions

    def _update_difficulty(self):
        if self.steps - self.last_difficulty_update > self.DIFFICULTY_INTERVAL:
            self.last_difficulty_update = self.steps
            self.zombie_speed = min(self.MAX_ZOMBIE_SPEED, self.zombie_speed + self.ZOMBIE_SPEED_INCREMENT)
            self.zombie_spawn_rate = min(self.MAX_ZOMBIE_SPAWN_RATE, self.zombie_spawn_rate + self.ZOMBIE_SPAWN_RATE_INCREMENT)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render zombies
        for zombie in self.zombies:
            glow_rect = zombie['rect'].inflate(10, 10)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.ellipse(s, self.COLOR_ZOMBIE_GLOW, s.get_rect())
            self.screen.blit(s, glow_rect.topleft)
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, zombie['rect'])

        # Render player
        glow_rect = self.player_rect.inflate(12, 12)
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.ellipse(s, self.COLOR_PLAYER_GLOW, s.get_rect())
        self.screen.blit(s, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect)

        # Render particles
        self._render_particles()

        # Render screen flash on hit
        if self.screen_flash_timer > 0:
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            flash_alpha = 100 * (self.screen_flash_timer / 5)
            flash_surface.fill((255, 255, 255, flash_alpha))
            self.screen.blit(flash_surface, (0, 0))
            self.screen_flash_timer -= 1
            
    def _render_ui(self):
        # Display time survived
        time_text = f"TIME: {self.steps / self.FPS:.1f}"
        time_surf = self.font_small.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (15, 15))

        # Display lives as hearts
        for i in range(self.lives):
            self._draw_heart(self.screen, self.WIDTH - 30 - i * 35, 30, 12)
        
        # Display game over/victory message
        if self.game_over:
            if self.lives > 0:
                message = "VICTORY!"
            else:
                message = "GAME OVER"
            
            text_surf = self.font_large.render(message, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            
            # Add a semi-transparent background for readability
            bg_rect = text_rect.inflate(20, 20)
            bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surf.fill((0, 0, 0, 150))
            self.screen.blit(bg_surf, bg_rect)
            self.screen.blit(text_surf, text_rect)

    def _draw_heart(self, surface, x, y, size):
        # Simple heart shape using two circles and a polygon
        pygame.gfxdraw.filled_circle(surface, x - size // 2, y - size // 4, size // 2, self.COLOR_HEART)
        pygame.gfxdraw.aacircle(surface, x - size // 2, y - size // 4, size // 2, self.COLOR_HEART)
        pygame.gfxdraw.filled_circle(surface, x + size // 2, y - size // 4, size // 2, self.COLOR_HEART)
        pygame.gfxdraw.aacircle(surface, x + size // 2, y - size // 4, size // 2, self.COLOR_HEART)
        points = [
            (x - size, y - size // 4),
            (x + size, y - size // 4),
            (x, y + size * 0.75)
        ]
        pygame.gfxdraw.filled_polygon(surface, points, self.COLOR_HEART)
        pygame.gfxdraw.aapolygon(surface, points, self.COLOR_HEART)

    def _create_hit_particles(self, position):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            radius = self.np_random.uniform(2, 5)
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': list(position),
                'vel': vel,
                'radius': radius,
                'life': life,
                'max_life': life
            })
    
    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            radius = int(p['radius'] * life_ratio)
            if radius > 0:
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                alpha = int(200 * life_ratio)
                color = self.COLOR_HIT_PARTICLE + (alpha,)
                
                s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (radius, radius), radius)
                self.screen.blit(s, (pos[0]-radius, pos[1]-radius))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "zombie_speed": self.zombie_speed,
            "zombie_spawn_rate": self.zombie_spawn_rate
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption(env.game_description)
    game_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    action = env.action_space.sample()
    action[0] = 0 # Start with no-op

    print(env.user_guide)

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        
        # Reset action
        action[0] = 0 # Movement: 0=none
        
        # Update action based on key presses
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        
        # The brief doesn't use these, but we map them anyway
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the game screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        game_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Time Survived: {info['steps']/env.FPS:.1f}s")
            pygame.time.wait(3000) # Wait 3 seconds before closing

    env.close()