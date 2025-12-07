
# Generated: 2025-08-28T05:09:28.984286
# Source Brief: brief_05476.md
# Brief Index: 5476

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ← and → to choose jump direction. ↑ jumps straight up. Hold SPACE for a higher jump."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Hop between rising platforms, collecting stars to score points. The platforms get faster over time. Don't fall off the bottom!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.TARGET_SCORE = 20
        self.MAX_STEPS = 1500

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # Colors
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (0, 0, 10)
        self.COLOR_PLAYER = (255, 60, 60)
        self.COLOR_PLAYER_GLOW = (255, 120, 120)
        self.COLOR_PLATFORM = (40, 220, 120)
        self.COLOR_PLATFORM_OUTLINE = (20, 180, 90)
        self.COLOR_STAR = (255, 220, 50)
        self.COLOR_STAR_GLOW = (255, 240, 150)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_PARTICLE_JUMP = (200, 200, 255)
        self.COLOR_PARTICLE_STAR = self.COLOR_STAR

        # Physics
        self.GRAVITY = 0.35
        self.JUMP_SMALL = 8.5
        self.JUMP_LARGE = 11.0
        self.JUMP_HORIZONTAL = 5.0
        self.PLAYER_FRICTION = -0.1

        # Player properties
        self.player_size = 12

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.platforms = None
        self.stars = None
        self.particles = None
        self.on_platform = False
        self.platform_speed = 0.0
        self.last_platform_y = 0.0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        
        self.platforms = []
        self.stars = []
        self.particles = []
        
        self.platform_speed = 1.0
        self.last_platform_y = self.HEIGHT + 20

        # Create initial platforms
        start_platform = pygame.Rect(self.WIDTH/2 - 50, self.HEIGHT/2 + 20, 100, 20)
        self.platforms.append(start_platform)
        for y in range(int(self.HEIGHT/2 + 20) + 60, self.HEIGHT + 100, 80):
             self._generate_platform(y)
        
        self.on_platform = True
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.01  # Small penalty for every frame to encourage speed
        self.steps += 1

        # --- Handle Input ---
        self._handle_input(action)

        # --- Update Game Logic ---
        self._update_player()
        landed, landing_reward = self._handle_platform_collisions()
        reward += landing_reward

        collection_reward = self._handle_star_collisions()
        reward += collection_reward

        self._update_world()
        self._update_particles()

        # --- Check Termination ---
        terminated = False
        if self.player_pos.y > self.HEIGHT + self.player_size:
            terminated = True
            reward -= 10  # Penalty for falling
            self.game_over = True
        elif self.score >= self.TARGET_SCORE:
            terminated = True
            reward += 100  # Large reward for winning
            self.game_over = True
            self.win = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean for jump power
        
        if self.on_platform and movement in [1, 3, 4]: # Can't jump down (2) or no-op (0)
            self.on_platform = False
            jump_power = self.JUMP_LARGE if space_held else self.JUMP_SMALL
            self.player_vel.y = -jump_power
            # Sound: Jump

            if movement == 1: # Up
                self.player_vel.x = 0
            elif movement == 3: # Left
                self.player_vel.x = -self.JUMP_HORIZONTAL
            elif movement == 4: # Right
                self.player_vel.x = self.JUMP_HORIZONTAL
            
            self._spawn_particles(
                pygame.Vector2(self.player_pos.x, self.player_pos.y), 
                self.COLOR_PARTICLE_JUMP, 15, -1
            )

    def _update_player(self):
        if not self.on_platform:
            self.player_vel.y += self.GRAVITY
        
        # Apply friction
        if abs(self.player_vel.x) > 0:
            friction = self.PLAYER_FRICTION * np.sign(self.player_vel.x)
            if abs(self.player_vel.x) > abs(friction):
                 self.player_vel.x += friction
            else:
                 self.player_vel.x = 0

        self.player_pos += self.player_vel

        # Screen bounds (left/right)
        if self.player_pos.x < self.player_size / 2:
            self.player_pos.x = self.player_size / 2
            self.player_vel.x *= -0.5
        elif self.player_pos.x > self.WIDTH - self.player_size / 2:
            self.player_pos.x = self.WIDTH - self.player_size / 2
            self.player_vel.x *= -0.5

    def _handle_platform_collisions(self):
        landed = False
        reward = 0
        player_rect = pygame.Rect(
            self.player_pos.x - self.player_size / 2, 
            self.player_pos.y - self.player_size, 
            self.player_size, 
            self.player_size
        )

        if self.player_vel.y > 0: # Only check for landing if falling
            for plat in self.platforms:
                if player_rect.colliderect(plat) and (player_rect.bottom - self.player_vel.y) <= plat.top:
                    self.player_pos.y = plat.top
                    self.player_vel.y = 0
                    self.player_vel.x = 0 # Stop horizontal movement on landing
                    self.on_platform = True
                    landed = True
                    reward += 0.1 # Reward for landing
                    # Sound: Land
                    break
        return landed, reward

    def _handle_star_collisions(self):
        reward = 0
        player_rect = pygame.Rect(
            self.player_pos.x - self.player_size, 
            self.player_pos.y - self.player_size * 1.5, 
            self.player_size * 2, 
            self.player_size * 2
        )
        for star in self.stars[:]:
            star_pos, star_platform = star
            star_rect = pygame.Rect(star_pos.x - 8, star_pos.y - 8, 16, 16)
            if player_rect.colliderect(star_rect):
                self.stars.remove(star)
                self.score += 1
                reward += 1.0 # Reward for collecting a star
                # Sound: Collect Star
                self._spawn_particles(star_pos, self.COLOR_PARTICLE_STAR, 20, 1)
                break
        return reward

    def _update_world(self):
        # Move platforms and stars up
        scroll_speed = self.platform_speed
        self.player_pos.y += scroll_speed
        for plat in self.platforms:
            plat.y += scroll_speed
        for star in self.stars:
            star[0].y += scroll_speed

        # Remove off-screen elements
        self.platforms = [p for p in self.platforms if p.bottom > 0]
        self.stars = [s for s in self.stars if s[0].y > -20]
        
        # Add new platforms
        if self.platforms:
            self.last_platform_y = min(p.y for p in self.platforms)
        
        while self.last_platform_y > -20:
            new_y = self.last_platform_y - self.np_random.integers(60, 100)
            self._generate_platform(new_y)
            self.last_platform_y = new_y

        # Increase difficulty
        if self.steps > 0 and self.steps % 100 == 0:
            self.platform_speed += 0.01

    def _generate_platform(self, y_pos):
        last_x = self.platforms[-1].centerx if self.platforms else self.WIDTH / 2
        width = self.np_random.integers(60, 150)
        max_offset = self.WIDTH/2 - width/2 - 20
        offset = self.np_random.integers(-140, 141)
        
        x_pos = last_x + offset
        x_pos = np.clip(x_pos, width / 2, self.WIDTH - width / 2)

        new_platform = pygame.Rect(x_pos - width / 2, y_pos, width, 20)
        self.platforms.append(new_platform)
        
        # Chance to spawn a star
        if self.np_random.random() < 0.35:
            star_pos = pygame.Vector2(new_platform.centerx, new_platform.top - 15)
            self.stars.append([star_pos, new_platform])

    def _spawn_particles(self, pos, color, count, direction):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed * direction)
            lifetime = self.np_random.integers(15, 30)
            self.particles.append([pygame.Vector2(pos), vel, lifetime, color])

    def _update_particles(self):
        for p in self.particles[:]:
            p[0] += p[1] # pos += vel
            p[2] -= 1    # lifetime -= 1
            if p[2] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        # --- Render all game elements ---
        self._render_background()
        self._render_particles()
        self._render_platforms_and_stars()
        self._render_player()
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_background(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_platforms_and_stars(self):
        for plat in self.platforms:
            pygame.gfxdraw.box(self.screen, plat, self.COLOR_PLATFORM)
            pygame.gfxdraw.rectangle(self.screen, plat, self.COLOR_PLATFORM_OUTLINE)

        for star_pos, _ in self.stars:
            x, y = int(star_pos.x), int(star_pos.y)
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, x, y, 10, self.COLOR_STAR_GLOW + (80,))
            pygame.gfxdraw.aacircle(self.screen, x, y, 10, self.COLOR_STAR_GLOW + (80,))
            # Star body
            pygame.gfxdraw.filled_circle(self.screen, x, y, 7, self.COLOR_STAR)
            pygame.gfxdraw.aacircle(self.screen, x, y, 7, self.COLOR_STAR)

    def _render_player(self):
        x, y = self.player_pos.x, self.player_pos.y
        s = self.player_size
        
        # Points of the triangle
        p1 = (int(x), int(y - s))
        p2 = (int(x - s / 1.5), int(y + s / 2))
        p3 = (int(x + s / 1.5), int(y + s / 2))
        
        # Glow effect
        glow_s = s * 1.8
        gp1 = (int(x), int(y - glow_s))
        gp2 = (int(x - glow_s / 1.5), int(y + glow_s / 2))
        gp3 = (int(x + glow_s / 1.5), int(y + glow_s / 2))
        pygame.gfxdraw.filled_trigon(self.screen, gp1[0], gp1[1], gp2[0], gp2[1], gp3[0], gp3[1], self.COLOR_PLAYER_GLOW + (100,))
        pygame.gfxdraw.aatrigon(self.screen, gp1[0], gp1[1], gp2[0], gp2[1], gp3[0], gp3[1], self.COLOR_PLAYER_GLOW + (100,))
        
        # Player body
        pygame.gfxdraw.filled_trigon(self.screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], self.COLOR_PLAYER)
        pygame.gfxdraw.aatrigon(self.screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], self.COLOR_PLAYER)

    def _render_particles(self):
        for pos, vel, lifetime, color in self.particles:
            size = max(1, int(lifetime / 5))
            alpha = max(0, min(255, int(lifetime * 15)))
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), size, color + (alpha,))

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score} / {self.TARGET_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        steps_text = self.small_font.render(f"STEPS: {self.steps} / {self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "YOU WIN!" if self.win else "GAME OVER"
            end_text = self.font.render(end_text_str, True, self.COLOR_STAR if self.win else self.COLOR_PLAYER)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "platform_speed": self.platform_speed,
            "player_y_vel": self.player_vel.y,
        }

    def close(self):
        pygame.quit()
        super().close()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- To play manually ---
    # This requires a window and is not part of the core headless environment.
    # It's for demonstration and debugging.
    try:
        import os
        os.environ["SDL_VIDEODRIVER"] = "dummy" # Ensure headless for gym
        pygame.display.init()
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption(env.game_description)
        clock = pygame.time.Clock()

        obs, info = env.reset()
        terminated = False
        
        while not terminated:
            # Action mapping for human play
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 0 # Unused in this game
            
            action = [movement, space_held, shift_held]

            obs, reward, terminated, truncated, info = env.step(action)
            
            # Draw the observation from the environment to the screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()

            clock.tick(30) # Match the auto_advance rate
            
    finally:
        env.close()
        pygame.quit()