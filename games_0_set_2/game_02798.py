
# Generated: 2025-08-27T21:28:17.713204
# Source Brief: brief_02798.md
# Brief Index: 2798

        
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
        "Controls: Use arrow keys to jump. Hold Space for a power jump. Hold Shift to dash in the direction you are pressing."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Hop your spaceship up a field of platforms, dodging deadly obstacles. The higher you get, the faster it scrolls!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.WORLD_HEIGHT = 10000  # The total height of the level to reach the top

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
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        self.game_over_font = pygame.font.SysFont("monospace", 50, bold=True)

        # Colors
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (100, 200, 255)
        self.COLOR_PLATFORM = (0, 200, 100)
        self.COLOR_PLATFORM_OUTLINE = (0, 150, 75)
        self.COLOR_OBSTACLE = (255, 50, 50)
        self.COLOR_OBSTACLE_GLOW = (255, 150, 150)
        self.COLOR_TEXT = (255, 255, 255)

        # Physics and game parameters
        self.GRAVITY = 0.35
        self.MAX_FALL_SPEED = 8
        self.JUMP_BASE_POWER = -8.0
        self.JUMP_SIDE_POWER = 4.0
        self.POWER_JUMP_MOD = 1.4
        self.DASH_MOD = 1.8
        self.AIR_CONTROL = 0.3
        self.MAX_STEPS = 5000
        
        # Initialize state variables
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_on_platform = False
        self.player_rect = None
        self.platform_player_is_on = None

        self.platforms = []
        self.obstacles = []
        self.particles = []
        self.stars = []
        
        self.camera_y = 0.0
        self.scroll_y_total = 0.0
        self.scroll_speed_base = 1.0
        self.scroll_speed_current = 1.0

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_pos = pygame.Vector2(self.WIDTH // 2, self.HEIGHT - 50)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_on_platform = True
        self.platform_player_is_on = None

        self.camera_y = 0.0
        self.scroll_y_total = 0.0
        self.scroll_speed_current = self.scroll_speed_base

        # Procedural generation
        self.platforms = []
        self.obstacles = []
        self.particles = []
        self._generate_initial_platforms()
        self.stars = [
            (
                self.np_random.integers(0, self.WIDTH),
                self.np_random.integers(0, self.HEIGHT),
                self.np_random.uniform(0.5, 2.0),
            )
            for _ in range(150)
        ]
        
        return self._get_observation(), self._get_info()

    def _generate_initial_platforms(self):
        # Start platform
        start_plat = pygame.Rect(self.WIDTH // 2 - 50, self.HEIGHT - 30, 100, 20)
        self.platforms.append(start_plat)
        self.player_on_platform = True
        self.platform_player_is_on = start_plat
        self.player_pos.y = start_plat.top - 10

        # Fill the screen with platforms
        y = self.HEIGHT - 120
        while y > -50:
            self._generate_platform_row(y)
            y -= self.np_random.integers(80, 150)
            
    def _generate_platform_row(self, y_pos):
        num_platforms = self.np_random.integers(1, 4)
        x_positions = self.np_random.choice(range(50, self.WIDTH - 150, 100), num_platforms, replace=False)
        for x in x_positions:
            width = self.np_random.integers(70, 150)
            plat = pygame.Rect(x, y_pos, width, 20)
            self.platforms.append(plat)
            # Chance to spawn an obstacle on a platform
            if self.np_random.random() < 0.15 and self.scroll_y_total > 500:
                obs_x = x + width // 2
                obs_y = y_pos - 15
                self.obstacles.append(pygame.Vector2(obs_x, obs_y))
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.01  # Survival reward

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        # --- Action Handling ---
        if self.player_on_platform:
            jump_power = self.JUMP_BASE_POWER
            side_power = 0
            
            if space_held:
                jump_power *= self.POWER_JUMP_MOD
            
            # Movement determines jump direction
            if movement == 0: # None
                pass # Simple vertical jump
            elif movement == 1: # Up
                jump_power *= 1.1 # Slightly higher jump
            elif movement == 2: # Down
                # Drop through platform
                self.player_pos.y += 5 
                self.platform_player_is_on = None 
                jump_power = 0
            elif movement == 3: # Left
                side_power = -self.JUMP_SIDE_POWER
            elif movement == 4: # Right
                side_power = self.JUMP_SIDE_POWER

            # Shift dash modifier
            if shift_held:
                jump_power *= 0.7 # Less height, more distance
                side_power *= self.DASH_MOD
                if movement == 1: # Up-dash
                    jump_power *= self.DASH_MOD
                # sound: dash_whoosh.wav

            self.player_vel = pygame.Vector2(side_power, jump_power)
            self.player_on_platform = False
            self._spawn_particles(20, self.player_pos + pygame.Vector2(0, 10), (abs(side_power) + abs(jump_power)) * 1.5)
            # sound: jump.wav

        # --- Physics Update ---
        # Air control
        if not self.player_on_platform:
            if movement == 3: # Left
                self.player_vel.x -= self.AIR_CONTROL
            elif movement == 4: # Right
                self.player_vel.x += self.AIR_CONTROL

        # Gravity
        self.player_vel.y += self.GRAVITY
        self.player_vel.y = min(self.player_vel.y, self.MAX_FALL_SPEED)

        # Update position
        self.player_pos += self.player_vel

        # Horizontal screen bounds
        if self.player_pos.x < 10:
            self.player_pos.x = 10
            self.player_vel.x = 0
        if self.player_pos.x > self.WIDTH - 10:
            self.player_pos.x = self.WIDTH - 10
            self.player_vel.x = 0
            
        self.player_rect = pygame.Rect(self.player_pos.x - 10, self.player_pos.y - 10, 20, 20)

        # --- World Scrolling ---
        self.scroll_speed_current = self.scroll_speed_base + (self.scroll_y_total // 500) * 0.05
        
        cam_target_y = self.player_pos.y - self.HEIGHT * 0.6
        self.camera_y += (cam_target_y - self.camera_y) * 0.08
        
        scroll_this_frame = self.scroll_speed_current
        self.camera_y += scroll_this_frame
        self.scroll_y_total += scroll_this_frame

        # --- Collision Detection ---
        # Platforms
        if self.player_vel.y > 0: # Only check for landing if falling
            for plat in self.platforms:
                if plat != self.platform_player_is_on and self.player_rect.colliderect(plat):
                    if self.player_pos.y < plat.top + self.player_vel.y:
                        self.player_on_platform = True
                        self.platform_player_is_on = plat
                        self.player_pos.y = plat.top - 10
                        self.player_vel = pygame.Vector2(0, 0)
                        reward += 1.0
                        self.score += 10
                        self._spawn_particles(5, self.player_pos + pygame.Vector2(0, 10), 3)
                        # sound: land.wav
                        break
        
        # Obstacles
        for obs_pos in self.obstacles:
            if self.player_rect.collidepoint(obs_pos.x, obs_pos.y - self.camera_y):
                self.game_over = True
                reward = -50.0
                self.score -= 500
                self._spawn_particles(50, self.player_pos, 10, self.COLOR_OBSTACLE)
                # sound: explosion.wav
                break

        # --- Entity Management ---
        self._manage_world_entities()

        # --- Termination Conditions ---
        # Fall off screen
        if self.player_pos.y - self.camera_y > self.HEIGHT:
            self.game_over = True
            reward = -50.0
            
        # Win condition
        if self.scroll_y_total >= self.WORLD_HEIGHT:
            self.game_over = True
            self.win = True
            reward = 100.0
            self.score += 10000

        # Max steps
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        self.score += int(scroll_this_frame)
        
        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,
            self._get_info()
        )

    def _manage_world_entities(self):
        # Remove off-screen entities
        self.platforms = [p for p in self.platforms if p.top - self.camera_y < self.HEIGHT]
        self.obstacles = [o for o in self.obstacles if o.y - self.camera_y < self.HEIGHT]

        # Generate new platforms at the top
        last_platform_y = min([p.y for p in self.platforms] or [self.HEIGHT])
        if last_platform_y - self.camera_y > -50:
            self._generate_platform_row(last_platform_y - self.np_random.integers(80, 150))
    
    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['radius'] -= 0.1
            if p['lifetime'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)
            else:
                screen_pos = p['pos'] - pygame.Vector2(0, self.camera_y)
                alpha = max(0, min(255, int(255 * (p['lifetime'] / p['start_life']))))
                color = (*p['color'], alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), int(p['radius']), color)

    def _spawn_particles(self, count, pos, power, color=None):
        if color is None:
            color = self.COLOR_PLAYER_GLOW
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 1.0) * power
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifetime = self.np_random.integers(15, 40)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifetime': lifetime,
                'start_life': lifetime,
                'radius': self.np_random.uniform(2, 5),
                'color': color
            })
            
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw stars with parallax
        for x, y, speed in self.stars:
            screen_y = (y - self.camera_y * (0.1 * speed)) % self.HEIGHT
            size = int(speed)
            pygame.draw.rect(self.screen, (200, 200, 255), (x, screen_y, size, size))

        # Draw obstacles
        for obs_pos in self.obstacles:
            screen_pos = obs_pos - pygame.Vector2(0, self.camera_y)
            if -20 < screen_pos.y < self.HEIGHT + 20:
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), 10, self.COLOR_OBSTACLE)
                pygame.gfxdraw.aacircle(self.screen, int(screen_pos.x), int(screen_pos.y), 10, self.COLOR_OBSTACLE)
                pygame.gfxdraw.aacircle(self.screen, int(screen_pos.x), int(screen_pos.y), 15, self.COLOR_OBSTACLE_GLOW + (100,))

        # Draw platforms
        for plat in self.platforms:
            screen_rect = plat.move(0, -self.camera_y)
            if -20 < screen_rect.y < self.HEIGHT:
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, screen_rect, border_radius=5)
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM_OUTLINE, screen_rect, 2, border_radius=5)
        
        # Draw particles
        self._update_and_draw_particles()

        # Draw player
        if not (self.game_over and not self.win):
            screen_pos = self.player_pos - pygame.Vector2(0, self.camera_y)
            # Player glow
            glow_radius = 15 + math.sin(self.steps * 0.1) * 3
            pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), int(glow_radius), self.COLOR_PLAYER_GLOW + (50,))
            pygame.gfxdraw.aacircle(self.screen, int(screen_pos.x), int(screen_pos.y), int(glow_radius), self.COLOR_PLAYER_GLOW + (100,))
            
            # Player body
            player_points = [
                (screen_pos.x, screen_pos.y - 12),
                (screen_pos.x - 10, screen_pos.y + 8),
                (screen_pos.x + 10, screen_pos.y + 8),
            ]
            pygame.gfxdraw.aapolygon(self.screen, player_points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, player_points, self.COLOR_PLAYER)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        progress = min(100, int((self.scroll_y_total / self.WORLD_HEIGHT) * 100))
        progress_text = self.font.render(f"PROGRESS: {progress}%", True, self.COLOR_TEXT)
        self.screen.blit(progress_text, (self.WIDTH - progress_text.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win else "GAME OVER"
            end_text = self.game_over_font.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "progress_percent": (self.scroll_y_total / self.WORLD_HEIGHT) * 100
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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


# Example of how to run the environment
if __name__ == '__main__':
    # For human play
    import sys
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human interaction
    pygame.display.set_caption("Vertical Hopper")
    real_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    running = True
    
    # Action state
    movement = 0
    space_held = 0
    shift_held = 0
    
    print(env.user_guide)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            
            # Update action state based on keys
            movement = 0 # Default no-op
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            elif keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
                
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Run at 30 FPS
        
    env.close()
    pygame.quit()
    sys.exit()